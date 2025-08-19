#include <nuTens/propagator/propagator.hpp>
#include <nuTens/propagator/DP-propagator.hpp>
#include <nuTens/propagator/const-density-solver.hpp>
#include <tests/barger-propagator.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/logging.hpp>

// nuFast c++ implementation
#include <tests/nuFast.hpp>

using namespace nuTens;
using namespace testing;


constexpr std::complex<float> imagUnit(0.0, 1.0);

class PMNSmatrix
{
  public:
    PMNSmatrix()
    {
        // set up the three matrices to build the mixing matrix
        _m1 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false);
        _m2 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false);
        _m3 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false);
    }

    void build(const Tensor &theta12, const Tensor &theta13, const Tensor &theta23, const Tensor &deltaCP)
    {
        _m1.setValue({0, 0, 0}, Tensor({1.0}));
        _m1.setValue({0, 1, 1}, Tensor::cos(theta23));
        _m1.setValue({0, 1, 2}, Tensor::sin(theta23));
        _m1.setValue({0, 2, 1}, -Tensor::sin(theta23));
        _m1.setValue({0, 2, 2}, Tensor::cos(theta23));

        std::cout << "_m1: " << _m1 << std::endl;

        _m2.setValue({0, 1, 1}, Tensor({1.0}));
        _m2.setValue({0, 0, 0}, Tensor::cos(theta13));
        _m2.setValue({0, 0, 2}, Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(deltaCP, -imagUnit))));
        _m2.setValue({0, 2, 0}, -Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(deltaCP, imagUnit))));
        _m2.setValue({0, 2, 2}, Tensor::cos(theta13));

        std::cout << "_m2: " << _m2 << std::endl;

        _m3.setValue({0, 2, 2}, Tensor({1.0}));
        _m3.setValue({0, 0, 0}, Tensor::cos(theta12));
        _m3.setValue({0, 0, 1}, Tensor::sin(theta12));
        _m3.setValue({0, 1, 0}, -Tensor::sin(theta12));
        _m3.setValue({0, 1, 1}, Tensor::cos(theta12));

        std::cout << "_m3: " << _m3 << std::endl;

        // Build PMNS
        matrix = Tensor::matmul(_m1, Tensor::matmul(_m2, _m3));
    }

    Tensor matrix;

  private:
    Tensor _m1;
    Tensor _m2;
    Tensor _m3;
};


int main()
{
    NT_PROFILE_BEGINSESSION("DP-propagator-test");

    NT_PROFILE();

    float m1 = 0.0001 * units::eV;
    float m2 = 0.008 * units::eV;
    float m3 = 0.02 * units::eV;
    float energy = 0.5 * units::GeV;
    float baseline = 295.0 * units::km;
    float density = 2.6;

    // set the tensors we will use to calculate matter eigenvalues
    Tensor masses = Tensor({m1, m2, m3}, dtypes::kComplexFloat).addBatchDim().requiresGrad(true);

    auto theta23 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false);
    auto theta13 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false);
    auto theta12 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false);
    auto deltaCP = Tensor::zeros({1}).dType(dtypes::kComplexFloat).requiresGrad(false);

    Tensor dmsq21 = Tensor({m1 * m1 - m2 * m2}, dtypes::kComplexFloat).requiresGrad(true);
    Tensor dmsq31 = Tensor({m1 * m1 - m3 * m3}, dtypes::kComplexFloat).requiresGrad(true);

    Tensor energies = Tensor::ones({1, 1}, dtypes::kComplexFloat).requiresGrad(false).hasBatchDim(true);
    energies.setValue({0, 0}, energy);

    Propagator tensorPropagator(3, baseline);
    auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(3, density);
    tensorPropagator.setMatterSolver(tensorSolver);
    tensorPropagator.setMasses(masses);
    tensorPropagator.setEnergies(energies);

    DPpropagator dpPropagator(baseline, false, density, 10);
    dpPropagator.setEnergies(energies);

    PMNSmatrix pmns;

    ThreeFlavourBarger barger;
    
    // test that Propagator gives expected oscillation probabilites for a range
    // of thetas
    for (int i = 0; i <= 20; i++)
    {
        float theta = (-1.0 + 2.0 * (float)i / 20.0) * 0.5 * M_PI;

        std::cout << "########## theta = " << theta << " ##########" << std::endl;

        theta23.setValue(theta, 0);
        theta13.setValue(0.8, 0);
        theta12.setValue(0.8, 0);

        deltaCP.setValue({0}, M_PI / 2.0);

        // calculate new values of the mixing matrix
        pmns.build(theta12, theta13, theta23, deltaCP);

        tensorPropagator.setMixingMatrix(pmns.matrix);
        dpPropagator.setParameters(theta12, theta23, theta13, deltaCP, dmsq21, dmsq31);
        barger.setParams(
            masses.getValue<float>({0, 0}), masses.getValue<float>({0, 1}), masses.getValue<float>({0, 2}), 
            theta12.getValue(0), theta13.getValue(0), theta23.getValue(0), 
            deltaCP.getValue<float>(),
            baseline, density, /*antiNeutrino=*/false
        );

        // #######################################################################################
        std:: cout << std::endl << "---------------------------------------------------------" << std::endl;

        Tensor eigenVals;
        Tensor eigenVecs;
        tensorSolver->calculateEigenvalues(eigenVecs, eigenVals);

        std::cout << "  Re[evecs]: " << eigenVecs.real() << std::endl;
        std::cout << "  Im[evecs]: " << eigenVecs.imag() << std::endl;

        // first check that the effective dM^2 from the tensor solver is what we
        // expect
        std::cout << "tensorSolver eigenvals: " << std::endl;
        std::cout << eigenVals << std::endl;
        auto calcV1 = eigenVals.getValue<float>({0, 0});
        auto calcV2 = eigenVals.getValue<float>({0, 1});
        auto calcV3 = eigenVals.getValue<float>({0, 2});

        float effDm21sq = (calcV2 - calcV1) * 2.0 * energy;
        float effDm31sq = (calcV3 - calcV1) * 2.0 * energy;

        std::cout << "tensor eff M1^2: " << calcV1 * 2.0 * energy << std::endl;
        std::cout << "tensor eff M2^2: " << calcV2 * 2.0 * energy << std::endl;
        std::cout << "tensor eff M3^2: " << calcV3 * 2.0 * energy << std::endl;

        std::cout << "tensor eff Dm21^2: " << effDm21sq << std::endl;
        std::cout << "tensor eff Dm31^2: " << effDm31sq << std::endl;

        std::cout << std::endl;

        Tensor lambda1, lambda2, lambda3, Dlambda21, Dlambda31, Dlambda32;
        dpPropagator.calculateEigenvalues(lambda1, lambda2, lambda3, Dlambda21, Dlambda31, Dlambda32);

        std::cout << "DP eff M1^2: " << lambda1.getValue<float>({0}) << std::endl;
        std::cout << "DP eff M2^2: " << lambda2.getValue<float>({0}) << std::endl;
        std::cout << "DP eff M3^2: " << lambda3.getValue<float>({0}) << std::endl;

        std::cout << "DP eff Dm21^2: " << Dlambda21.getValue<float>({0}) << std::endl;
        std::cout << "DP eff Dm31^2: " << Dlambda31.getValue<float>({0}) << std::endl;

        std::cout << std::endl;

        std::cout << "Barger eff M1^2: " << barger.calculateEffectiveM2(energy, 0) << std::endl;
        std::cout << "Barger eff M2^2: " << barger.calculateEffectiveM2(energy, 1) << std::endl;
        std::cout << "Barger eff M3^2: " << barger.calculateEffectiveM2(energy, 2) << std::endl;

        std::cout << "Barger eff Dm21^2: " << barger.calculateEffectiveM2(energy, 1) - barger.calculateEffectiveM2(energy, 0) << std::endl;
        std::cout << "Barger eff Dm31^2: " << barger.calculateEffectiveM2(energy, 2) - barger.calculateEffectiveM2(energy, 0) << std::endl;

        
        // ##########################################################################################

        Tensor probabilities = tensorPropagator.calculateProbs();
        Tensor dpProbabilities = dpPropagator.calculateProbs();
        
        std:: cout << "---------------------------------------------------------" << std::endl << std::endl;

        std::cout << "probs:" << probabilities << std::endl;
        std::cout << "dpProbs:" << dpProbabilities << std::endl;

        double probs_returned[3][3];
        Probability_Matter_LBL(
            std::sin(theta12.getValue(0)) * std::sin(theta12.getValue(0)),
            std::sin(theta13.getValue(0)) * std::sin(theta13.getValue(0)),
            std::sin(theta23.getValue(0)) * std::sin(theta23.getValue(0)), 
            deltaCP.getValue<float>(), 
            m1 * m1 - m2 * m2,
            m1 * m1 - m3 * m3,
            baseline / units::km,
            energies.getValue<float>() / units::GeV, 
            1.0, 
            density,
            10, 
            &probs_returned
        );
        
        NT_INFO("[0, 0] :: propagator: {0:.4f} :: DP propagator: {1:.4f} :: nuFast: {2:.4f}", probabilities.getValue<float>({0, 0, 0}), dpProbabilities.getValue<float>({0, 0, 0}), probs_returned[0][0]);
        NT_INFO("[0, 1] :: propagator: {0:.4f} :: DP propagator: {1:.4f} :: nuFast: {2:.4f}", probabilities.getValue<float>({0, 0, 1}), dpProbabilities.getValue<float>({0, 1, 0}), probs_returned[1][0]);
        NT_INFO("[0, 2] :: propagator: {0:.4f} :: DP propagator: {1:.4f} :: nuFast: {2:.4f}", probabilities.getValue<float>({0, 0, 2}), dpProbabilities.getValue<float>({0, 2, 0}), probs_returned[2][0]);
        NT_INFO("[1, 0] :: propagator: {0:.4f} :: DP propagator: {1:.4f} :: nuFast: {2:.4f}", probabilities.getValue<float>({0, 1, 0}), dpProbabilities.getValue<float>({0, 0, 1}), probs_returned[0][1]);
        NT_INFO("[1, 1] :: propagator: {0:.4f} :: DP propagator: {1:.4f} :: nuFast: {2:.4f}", probabilities.getValue<float>({0, 1, 1}), dpProbabilities.getValue<float>({0, 1, 1}), probs_returned[1][1]);
        NT_INFO("[1, 2] :: propagator: {0:.4f} :: DP propagator: {1:.4f} :: nuFast: {2:.4f}", probabilities.getValue<float>({0, 1, 2}), dpProbabilities.getValue<float>({0, 2, 1}), probs_returned[2][1]);
        NT_INFO("[2, 0] :: propagator: {0:.4f} :: DP propagator: {1:.4f} :: nuFast: {2:.4f}", probabilities.getValue<float>({0, 2, 0}), dpProbabilities.getValue<float>({0, 0, 2}), probs_returned[0][2]);
        NT_INFO("[2, 1] :: propagator: {0:.4f} :: DP propagator: {1:.4f} :: nuFast: {2:.4f}", probabilities.getValue<float>({0, 2, 1}), dpProbabilities.getValue<float>({0, 1, 2}), probs_returned[1][2]);
        NT_INFO("[2, 2] :: propagator: {0:.4f} :: DP propagator: {1:.4f} :: nuFast: {2:.4f}", probabilities.getValue<float>({0, 2, 2}), dpProbabilities.getValue<float>({0, 2, 2}), probs_returned[2][2]);
        
        // TEST_EXPECTED(probabilities.getValue<float>({0, 0, 0}), dpProbabilities.getValue<float>({0, 0, 0}),
        //               "probability for alpha == beta == 0", 0.00001)

        // TEST_EXPECTED(probabilities.getValue<float>({0, 0, 1}), dpProbabilities.getValue<float>({0, 0, 1}),
        //               "probability for alpha == 0, beta == 1", 0.00001)

        // TEST_EXPECTED(probabilities.getValue<float>({0, 0, 2}), dpProbabilities.getValue<float>({0, 0, 2}),
        //               "probability for alpha == 0, beta == 2", 0.00001)

        // TEST_EXPECTED(probabilities.getValue<float>({0, 1, 0}), dpProbabilities.getValue<float>({0, 1, 0}),
        //               "probability for alpha == 1, beta == 1", 0.00001)

        // TEST_EXPECTED(probabilities.getValue<float>({0, 1, 1}), dpProbabilities.getValue<float>({0, 1, 1}),
        //               "probability for alpha == beta == 1", 0.00001)

        // TEST_EXPECTED(probabilities.getValue<float>({0, 1, 2}), dpProbabilities.getValue<float>({0, 1, 2}),
        //               "probability for alpha == 1, beta == 2", 0.00001)

        // TEST_EXPECTED(probabilities.getValue<float>({0, 2, 0}), dpProbabilities.getValue<float>({0, 2, 0}),
        //               "probability for alpha == 2, beta == 0", 0.00001)

        // TEST_EXPECTED(probabilities.getValue<float>({0, 2, 1}), dpProbabilities.getValue<float>({0, 2, 1}),
        //               "probability for alpha == 2, beta == 1", 0.00001)

        // TEST_EXPECTED(probabilities.getValue<float>({0, 2, 2}), dpProbabilities.getValue<float>({0, 2, 2}),
        //               "probability for alpha == beta == 2", 0.00001)


        std::cout << "###############################" << std::endl << std::endl;
    }

    NT_PROFILE_ENDSESSION();
}