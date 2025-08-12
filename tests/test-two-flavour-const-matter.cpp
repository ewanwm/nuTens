#include <nuTens/propagator/propagator.hpp>
#include <nuTens/propagator/const-density-solver.hpp>
#include <tests/barger-propagator.hpp>
#include <tests/test-utils.hpp>

using namespace nuTens;
using namespace testing;

int main()
{
    NT_PROFILE_BEGINSESSION("two-flavour-const-matter-test");

    NT_PROFILE();

    float m1 = 0.0;
    float m2 = 0.008 * units::eV * units::eV;
    float energy = 0.5 * units::GeV;
    float baseline = 295.0 * units::km;
    float density = 2.6;

    // set the tensors we will use to calculate matter eigenvalues
    Tensor masses = Tensor({m1, m2}, dtypes::kFloat).addBatchDim().requiresGrad(true);

    Tensor energies = Tensor::ones({1, 1}, dtypes::kFloat).requiresGrad(false).hasBatchDim(true);
    energies.setValue({0, 0}, energy);
    energies.requiresGrad(true);

    auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(2, density);

    Propagator tensorPropagator(2, baseline);
    
    TwoFlavourBarger bargerProp{};

    // test that Propagator gives expected oscillation probabilites for a range
    // of thetas
    for (int i = 0; i <= 20; i++)
    {
        float theta = (-1.0 + 2.0 * (float)i / 20.0) * 0.49 * M_PI;

        bargerProp.setParams(m1, m2, theta, baseline, density);

        // construct the mixing matrix for current theta value
        Tensor PMNS = Tensor::ones({1, 2, 2}, dtypes::kComplexFloat).requiresGrad(false);
        PMNS.setValue({0, 0, 0}, std::cos(theta));
        PMNS.setValue({0, 0, 1}, std::sin(theta));
        PMNS.setValue({0, 1, 0}, -std::sin(theta));
        PMNS.setValue({0, 1, 1}, std::cos(theta));
        PMNS.requiresGrad(true);

        tensorPropagator.setMatterSolver(tensorSolver);
        tensorPropagator.setMixingMatrix(PMNS);
        tensorPropagator.setMasses(masses);

        tensorPropagator.setEnergies(energies);

        Tensor eigenVals;
        Tensor eigenVecs;
        tensorSolver->calculateEigenvalues(eigenVecs, eigenVals);

        std::cout << "######## theta = " << theta << " ########" << std::endl;

        // first check that the effective dM^2 from the tensor solver is what we
        // expect
        std::cout << "tensorSolver eigenvals: " << std::endl;
        std::cout << eigenVals << std::endl;
        auto calcV1 = eigenVals.getValue<float>({0, 0});
        auto calcV2 = eigenVals.getValue<float>({0, 1});
        float effDm2 = (calcV1 - calcV2) * 2.0 * energy;

        TEST_EXPECTED(effDm2, bargerProp.calculateEffectiveDm2(energy),
                      "effective dM^2 for theta == " + std::to_string(theta), 0.00001)

        // now check the actual mixing matrix entries
        Tensor PMNSeff = Tensor::matmul(PMNS, eigenVecs);
        std::cout << "effective PMNS: " << std::endl;
        std::cout << "[0,0] :: tensor solver: " << PMNSeff.getValue<float>({0, 0, 0}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 0, 0) << std::endl;
        std::cout << "[0,1] :: tensor solver: " << PMNSeff.getValue<float>({0, 0, 1}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 0, 1) << std::endl;
        std::cout << "[1,0] :: tensor solver: " << PMNSeff.getValue<float>({0, 1, 0}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 1, 0) << std::endl;
        std::cout << "[1,1] :: tensor solver: " << PMNSeff.getValue<float>({0, 1, 1}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 1, 1) << std::endl;

        TEST_EXPECTED(std::abs(PMNSeff.getValue<float>({0, 0, 0})), std::abs(bargerProp.getPMNSelement(energy, 0, 0)),
                      "PMNS[0,0] for theta == " + std::to_string(theta), 0.00001)

        TEST_EXPECTED(std::abs(PMNSeff.getValue<float>({0, 1, 1})), std::abs(bargerProp.getPMNSelement(energy, 1, 1)),
                      "PMNS[1,1] for theta == " + std::to_string(theta), 0.00001)

        TEST_EXPECTED(std::abs(PMNSeff.getValue<float>({0, 0, 1})), std::abs(bargerProp.getPMNSelement(energy, 0, 1)),
                      "PMNS[0,1] for theta == " + std::to_string(theta), 0.00001)

        TEST_EXPECTED(std::abs(PMNSeff.getValue<float>({0, 1, 0})), std::abs(bargerProp.getPMNSelement(energy, 1, 0)),
                      "PMNS[1,0] for theta == " + std::to_string(theta), 0.00001)



        Tensor probabilities = tensorPropagator.calculateProbs();
        std::cout << "Oscillation probabilities:" << std::endl;
        std::cout << "[0,0] :: tensor solver: " << probabilities.getValue<float>({0, 0, 0}) << " :: barger: " <<  bargerProp.calculateProb(energy, 0, 0) << std::endl;
        std::cout << "[0,1] :: tensor solver: " << probabilities.getValue<float>({0, 0, 1}) << " :: barger: " <<  bargerProp.calculateProb(energy, 0, 1) << std::endl;
        std::cout << "[1,0] :: tensor solver: " << probabilities.getValue<float>({0, 1, 0}) << " :: barger: " <<  bargerProp.calculateProb(energy, 1, 0) << std::endl;
        std::cout << "[1,1] :: tensor solver: " << probabilities.getValue<float>({0, 1, 1}) << " :: barger: " <<  bargerProp.calculateProb(energy, 1, 1) << std::endl;


        TEST_EXPECTED(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0),
                      "probability for alpha == beta == 0", 0.00001)

        TEST_EXPECTED(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1),
                      "probability for alpha == beta == 1", 0.00001)

        TEST_EXPECTED(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1),
                      "probability for alpha == 0, beta == 1", 0.00001)

        TEST_EXPECTED(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0),
                      "probability for alpha == 1, beta == 0", 0.00001)

        std::cout << "###############################" << std::endl << std::endl;
    }

    NT_PROFILE_ENDSESSION();
}