#include <gtest/gtest.h>
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <tests/barger-propagator.hpp>
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/pmns-matrix.hpp>

using namespace nuTens;
using namespace nuTens::testing;

class ThreeFlavourOscillations :public gtest::TestWithParam<float> {
protected:

    float theta12;
    float theta13;
    float theta23;
    float deltaCP;

    float m1;
    float m2;
    float m3;

    float energy;
    float baseline;
    float density;
    Tensor masses;
    Tensor energies;

    // set up common values to use across tests
    void SetUp() {

        m1 = 0.0;
        m2 = 0.008 * units::eV * units::eV;
        m3 = 0.01 * units::eV * units::eV;

        theta23 = 0.23 * M_PI;
        theta13 = 0.13 * M_PI;
        deltaCP = 0.25 * M_PI;

        energy = 0.5 * units::GeV;
        baseline = 295.0 * units::km;

        masses = Tensor({m1, m2, m3}, dtypes::kComplexDouble).addBatchDim();

        energies = Tensor::ones({1, 1}, dtypes::kComplexDouble).requiresGrad(false);
        energies.setValue({0, 0}, energy);

        density = 2.6;
    }

    void testConstDensity(bool antiNu) {

        // get parameterised theta value
        float theta12 = GetParam();

        NT_INFO("\n#### const density test for theta12 = {} ####", theta12);
       
        Propagator tensorPropagator(3, baseline);
        auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(3, density);
        
        ThreeFlavourBarger bargerProp{};

        bargerProp.setParams(m1, m2, m3, theta12, theta13, theta23, deltaCP, baseline, density, antiNu);
        
        NT_INFO("alpha():  {}", bargerProp.alpha(energy));
        NT_INFO("beta():   {}", bargerProp.beta(energy));
        NT_INFO("gamma():  {}", bargerProp.gamma(energy));
        NT_INFO("");

        // construct the mixing matrix for current theta value
        PMNSmatrix pmns;
        
        pmns.setParameterValues(theta12, theta13, theta23, deltaCP);
        
        Tensor pmnsTensor = pmns.build();

        Tensor outerTensor = Tensor::outer(pmnsTensor.getValues({0,0,"..."}), pmnsTensor.getValues({0,0,"..."}).conj());

        tensorPropagator.setMatterSolver(tensorSolver);
        tensorPropagator.setMixingMatrix(pmns.build());
        tensorPropagator.setMasses(masses);
        tensorPropagator.setAntiNeutrino(antiNu);

        tensorPropagator.setEnergies(energies);

        std::cout << "Re[PMNS]:\n" << pmns.build().real() << std::endl;
        std::cout << "Im[PMNS]:\n" << pmns.build().imag() << std::endl << std::endl;

        Tensor eigenVals;
        Tensor eigenVecs;

        tensorSolver->calculateEigenvalues(eigenVecs, eigenVals);

        // first check that the effective dM^2 from the tensor solver is what we
        // expect
        auto calcV1 = eigenVals.getValue<float>({0, 0}) * 2.0 * energy;
        auto calcV2 = eigenVals.getValue<float>({0, 1}) * 2.0 * energy;
        auto calcV3 = eigenVals.getValue<float>({0, 2}) * 2.0 * energy;
        
        NT_INFO("Effective masses:");
        NT_INFO("M1: tensor solver: {:.7f} :: barger: {:.7f}", calcV1, bargerProp.calculateEffectiveM2(energy, 0));
        NT_INFO("M2: tensor solver: {:.7f} :: barger: {:.7f}", calcV2, bargerProp.calculateEffectiveM2(energy, 1));
        NT_INFO("M3: tensor solver: {:.7f} :: barger: {:.7f}", calcV3, bargerProp.calculateEffectiveM2(energy, 2));
        NT_INFO("");
        NT_INFO("Effective dM^2's:");
        NT_INFO("dM^2_21: tensor solver: {:.7f} :: barger: {:.7f}", calcV2 - calcV1, bargerProp.calculateEffectiveM2(energy, 1) - bargerProp.calculateEffectiveM2(energy, 0));
        NT_INFO("dM^2_32: tensor solver: {:.7f} :: barger: {:.7f}", calcV3 - calcV2, bargerProp.calculateEffectiveM2(energy, 2) - bargerProp.calculateEffectiveM2(energy, 1));
        NT_INFO("dM^2_31: tensor solver: {:.7f} :: barger: {:.7f}", calcV3 - calcV1, bargerProp.calculateEffectiveM2(energy, 2) - bargerProp.calculateEffectiveM2(energy, 0));
        

        Tensor hamiltonianTensor = tensorSolver->getHamiltonian();
        NT_INFO("");
        NT_INFO("Hamiltonian:");
        NT_INFO("[0,0] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 0, 0}), hamiltonianTensor.imag().getValue<float>({0, 0, 0}), bargerProp.getHamiltonianElement(energy, 0, 0).real(), bargerProp.getHamiltonianElement(energy, 0, 0).imag());
        NT_INFO("[0,1] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 0, 1}), hamiltonianTensor.imag().getValue<float>({0, 0, 1}), bargerProp.getHamiltonianElement(energy, 0, 1).real(), bargerProp.getHamiltonianElement(energy, 0, 1).imag());
        NT_INFO("[0,2] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 0, 2}), hamiltonianTensor.imag().getValue<float>({0, 0, 2}), bargerProp.getHamiltonianElement(energy, 0, 2).real(), bargerProp.getHamiltonianElement(energy, 0, 2).imag());
        NT_INFO("");
        NT_INFO("[1,0] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 1, 0}), hamiltonianTensor.imag().getValue<float>({0, 1, 0}), bargerProp.getHamiltonianElement(energy, 1, 0).real(), bargerProp.getHamiltonianElement(energy, 1, 0).imag());
        NT_INFO("[1,1] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 1, 1}), hamiltonianTensor.imag().getValue<float>({0, 1, 1}), bargerProp.getHamiltonianElement(energy, 1, 1).real(), bargerProp.getHamiltonianElement(energy, 1, 1).imag());
        NT_INFO("[1,2] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 1, 2}), hamiltonianTensor.imag().getValue<float>({0, 1, 2}), bargerProp.getHamiltonianElement(energy, 1, 2).real(), bargerProp.getHamiltonianElement(energy, 1, 2).imag());
        NT_INFO("");
        NT_INFO("[2,0] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 2, 0}), hamiltonianTensor.imag().getValue<float>({0, 2, 0}), bargerProp.getHamiltonianElement(energy, 2, 0).real(), bargerProp.getHamiltonianElement(energy, 2, 0).imag());
        NT_INFO("[2,1] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 2, 1}), hamiltonianTensor.imag().getValue<float>({0, 2, 1}), bargerProp.getHamiltonianElement(energy, 2, 1).real(), bargerProp.getHamiltonianElement(energy, 2, 1).imag());
        NT_INFO("[2,2] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 2, 2}), hamiltonianTensor.imag().getValue<float>({0, 2, 2}), bargerProp.getHamiltonianElement(energy, 2, 2).real(), bargerProp.getHamiltonianElement(energy, 2, 2).imag());

        NT_INFO("");
        NT_INFO("X[0,0]: ({}, {})", bargerProp.getTransitionMatrixElement(energy, 0,0).real(), bargerProp.getTransitionMatrixElement(energy, 0,0).imag());
        NT_INFO("X[0,1]: ({}, {})", bargerProp.getTransitionMatrixElement(energy, 0,1).real(), bargerProp.getTransitionMatrixElement(energy, 0,1).imag());
        NT_INFO("X[0,2]: ({}, {})", bargerProp.getTransitionMatrixElement(energy, 0,2).real(), bargerProp.getTransitionMatrixElement(energy, 0,2).imag());
        NT_INFO("");
        NT_INFO("X[1,0]: ({}, {})", bargerProp.getTransitionMatrixElement(energy, 1,0).real(), bargerProp.getTransitionMatrixElement(energy, 1,0).imag());
        NT_INFO("X[1,1]: ({}, {})", bargerProp.getTransitionMatrixElement(energy, 1,1).real(), bargerProp.getTransitionMatrixElement(energy, 1,1).imag());
        NT_INFO("X[1,2]: ({}, {})", bargerProp.getTransitionMatrixElement(energy, 1,2).real(), bargerProp.getTransitionMatrixElement(energy, 1,2).imag());
        NT_INFO("");
        NT_INFO("X[2,0]: ({}, {})", bargerProp.getTransitionMatrixElement(energy, 2,0).real(), bargerProp.getTransitionMatrixElement(energy, 2,0).imag());
        NT_INFO("X[2,1]: ({}, {})", bargerProp.getTransitionMatrixElement(energy, 2,1).real(), bargerProp.getTransitionMatrixElement(energy, 2,1).imag());
        NT_INFO("X[2,2]: ({}, {})", bargerProp.getTransitionMatrixElement(energy, 2,2).real(), bargerProp.getTransitionMatrixElement(energy, 2,2).imag());
        NT_INFO("");

        Tensor probabilities = tensorPropagator.calculateProbs();
        NT_INFO("");
        NT_INFO("Oscillation probabilities:");
        NT_INFO("[0,0] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0));
        NT_INFO("[0,1] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1));
        NT_INFO("[0,2] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 0, 2}), bargerProp.calculateProb(energy, 0, 2));
        NT_INFO("");
        NT_INFO("[1,0] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0));
        NT_INFO("[1,1] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1));
        NT_INFO("[1,2] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 1, 2}), bargerProp.calculateProb(energy, 1, 2));
        NT_INFO("");
        NT_INFO("[2,0] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 2, 0}), bargerProp.calculateProb(energy, 2, 0));
        NT_INFO("[2,1] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 2, 1}), bargerProp.calculateProb(energy, 2, 1));
        NT_INFO("[2,2] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 2, 2}), bargerProp.calculateProb(energy, 2, 2));

        // ASSERT_NEAR(effDm2, bargerProp.calculateEffectiveDm2(energy),
        //             0.00001);

        // // now check the actual mixing matrix entries
        // Tensor PMNSeff = Tensor::matmul(PMNS, eigenVecs);
        // std::cout << "effective PMNS: " << std::endl;
        // std::cout << "[0,0] :: tensor solver: " << PMNSeff.getValue<float>({0, 0, 0}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 0, 0) << std::endl;
        // std::cout << "[0,1] :: tensor solver: " << PMNSeff.getValue<float>({0, 0, 1}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 0, 1) << std::endl;
        // std::cout << "[1,0] :: tensor solver: " << PMNSeff.getValue<float>({0, 1, 0}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 1, 0) << std::endl;
        // std::cout << "[1,1] :: tensor solver: " << PMNSeff.getValue<float>({0, 1, 1}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 1, 1) << std::endl;

        // ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 0, 0})), std::abs(bargerProp.getPMNSelement(energy, 0, 0)),
        //             0.00001);

        // ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 1, 1})), std::abs(bargerProp.getPMNSelement(energy, 1, 1)),
        //             0.00001);

        // ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 0, 1})), std::abs(bargerProp.getPMNSelement(energy, 0, 1)),
        //             0.00001);

        // ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 1, 0})), std::abs(bargerProp.getPMNSelement(energy, 1, 0)),
        //             0.00001);



        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0),
                    1e-5);

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1),
                    1e-5);

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 2}), bargerProp.calculateProb(energy, 0, 2),
                    1e-5);


        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0),
                    1e-5);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1),
                    1e-5);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 2}), bargerProp.calculateProb(energy, 1, 2),
                    1e-5);



        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 0}), bargerProp.calculateProb(energy, 2, 0),
                    1e-5);

        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 1}), bargerProp.calculateProb(energy, 2, 1),
                    1e-5);

        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 2}), bargerProp.calculateProb(energy, 2, 2),
                    1e-5);

    }
};

// test that Propagator gives expected oscillation probabilites for a range
// of thetas
// TEST_P(TwoFlavourOscillations, VacuumOscProbs) {
// 
//     // get parameterised theta value
//     float theta = GetParam();
// 
//     std::cout << "\n#### const density test for theta = " << theta << " ####" << std::endl;   
// 
//     Propagator tensorPropagator(2, baseline);
//     tensorPropagator.setMasses(masses);
// 
//     // will use this for baseline for comparisons
//     TwoFlavourBarger bargerProp{};
// 
//     bargerProp.setParams(m1, m2, theta, baseline);
// 
//     // construct the mixing matrix for current theta value
//     Tensor PMNS = Tensor::ones({1, 2, 2}, dtypes::kComplexFloat).requiresGrad(false);
//     PMNS.setValue({0, 0, 0}, std::cos(theta));
//     PMNS.setValue({0, 0, 1}, -std::sin(theta));
//     PMNS.setValue({0, 1, 0}, std::sin(theta));
//     PMNS.setValue({0, 1, 1}, std::cos(theta));
// 
//     tensorPropagator.setMixingMatrix(PMNS);
// 
//     tensorPropagator.setEnergies(energies);
// 
//     Tensor probabilities = tensorPropagator.calculateProbs();
// 
//     ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0),
//                 0.00001);
// 
//     ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1),
//                 0.00001);
// 
//     ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1),
//                 0.00001);
// 
//     ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0),
//                 0.00001);
// }

// test const density matter oscillations
TEST_P(ThreeFlavourOscillations, ConstDensityOscProbsNu) {

    testConstDensity(/*antiNu=*/false);
}

// test const density matter oscillations for anti-neutrinos
TEST_P(ThreeFlavourOscillations, ConstDensityOscProbsAntiNu) {
    
    testConstDensity(/*antiNu=*/true);
}

INSTANTIATE_TEST_CASE_P(
    OscProb,
    ThreeFlavourOscillations,
    ::testing::Values(
        -M_PI, -0.8*M_PI, -0.6*M_PI, -0.4*M_PI, -0.2*M_PI, 0.0, 0.2*M_PI, 0.4*M_PI, 0.6*M_PI, 0.8*M_PI, M_PI
));
