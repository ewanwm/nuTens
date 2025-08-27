#include <gtest/gtest.h>
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <tests/barger-propagator.hpp>
#include <nuTens/propagator/const-density-solver.hpp>

using namespace nuTens;
using namespace nuTens::testing;

class TwoFlavourOscillations :public gtest::TestWithParam<float> {
protected:

    float theta;

    float m1;
    float m2;
    float energy;
    float baseline;
    float density;
    Tensor masses;
    Tensor energies;

    // set up common values to use across tests
    void SetUp() {

        m1 = 0.0;
        m2 = 0.008 * units::eV * units::eV;
        energy = 0.5 * units::GeV;
        baseline = 295.0 * units::km;

        masses = Tensor({m1, m2}, dtypes::kComplexFloat).addBatchDim();

        energies = Tensor::ones({1, 1}, dtypes::kComplexFloat).requiresGrad(false);
        energies.setValue({0, 0}, energy);

        density = 2.6;
    }

    void testConstDensity(bool antiNu) {

        // get parameterised theta value
        float theta = GetParam();

        std::cout << "\n#### const density test for theta = " << theta << " ####" << std::endl;
       
        Propagator tensorPropagator(2, baseline);
        auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(2, density);
        
        TwoFlavourBarger bargerProp{};

        bargerProp.setParams(m1, m2, theta, baseline, density, antiNu);
        
        std::cout << "lm():        " << bargerProp.lm() << std::endl;
        std::cout << "ang:         " << bargerProp.calculateEffectiveAngle(energy) << std::endl;
        std::cout << "dm2:         " << bargerProp.calculateEffectiveDm2(energy) << std::endl;
        std::cout << "off-diag:    " << bargerProp.calculateProb(energy, 0, 1) << std::endl << std::endl;

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
        tensorPropagator.setAntiNeutrino(antiNu);

        tensorPropagator.setEnergies(energies);

        Tensor eigenVals;
        Tensor eigenVecs;

        tensorSolver->calculateEigenvalues(eigenVecs, eigenVals);

        // first check that the effective dM^2 from the tensor solver is what we
        // expect
        std::cout << "tensorSolver eigenvals: " << std::endl;
        std::cout << eigenVals << std::endl;
        auto calcV1 = eigenVals.getValue<float>({0, 0});
        auto calcV2 = eigenVals.getValue<float>({0, 1});
        float effDm2 = (calcV1 - calcV2) * 2.0 * energy;

        ASSERT_NEAR(effDm2, bargerProp.calculateEffectiveDm2(energy),
                    0.00001);

        // now check the actual mixing matrix entries
        Tensor PMNSeff = Tensor::matmul(PMNS, eigenVecs);
        std::cout << "effective PMNS: " << std::endl;
        std::cout << "[0,0] :: tensor solver: " << PMNSeff.getValue<float>({0, 0, 0}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 0, 0) << std::endl;
        std::cout << "[0,1] :: tensor solver: " << PMNSeff.getValue<float>({0, 0, 1}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 0, 1) << std::endl;
        std::cout << "[1,0] :: tensor solver: " << PMNSeff.getValue<float>({0, 1, 0}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 1, 0) << std::endl;
        std::cout << "[1,1] :: tensor solver: " << PMNSeff.getValue<float>({0, 1, 1}) << " :: barger: " <<  bargerProp.getPMNSelement(energy, 1, 1) << std::endl;

        ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 0, 0})), std::abs(bargerProp.getPMNSelement(energy, 0, 0)),
                    0.00001);

        ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 1, 1})), std::abs(bargerProp.getPMNSelement(energy, 1, 1)),
                    0.00001);

        ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 0, 1})), std::abs(bargerProp.getPMNSelement(energy, 0, 1)),
                    0.00001);

        ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 1, 0})), std::abs(bargerProp.getPMNSelement(energy, 1, 0)),
                    0.00001);

        Tensor probabilities = tensorPropagator.calculateProbs();
        std::cout << "Oscillation probabilities:" << std::endl;
        std::cout << "[0,0] :: tensor solver: " << probabilities.getValue<float>({0, 0, 0}) << " :: barger: " <<  bargerProp.calculateProb(energy, 0, 0) << std::endl;
        std::cout << "[0,1] :: tensor solver: " << probabilities.getValue<float>({0, 0, 1}) << " :: barger: " <<  bargerProp.calculateProb(energy, 0, 1) << std::endl;
        std::cout << "[1,0] :: tensor solver: " << probabilities.getValue<float>({0, 1, 0}) << " :: barger: " <<  bargerProp.calculateProb(energy, 1, 0) << std::endl;
        std::cout << "[1,1] :: tensor solver: " << probabilities.getValue<float>({0, 1, 1}) << " :: barger: " <<  bargerProp.calculateProb(energy, 1, 1) << std::endl;


        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0),
                    0.00001);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1),
                    0.00001);

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1),
                    0.00001);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0),
                    0.00001);

    }
};

// test that Propagator gives expected oscillation probabilites for a range
// of thetas
TEST_P(TwoFlavourOscillations, VacuumOscProbs) {

    // get parameterised theta value
    float theta = GetParam();

    std::cout << "\n#### vacuum test for theta = " << theta << " ####" << std::endl;   

    Propagator tensorPropagator(2, baseline);
    tensorPropagator.setMasses(masses);

    // will use this for baseline for comparisons
    TwoFlavourBarger bargerProp{};

    bargerProp.setParams(m1, m2, theta, baseline);

    // construct the mixing matrix for current theta value
    Tensor PMNS = Tensor::ones({1, 2, 2}, dtypes::kComplexFloat).requiresGrad(false);
    PMNS.setValue({0, 0, 0}, std::cos(theta));
    PMNS.setValue({0, 0, 1}, -std::sin(theta));
    PMNS.setValue({0, 1, 0}, std::sin(theta));
    PMNS.setValue({0, 1, 1}, std::cos(theta));

    tensorPropagator.setMixingMatrix(PMNS);

    tensorPropagator.setEnergies(energies);

    Tensor probabilities = tensorPropagator.calculateProbs();

    ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0),
                0.00001);

    ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1),
                0.00001);

    ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1),
                0.00001);

    ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0),
                0.00001);
}

// test const density matter oscillations
TEST_P(TwoFlavourOscillations, ConstDensityOscProbsNu) {

    testConstDensity(/*antiNu=*/false);
}

// test const density matter oscillations for anti-neutrinos
TEST_P(TwoFlavourOscillations, ConstDensityOscProbsAntiNu) {
    
    testConstDensity(/*antiNu=*/true);
}

INSTANTIATE_TEST_CASE_P(
    OscProb,
    TwoFlavourOscillations,
    ::testing::Values(
        -M_PI, -0.8*M_PI, -0.6*M_PI, -0.4*M_PI, -0.2*M_PI, 0.0, 0.2*M_PI, 0.4*M_PI, 0.6*M_PI, 0.8*M_PI, M_PI
));
