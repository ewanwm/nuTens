#include <gtest/gtest.h>
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <nuTens/propagator/propagator.hpp>
#include <nuTens/propagator/DP-propagator.hpp>
#include <nuTens/propagator/const-density-solver.hpp>
#include <tests/barger-propagator.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/logging.hpp>
#include <nuTens/propagator/pmns-matrix.hpp>

// nuFast c++ implementation
#include <tests/nuFast.hpp>

using namespace nuTens;
using namespace nuTens::testing;

class DPpropagatorTest :public gtest::TestWithParam<float> {

  protected:

    float m1 = 0.0   * units::eV;
    float m2 = 0.008 * units::eV;
    float m3 = 0.02  * units::eV;
    float dcp = M_PI / 4.0;
    float energy = 0.5 * units::GeV;
    float baseline = 295.0 * units::km;
    float density = 0.0; //2.6;

    // set the tensors we will use to calculate matter eigenvalues
    Tensor masses = Tensor({m1, m2, m3}, dtypes::kComplexFloat).addBatchDim().requiresGrad(true);

    Tensor theta23 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor theta13 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor theta12 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor deltaCP = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor dmsq21 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor dmsq31 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

    Tensor energies = Tensor::ones({1, 1}, dtypes::kComplexFloat).requiresGrad(false).hasBatchDim(true);

    Propagator tensorPropagator = Propagator(3, baseline);
    std::shared_ptr<ConstDensityMatterSolver> tensorSolver = std::make_shared<ConstDensityMatterSolver>(3, density);

    DPpropagator dpPropagator = DPpropagator(baseline, false, density, 10);

    PMNSmatrix pmns;

    ThreeFlavourBarger barger;

    void SetUp() {
    
        energies.setValue({0, 0}, energy);

        tensorPropagator.setMatterSolver(tensorSolver);
        tensorPropagator.setMasses(masses);
        tensorPropagator.setEnergies(energies);
        
        dpPropagator.setEnergies(energies);
        dpPropagator.setParameters(theta12, theta23, theta13, deltaCP, dmsq21, dmsq31);
    }

    /// set the oscillation parameter values
    void _setParamValues() {
        // get parameterised theta value
        float theta = GetParam();

        NT_INFO("########## theta = {} ##########", theta);

        theta23.setValue({0}, theta);
        theta13.setValue({0}, 0.3 * M_PI);
        theta12.setValue({0}, 0.2 * M_PI);

        dmsq21.setValue({0}, m1 * m1 - m2 * m2);
        dmsq31.setValue({0}, m1 * m1 - m3 * m3);

        deltaCP.setValue({0}, dcp);

        // calculate new values of the mixing matrix
        pmns.setParameterValues(theta12.getValue<float>({0}), theta13.getValue<float>({0}), theta23.getValue<float>({0}), deltaCP.getValue<float>({0}));
    }

    /// compare DP propagator oscillation probabilities to the "official" nufast code
    void compareNufast(bool antineutrino) {

        _setParamValues();

        dpPropagator.setAntiNeutrino(antineutrino);
        
        // get propagator probabilities
        Tensor dpProbabilities = dpPropagator.calculateProbs();

        // get the nuFast probabilities
        double probs_returned[3][3];
        Probability_Matter_LBL(
            std::sin(theta12.getValue<float>({0})) * std::sin(theta12.getValue<float>({0})),
            std::sin(theta13.getValue<float>({0})) * std::sin(theta13.getValue<float>({0})),
            std::sin(theta23.getValue<float>({0})) * std::sin(theta23.getValue<float>({0})), 
            deltaCP.getValue<float>({0}), 
            m1 * m1 - m2 * m2,
            m1 * m1 - m3 * m3,
            baseline / units::km,
            (0.5 - (float)antineutrino) * 2.0 * energies.getValue<float>() / units::GeV, 
            1.0, 
            density,
            10, 
            &probs_returned
        );

        NT_INFO("[0, 0] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 0, 0}), probs_returned[0][0]);
        NT_INFO("[0, 1] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 0, 1}), probs_returned[0][1]);
        NT_INFO("[0, 2] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 0, 2}), probs_returned[0][2]);
        NT_INFO("[1, 0] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 1, 0}), probs_returned[1][0]);
        NT_INFO("[1, 1] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 1, 1}), probs_returned[1][1]);
        NT_INFO("[1, 2] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 1, 2}), probs_returned[1][2]);
        NT_INFO("[2, 0] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 2, 0}), probs_returned[2][0]);
        NT_INFO("[2, 1] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 2, 1}), probs_returned[2][1]);
        NT_INFO("[2, 2] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 2, 2}), probs_returned[2][2]);

        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 0, 0}), probs_returned[0][0], 1e-6);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 0, 1}), probs_returned[0][1], 1e-6);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 0, 2}), probs_returned[0][2], 1e-6);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 1, 0}), probs_returned[1][0], 1e-6);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 1, 1}), probs_returned[1][1], 1e-6);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 1, 2}), probs_returned[1][2], 1e-6);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 2, 0}), probs_returned[2][0], 1e-6);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 2, 1}), probs_returned[2][1], 1e-6);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 2, 2}), probs_returned[2][2], 1e-6);

    }

    /// compare DP propagator oscillation probabilities to the usual propagator
    void comparePropagator(bool antineutrino) {

        _setParamValues();

        Tensor pmnsTensor = pmns.build();

        tensorPropagator.setMixingMatrix(pmnsTensor);
        tensorPropagator.setAntiNeutrino(antineutrino);
        
        dpPropagator.setAntiNeutrino(antineutrino);

        // get propagator probabilities
        Tensor probabilities = tensorPropagator.calculateProbs();
        Tensor dpProbabilities = dpPropagator.calculateProbs();

        NT_INFO("[0, 0] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 0, 0}), dpProbabilities.getValue<float>({0, 0, 0}));
        NT_INFO("[0, 1] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 0, 1}), dpProbabilities.getValue<float>({0, 0, 1}));
        NT_INFO("[0, 2] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 0, 2}), dpProbabilities.getValue<float>({0, 0, 2}));
        NT_INFO("[1, 0] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 1, 0}), dpProbabilities.getValue<float>({0, 1, 0}));
        NT_INFO("[1, 1] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 1, 1}), dpProbabilities.getValue<float>({0, 1, 1}));
        NT_INFO("[1, 2] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 1, 2}), dpProbabilities.getValue<float>({0, 1, 2}));
        NT_INFO("[2, 0] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 2, 0}), dpProbabilities.getValue<float>({0, 2, 0}));
        NT_INFO("[2, 1] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 2, 1}), dpProbabilities.getValue<float>({0, 2, 1}));
        NT_INFO("[2, 2] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 2, 2}), dpProbabilities.getValue<float>({0, 2, 2}));

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), dpProbabilities.getValue<float>({0, 0, 0}), 1e-6);
        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), dpProbabilities.getValue<float>({0, 0, 1}), 1e-6);
        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 2}), dpProbabilities.getValue<float>({0, 0, 2}), 1e-6);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), dpProbabilities.getValue<float>({0, 1, 0}), 1e-6);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), dpProbabilities.getValue<float>({0, 1, 1}), 1e-6);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 2}), dpProbabilities.getValue<float>({0, 1, 2}), 1e-6);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 0}), dpProbabilities.getValue<float>({0, 2, 0}), 1e-6);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 1}), dpProbabilities.getValue<float>({0, 2, 1}), 1e-6);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 2}), dpProbabilities.getValue<float>({0, 2, 2}), 1e-6);
    }
};


// compare dpPropagator osc probs with Propagator osc probs
TEST_P(DPpropagatorTest, CompareToPropagator) {

    comparePropagator(false);

}
TEST_P(DPpropagatorTest, CompareToPropagator_antinu) {

    comparePropagator(true);

}

// compare dpPropagator osc probs with nuFast
TEST_P(DPpropagatorTest, CompareToNuFast) {

    compareNufast(false);

}
TEST_P(DPpropagatorTest, CompareToNuFast_antinu) {

    compareNufast(true);

}

INSTANTIATE_TEST_CASE_P(
    OscProb,
    DPpropagatorTest,
    ::testing::Values(
        -M_PI, 
        -0.8*M_PI, 
        -0.5*M_PI, 
        -0.2*M_PI, 
        0.0, 
        0.3*M_PI, 
        0.5*M_PI,
        0.7*M_PI,
        M_PI
));
