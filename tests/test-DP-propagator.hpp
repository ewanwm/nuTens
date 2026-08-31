#pragma once

#include <gtest/gtest.h> // NOLINT
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <nuTens/propagator/DP-propagator.hpp>
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/pmns-matrix.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <nuTens/testing/barger-propagator.hpp>
#include <nuTens/testing/printing.hpp>
#include <nuTens/testing/utils.hpp>
#include <nuTens/utils/logging.hpp>

// nuFast c++ implementation
#include <nuTens/testing/nuFast.hpp>

using namespace nuTens;
using namespace nuTens::testing;

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
class DPpropagatorTest : public gtest::TestWithParam<std::tuple<float, dtypes::deviceType>>
{

  protected:
    // NOLINTBEGIN(cppcoreguidelines-non-private-member-variables-in-classes)
    float mass1 = 0.0 * units::eV;
    float mass2 = 0.008 * units::eV;
    float mass3 = 0.02 * units::eV;
    float dcp = M_PI / 4.0;
    float energy = 0.5 * units::GeV;
    float baseline = 295.0 * units::km;
    float density = 2.6;
    float tolerance = 1e-5;

    // set the tensors we will use to calculate matter eigenvalues
    Tensor masses;

    float theta23;

    float theta13 = 0.3 * M_PI;
    float theta12 = 0.2 * M_PI;

    dtypes::deviceType device;

    Tensor theta23tensor;
    Tensor theta13tensor;
    Tensor theta12tensor;
    Tensor deltaCPtensor;
    Tensor dmsq21tensor;
    Tensor dmsq31tensor;

    Tensor energies;

    Propagator tensorPropagator = Propagator(1);
    std::shared_ptr<ConstDensityMatterSolver> tensorSolver;

    DPpropagator dpPropagator = DPpropagator(1);
    DPpropagator dpPropagatorVac = DPpropagator(1);

    PMNSmatrix pmns;

    ThreeFlavourBarger<> barger();
    // NOLINTEND(cppcoreguidelines-non-private-member-variables-in-classes)

    void SetUp() override
    {

        device = std::get<1>(GetParam());
        // skip if no GPU available
        SKIP_GPU(device);

        // set up propagators
        tensorPropagator = Propagator(3, device).setBaseline(baseline);
        tensorSolver = std::make_shared<ConstDensityMatterSolver>(3, device);

        dpPropagator = DPpropagator(10, device).setBaseline(baseline).setAntiNeutrino(false).setDensity(density);
        dpPropagatorVac = DPpropagator(10, device).setBaseline(baseline).setAntiNeutrino(false).setDensity(0.0);

        pmns = PMNSmatrix(device);

        // set up tensor values
        energies = Tensor::ones({1}, dtypes::kComplexFloat).requiresGrad(false).hasBatchDim(true).device(device);
        energies.setValue({0}, energy);

        theta23tensor = Tensor::zeros({1}, dtypes::kComplexFloat, device, false);
        theta13tensor = Tensor::zeros({1}, dtypes::kComplexFloat, device, false);
        theta12tensor = Tensor::zeros({1}, dtypes::kComplexFloat, device, false);
        deltaCPtensor = Tensor::zeros({1}, dtypes::kComplexFloat, device, false);
        dmsq21tensor = Tensor::zeros({1}, dtypes::kComplexFloat, device, false);
        dmsq31tensor = Tensor::zeros({1}, dtypes::kComplexFloat, device, false);

        masses = Tensor({mass1, mass2, mass3}, dtypes::kComplexFloat).addBatchDim().requiresGrad(true).device(device);

        tensorSolver->setDensity(density);

        tensorPropagator.setMatterSolver(tensorSolver);
        tensorPropagator.setMasses(masses);
        tensorPropagator.setEnergies(energies);

        dpPropagator.setEnergies(energies);
        dpPropagator.setTheta12(theta12tensor)
            .setTheta23(theta23tensor)
            .setTheta13(theta13tensor)
            .setDeltaCP(deltaCPtensor)
            .setDmsq21(dmsq21tensor)
            .setDmsq31(dmsq31tensor);
        dpPropagatorVac.setEnergies(energies);
        dpPropagatorVac.setTheta12(theta12tensor)
            .setTheta23(theta23tensor)
            .setTheta13(theta13tensor)
            .setDeltaCP(deltaCPtensor)
            .setDmsq21(dmsq21tensor)
            .setDmsq31(dmsq31tensor);
    }

    /// set the oscillation parameter values
    void _setParamValues(bool forceLowerOctant, bool interpretSinSquaredThetas)
    {
        // get parameterised theta value
        float theta = std::get<0>(GetParam());

        // allow user to force theta to be in lower octant
        // (allows correct comparison with nufast)
        if (forceLowerOctant)
        {
            theta = std::asin(std::abs(std::sin(theta)));
        }

        NT_INFO("########## theta = {} ##########", theta);

        theta23 = theta;

        if (interpretSinSquaredThetas)
        {
            theta23tensor.setValue({0}, std::sin(theta23) * std::sin(theta23));
            theta13tensor.setValue({0}, std::sin(theta13) * std::sin(theta13));
            theta12tensor.setValue({0}, std::sin(theta12) * std::sin(theta12));
        }
        else
        {
            theta23tensor.setValue({0}, theta23);
            theta13tensor.setValue({0}, theta13);
            theta12tensor.setValue({0}, theta12);
        }

        dmsq21tensor.setValue({0}, mass2 * mass2 - mass1 * mass1);
        dmsq31tensor.setValue({0}, mass3 * mass3 - mass1 * mass1);

        deltaCPtensor.setValue({0}, dcp);

        // calculate new values of the mixing matrix
        pmns.setTheta12(theta12).setTheta13(theta13).setTheta23(theta23).setDeltaCP(dcp);
    }

    void testParameterSetting()
    {
        dpPropagator.setBaseline(0.1);
        dpPropagator.setDensity(0.2);

        // need this to move above float values into coresponding tensors
        (void)dpPropagator.calculateProbs();

        ASSERT_EQ(dpPropagator.getBaseline(), 0.1);
        ASSERT_EQ(dpPropagator.getDensity(), 0.2);
    }

    /// compare DP propagator oscillation probabilities to the "official" nufast code

    // cognitive complexity is heavily inflated by the gtest macros
    // but they don't actually decrease readability
    // NOLINTBEGIN(readability-function-cognitive-complexity)
    void compareNufast(bool antineutrino, bool interpretSinSquaredThetas = false)
    {

        // need to force theta into lower octant as this is assumed by
        // nufast so otherwise result will differ and test will break
        _setParamValues(/*forceLowerOctant=*/true, interpretSinSquaredThetas);

        dpPropagator.setAntiNeutrino(antineutrino);
        dpPropagator.setSinSquaredThetas(interpretSinSquaredThetas);

        // get propagator probabilities
        Tensor dpProbabilities = dpPropagator.calculateProbs();

        // get the nuFast probabilities
        double probs_returned[3][3];
        Probability_Matter_LBL(std::sin(theta12) * std::sin(theta12), std::sin(theta13) * std::sin(theta13),
                               std::sin(theta23) * std::sin(theta23), dcp, mass1 * mass1 - mass2 * mass2,
                               mass1 * mass1 - mass3 * mass3, baseline / units::km,
                               (0.5 - (float)antineutrino) * 2.0 * energies.getValue<float>() / units::GeV, 1.0,
                               density, 10, &probs_returned);

        NT_INFO("[0, 0] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 0, 0}),
                probs_returned[0][0]);
        NT_INFO("[0, 1] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 0, 1}),
                probs_returned[0][1]);
        NT_INFO("[0, 2] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 0, 2}),
                probs_returned[0][2]);
        NT_INFO("[1, 0] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 1, 0}),
                probs_returned[1][0]);
        NT_INFO("[1, 1] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 1, 1}),
                probs_returned[1][1]);
        NT_INFO("[1, 2] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 1, 2}),
                probs_returned[1][2]);
        NT_INFO("[2, 0] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 2, 0}),
                probs_returned[2][0]);
        NT_INFO("[2, 1] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 2, 1}),
                probs_returned[2][1]);
        NT_INFO("[2, 2] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 2, 2}),
                probs_returned[2][2]);

        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 0, 0}), probs_returned[0][0], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 0, 1}), probs_returned[0][1], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 0, 2}), probs_returned[0][2], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 1, 0}), probs_returned[1][0], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 1, 1}), probs_returned[1][1], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 1, 2}), probs_returned[1][2], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 2, 0}), probs_returned[2][0], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 2, 1}), probs_returned[2][1], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 2, 2}), probs_returned[2][2], tolerance);
    }

    /// compare DP propagator oscillation probabilities to the "official" nufast code for vacuum oscillations
    void compareNufastVacuum(bool antineutrino, bool interpretSinSquaredThetas = false)
    {

        // need to force theta into lower octant as this is assumed by
        // nufast so otherwise result will differ and test will break
        _setParamValues(/*forceLowerOctant=*/true, interpretSinSquaredThetas);

        dpPropagatorVac.setAntiNeutrino(antineutrino);
        dpPropagator.setSinSquaredThetas(interpretSinSquaredThetas);

        // get propagator probabilities
        Tensor dpProbabilities = dpPropagatorVac.calculateProbs();

        // get the nuFast probabilities
        double probs_returned[3][3];
        Probability_Vacuum_LBL(std::sin(theta12) * std::sin(theta12), std::sin(theta13) * std::sin(theta13),
                               std::sin(theta23) * std::sin(theta23), dcp, mass1 * mass1 - mass2 * mass2,
                               mass1 * mass1 - mass3 * mass3, baseline / units::km,
                               (0.5 - (float)antineutrino) * 2.0 * energies.getValue<float>() / units::GeV,
                               &probs_returned);

        NT_INFO("[0, 0] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 0, 0}),
                probs_returned[0][0]);
        NT_INFO("[0, 1] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 0, 1}),
                probs_returned[0][1]);
        NT_INFO("[0, 2] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 0, 2}),
                probs_returned[0][2]);
        NT_INFO("[1, 0] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 1, 0}),
                probs_returned[1][0]);
        NT_INFO("[1, 1] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 1, 1}),
                probs_returned[1][1]);
        NT_INFO("[1, 2] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 1, 2}),
                probs_returned[1][2]);
        NT_INFO("[2, 0] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 2, 0}),
                probs_returned[2][0]);
        NT_INFO("[2, 1] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 2, 1}),
                probs_returned[2][1]);
        NT_INFO("[2, 2] :: DP propagator: {:.7f} :: nuFast: {:.7f}", dpProbabilities.getValue<float>({0, 2, 2}),
                probs_returned[2][2]);

        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 0, 0}), probs_returned[0][0], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 0, 1}), probs_returned[0][1], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 0, 2}), probs_returned[0][2], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 1, 0}), probs_returned[1][0], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 1, 1}), probs_returned[1][1], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 1, 2}), probs_returned[1][2], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 2, 0}), probs_returned[2][0], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 2, 1}), probs_returned[2][1], tolerance);
        ASSERT_NEAR(dpProbabilities.getValue<float>({0, 2, 2}), probs_returned[2][2], tolerance);
    }

    /// compare DP propagator oscillation probabilities to the usual propagator
    void comparePropagator(bool antineutrino, bool interpretSinSquaredThetas = false)
    {

        _setParamValues(/*forceLowerOctant=*/false, interpretSinSquaredThetas);

        Tensor pmnsTensor = pmns.build();

        tensorPropagator.setMixingMatrix(pmnsTensor);
        tensorPropagator.setAntiNeutrino(antineutrino);

        dpPropagator.setAntiNeutrino(antineutrino);
        dpPropagator.setSinSquaredThetas(interpretSinSquaredThetas);

        // get propagator probabilities
        Tensor probabilities = tensorPropagator.calculateProbs();
        Tensor dpProbabilities = dpPropagator.calculateProbs();

        NT_INFO("[0, 0] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 0, 0}),
                dpProbabilities.getValue<float>({0, 0, 0}));
        NT_INFO("[0, 1] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 0, 1}),
                dpProbabilities.getValue<float>({0, 0, 1}));
        NT_INFO("[0, 2] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 0, 2}),
                dpProbabilities.getValue<float>({0, 0, 2}));
        NT_INFO("[1, 0] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 1, 0}),
                dpProbabilities.getValue<float>({0, 1, 0}));
        NT_INFO("[1, 1] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 1, 1}),
                dpProbabilities.getValue<float>({0, 1, 1}));
        NT_INFO("[1, 2] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 1, 2}),
                dpProbabilities.getValue<float>({0, 1, 2}));
        NT_INFO("[2, 0] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 2, 0}),
                dpProbabilities.getValue<float>({0, 2, 0}));
        NT_INFO("[2, 1] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 2, 1}),
                dpProbabilities.getValue<float>({0, 2, 1}));
        NT_INFO("[2, 2] :: propagator: {:.7f} :: DP propagator: {:.7f}", probabilities.getValue<float>({0, 2, 2}),
                dpProbabilities.getValue<float>({0, 2, 2}));

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), dpProbabilities.getValue<float>({0, 0, 0}), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), dpProbabilities.getValue<float>({0, 0, 1}), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 2}), dpProbabilities.getValue<float>({0, 0, 2}), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), dpProbabilities.getValue<float>({0, 1, 0}), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), dpProbabilities.getValue<float>({0, 1, 1}), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 2}), dpProbabilities.getValue<float>({0, 1, 2}), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 0}), dpProbabilities.getValue<float>({0, 2, 0}), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 1}), dpProbabilities.getValue<float>({0, 2, 1}), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 2}), dpProbabilities.getValue<float>({0, 2, 2}), tolerance);
    }

    /// compare gradient from DPpropagator to regular Propagator
    void autogradTest()
    {
        _setParamValues(/*forceLowerOctant=*/false, /*interpretSinSquaredThetas=*/false);

        theta23tensor.requiresGrad(true);

        Tensor pmnsTensor = pmns.build().device(device);

        tensorPropagator.setMixingMatrix(pmnsTensor);

        // get Propagator probabilities
        Tensor probabilities = tensorPropagator.calculateProbs();
        Tensor muSurvivalProb = probabilities.getValues({0, 1, 1}).device(device);

        muSurvivalProb.backward();

        NT_INFO("Propagator:   d P_(mu->mu) / d theta_23 = {}", pmns.getTheta23Tensor().grad().getValue<float>());

        // get DPpropagator probabilities
        Tensor dpProbabilities = dpPropagator.calculateProbs();
        Tensor dpMuSurvivalProb = dpProbabilities.getValues({0, 1, 1});

        dpMuSurvivalProb.backward();

        NT_INFO("DPpropagator: d P_(mu->mu) / d theta_23 = {}", theta23tensor.grad().getValue<float>());

        // check that the values are close to each other
        ASSERT_NEAR(pmns.getTheta23Tensor().grad().getValue<float>(), theta23tensor.grad().getValue<float>(),
                    tolerance);
    }
    // NOLINTEND(readability-function-cognitive-complexity)
};
// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)