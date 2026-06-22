#include <tests/test-DP-propagator.hpp>

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
using namespace nuTens;
using namespace nuTens::testing;

// compare dpPropagator osc probs with Propagator osc probs
TEST_P(DPpropagatorTest /*unused*/, CompareToPropagator /*unused*/)
{

    comparePropagator(false);
}
TEST_P(DPpropagatorTest /*unused*/, CompareToPropagator_antinu /*unused*/)
{

    comparePropagator(true);
}

// compare dpPropagator osc probs with nuFast
TEST_P(DPpropagatorTest /*unused*/, CompareToNuFast /*unused*/)
{

    compareNufast(false);
}
TEST_P(DPpropagatorTest /*unused*/, CompareToNuFast_antinu /*unused*/)
{

    compareNufast(true);
}
// compare dpPropagator osc probs with nuFast
TEST_P(DPpropagatorTest /*unused*/, CompareToNuFastVacuum /*unused*/)
{

    compareNufastVacuum(false);
}
TEST_P(DPpropagatorTest /*unused*/, CompareToNuFastVacuum_antinu /*unused*/)
{

    compareNufastVacuum(true);
}
// compare dpPropagator osc probs with nuFast
TEST_P(DPpropagatorTest /*unused*/, CompareToNuFast_sinSquaredThetas /*unused*/)
{

    compareNufast(false, true);
}

// test that auto diff works and gives same value for both
// Propagator and DPpropagator
TEST_P(DPpropagatorTest /*unused*/, autogradTest /*unused*/)
{

    _setParamValues(/*forceLowerOctant=*/false, /*interpretSinSquaredThetas=*/false);

    theta23tensor.requiresGrad(true);

    Tensor pmnsTensor = pmns.build();
    tensorPropagator.setMixingMatrix(pmnsTensor);

    // get Propagator probabilities
    Tensor probabilities = tensorPropagator.calculateProbs();
    Tensor muSurvivalProb = probabilities.getValues({0, 1, 1});

    muSurvivalProb.backward();

    NT_INFO("Propagator:   d P_(mu->mu) / d theta_23 = {}", pmns.getTheta23Tensor().grad().getValue<float>());

    // get DPpropagator probabilities
    Tensor dpProbabilities = dpPropagator.calculateProbs();
    Tensor dpMuSurvivalProb = dpProbabilities.getValues({0, 1, 1});

    dpMuSurvivalProb.backward();

    NT_INFO("DPpropagator: d P_(mu->mu) / d theta_23 = {}", theta23tensor.grad().getValue<float>());

    // check that the values are close to each other
    ASSERT_NEAR(pmns.getTheta23Tensor().grad().getValue<float>(), theta23tensor.grad().getValue<float>(), tolerance);
}

INSTANTIATE_TEST_CASE_P(OscProb, DPpropagatorTest,
                        ::testing::Values(-M_PI, -0.8 * M_PI, -0.5 * M_PI, -0.2 * M_PI, 0.0, 0.3 * M_PI, 0.5 * M_PI,
                                          0.7 * M_PI, M_PI));

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)