#include <tests/test-DP-propagator.hpp>

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
using namespace nuTens;
using namespace nuTens::testing;

typedef DPpropagatorTester<DPpropagator> DPpropagatorTest;
typedef DPpropagatorTester<PrecompiledDPpropagator> PrecompiledDPpropagatorTest;

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

    autogradTest();
}

#if COMPILE_GPU_TESTS

INSTANTIATE_TEST_CASE_P(
    OscProb, DPpropagatorTest,
    ::testing::Values(std::make_tuple(-M_PI, dtypes::kCPU), std::make_tuple(-0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.6 * M_PI, dtypes::kCPU), std::make_tuple(-0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.0, dtypes::kCPU),
                      std::make_tuple(0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(0.6 * M_PI, dtypes::kCPU), std::make_tuple(0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(M_PI, dtypes::kCPU), std::make_tuple(-M_PI, dtypes::kCPU),
                      std::make_tuple(-0.8 * M_PI, dtypes::kGPU), std::make_tuple(-0.6 * M_PI, dtypes::kGPU),
                      std::make_tuple(-0.4 * M_PI, dtypes::kGPU), std::make_tuple(-0.2 * M_PI, dtypes::kGPU),
                      std::make_tuple(0.0, dtypes::kGPU), std::make_tuple(0.2 * M_PI, dtypes::kGPU),
                      std::make_tuple(0.4 * M_PI, dtypes::kGPU), std::make_tuple(0.6 * M_PI, dtypes::kGPU),
                      std::make_tuple(0.8 * M_PI, dtypes::kGPU), std::make_tuple(M_PI, dtypes::kGPU)));

#else

INSTANTIATE_TEST_CASE_P(
    OscProb, DPpropagatorTest,
    ::testing::Values(std::make_tuple(-M_PI, dtypes::kCPU), std::make_tuple(-0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.6 * M_PI, dtypes::kCPU), std::make_tuple(-0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.0, dtypes::kCPU),
                      std::make_tuple(0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(0.6 * M_PI, dtypes::kCPU), std::make_tuple(0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(M_PI, dtypes::kCPU)));

#endif

// compare dpPropagator osc probs with Propagator osc probs
TEST_P(PrecompiledDPpropagatorTest /*unused*/, CompareToPropagator /*unused*/)
{

    comparePropagator(false);
}
TEST_P(PrecompiledDPpropagatorTest /*unused*/, CompareToPropagator_antinu /*unused*/)
{

    comparePropagator(true);
}

// compare dpPropagator osc probs with nuFast
TEST_P(PrecompiledDPpropagatorTest /*unused*/, CompareToNuFast /*unused*/)
{

    compareNufast(false);
}
TEST_P(PrecompiledDPpropagatorTest /*unused*/, CompareToNuFast_antinu /*unused*/)
{

    compareNufast(true);
}
// compare dpPropagator osc probs with nuFast
TEST_P(PrecompiledDPpropagatorTest /*unused*/, CompareToNuFastVacuum /*unused*/)
{

    compareNufastVacuum(false);
}
TEST_P(PrecompiledDPpropagatorTest /*unused*/, CompareToNuFastVacuum_antinu /*unused*/)
{

    compareNufastVacuum(true);
}
// compare dpPropagator osc probs with nuFast
TEST_P(PrecompiledDPpropagatorTest /*unused*/, CompareToNuFast_sinSquaredThetas /*unused*/)
{

    compareNufast(false, true);
}

// test that auto diff works and gives same value for both
// Propagator and DPpropagator
TEST_P(PrecompiledDPpropagatorTest /*unused*/, autogradTest /*unused*/)
{

    autogradTest();
}

#if COMPILE_GPU_TESTS

INSTANTIATE_TEST_CASE_P(
    OscProb, PrecompiledDPpropagatorTest,
    ::testing::Values(std::make_tuple(-M_PI, dtypes::kCPU), std::make_tuple(-0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.6 * M_PI, dtypes::kCPU), std::make_tuple(-0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.0, dtypes::kCPU),
                      std::make_tuple(0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(0.6 * M_PI, dtypes::kCPU), std::make_tuple(0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(M_PI, dtypes::kCPU), std::make_tuple(-M_PI, dtypes::kCPU),
                      std::make_tuple(-0.8 * M_PI, dtypes::kGPU), std::make_tuple(-0.6 * M_PI, dtypes::kGPU),
                      std::make_tuple(-0.4 * M_PI, dtypes::kGPU), std::make_tuple(-0.2 * M_PI, dtypes::kGPU),
                      std::make_tuple(0.0, dtypes::kGPU), std::make_tuple(0.2 * M_PI, dtypes::kGPU),
                      std::make_tuple(0.4 * M_PI, dtypes::kGPU), std::make_tuple(0.6 * M_PI, dtypes::kGPU),
                      std::make_tuple(0.8 * M_PI, dtypes::kGPU), std::make_tuple(M_PI, dtypes::kGPU)));

#else

INSTANTIATE_TEST_CASE_P(
    OscProb, PrecompiledDPpropagatorTest,
    ::testing::Values(std::make_tuple(-M_PI, dtypes::kCPU), std::make_tuple(-0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.6 * M_PI, dtypes::kCPU), std::make_tuple(-0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.0, dtypes::kCPU),
                      std::make_tuple(0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(0.6 * M_PI, dtypes::kCPU), std::make_tuple(0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(M_PI, dtypes::kCPU)));

#endif
// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)