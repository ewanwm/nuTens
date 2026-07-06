#include <tests/test-two-flavour-osc.hpp>

using namespace nuTens;
using namespace nuTens::testing;

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)

// test that Propagator gives expected oscillation probabilites for a range
// of thetas
TEST_P(TwoFlavourOscillations /*unused*/, VacuumOscProbs /*unused*/)
{
    testVacuum(/*antiNu=*/false);
}

// test const density matter oscillations
TEST_P(TwoFlavourOscillations /*unused*/, ConstDensityOscProbsNu /*unused*/)
{

    testConstDensity(/*antiNu=*/false);
}

// test const density matter oscillations for anti-neutrinos
TEST_P(TwoFlavourOscillations /*unused*/, ConstDensityOscProbsAntiNu /*unused*/)
{

    testConstDensity(/*antiNu=*/true);
}

#if COMPILE_GPU_TESTS

INSTANTIATE_TEST_CASE_P(
    OscProb, TwoFlavourOscillations,
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
    OscProb, TwoFlavourOscillations,
    ::testing::Values(std::make_tuple(-M_PI, dtypes::kCPU), std::make_tuple(-0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.6 * M_PI, dtypes::kCPU), std::make_tuple(-0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.0, dtypes::kCPU),
                      std::make_tuple(0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(0.6 * M_PI, dtypes::kCPU), std::make_tuple(0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(M_PI, dtypes::kCPU)));

#endif

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)