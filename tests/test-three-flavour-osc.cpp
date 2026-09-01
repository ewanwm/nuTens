#include <tests/test-three-flavour-osc.hpp>

using namespace nuTens;
using namespace nuTens::testing;

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)

// test const density matter oscillations
TEST_P(ThreeFlavourOscillations /*unused*/, ConstDensityEigenvaluesNu /*unused*/)
{

    testConstDensityEvals(/*antiNu=*/false);
}

// test const density matter oscillations for anti-neutrinos
TEST_P(ThreeFlavourOscillations /*unused*/, ConstDensityEigenvaluesAntiNu /*unused*/)
{

    testConstDensityEvals(/*antiNu=*/true);
}

// test const density matter oscillations
TEST_P(ThreeFlavourOscillations /*unused*/, ConstDensityHamiltonianNu /*unused*/)
{

    testConstDensityHamiltonian(/*antiNu=*/false);
}

// test const density matter oscillations for anti-neutrinos
TEST_P(ThreeFlavourOscillations /*unused*/, ConstDensityHamiltonianAntiiNu /*unused*/)
{

    testConstDensityHamiltonian(/*antiNu=*/true);
}

// test const density matter oscillations
TEST_P(ThreeFlavourOscillations /*unused*/, ConstDensityOscProbsNu /*unused*/)
{

    testConstDensityOscillations(/*antiNu=*/false);
}

// test const density matter oscillations for anti-neutrinos
TEST_P(ThreeFlavourOscillations /*unused*/, ConstDensityOscProbsAntiNu /*unused*/)
{

    testConstDensityOscillations(/*antiNu=*/true);
}

// test vacuum oscillations
TEST_P(ThreeFlavourOscillations /*unused*/, VacuumOscProbsNu /*unused*/)
{

    testVacuum(/*antiNu=*/false);
}

// test const density matter oscillations for anti-neutrinos
TEST_P(ThreeFlavourOscillations /*unused*/, VacuumOscProbsAntiNu /*unused*/)
{

    testVacuum(/*antiNu=*/true);
}

#if COMPILE_GPU_TESTS

INSTANTIATE_TEST_CASE_P(
    OscProb, ThreeFlavourOscillations,
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
    OscProb, ThreeFlavourOscillations,
    ::testing::Values(std::make_tuple(-M_PI, dtypes::kCPU), std::make_tuple(-0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.6 * M_PI, dtypes::kCPU), std::make_tuple(-0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(-0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.0, dtypes::kCPU),
                      std::make_tuple(0.2 * M_PI, dtypes::kCPU), std::make_tuple(0.4 * M_PI, dtypes::kCPU),
                      std::make_tuple(0.6 * M_PI, dtypes::kCPU), std::make_tuple(0.8 * M_PI, dtypes::kCPU),
                      std::make_tuple(M_PI, dtypes::kCPU)));

#endif

/////////////////////////////////////
// now non parameterised tests
/////////////////////////////////////

TEST(ThreeFlavourOscillations /*unused*/, batchedOscProbs /*unused*/)
{

    testBatching(/*dType=*/dtypes::kFloat, /*deviceType=*/dtypes::kCPU, false);
}

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)