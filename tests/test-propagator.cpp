#include <tests/test-propagator.hpp>

using namespace nuTens;

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)

// cognitive complexity is heavily inflated by the gtest macros
// but they don't actually decrease readability
// NOLINTBEGIN(readability-function-cognitive-complexity)

TEST(Propagator /*unused*/, InitialisationOrderMatterSolverFirst /*unused*/)
{
    // check that order of initialisation of matter solver and parameters doesn't matter

    Tensor energies = Tensor::ones({10}, dtypes::kComplexFloat);
    Tensor masses = Tensor::ones({1, 3});
    Tensor diagonal = Tensor({1.0, 1.0, 1.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor mixingMatrix = Tensor::diag(diagonal).unsqueeze(0);

    Propagator matterSolverFirst(/*nGenerations=*/3);
    auto matterSolver1 = std::make_shared<ConstDensityMatterSolver>(3);

    // try setting the matter solver before setting all parameters
    matterSolverFirst.setMatterSolver(matterSolver1);
    matterSolverFirst.setEnergies(energies);
    matterSolverFirst.setMasses(masses);
    matterSolverFirst.setAntiNeutrino(true);
    matterSolverFirst.setMixingMatrix(mixingMatrix);

    ASSERT_EQ(matterSolver1->getEnergies(), energies);
    ASSERT_EQ(matterSolver1->getMasses(), masses);
    ASSERT_EQ(matterSolver1->getMixingMatrix(), mixingMatrix);
    ASSERT_EQ(matterSolver1->getAntiNeutrino(), true);
}

TEST(Propagator /*unused*/, InitialisationOrderMatterSolverAfter /*unused*/)
{
    // check that order of initialisation of matter solver and parameters doesn't matter

    Tensor energies = Tensor::ones({10}, dtypes::kComplexFloat);
    Tensor masses = Tensor::ones({1, 3});
    Tensor diagonal = Tensor({1.0, 1.0, 1.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor mixingMatrix = Tensor::diag(diagonal).unsqueeze(0);

    // now try setting the matter solver after setting all parameters
    Propagator matterSolverAfter(/*nGenerations=*/3);
    auto matterSolver2 = std::make_shared<ConstDensityMatterSolver>(3);
    matterSolverAfter.setEnergies(energies);
    matterSolverAfter.setMasses(masses);
    matterSolverAfter.setMixingMatrix(mixingMatrix);
    matterSolverAfter.setAntiNeutrino(true);
    matterSolverAfter.setMatterSolver(matterSolver2);

    ASSERT_EQ(matterSolver2->getEnergies(), energies);
    ASSERT_EQ(matterSolver2->getMasses(), masses);
    ASSERT_EQ(matterSolver2->getMixingMatrix(), mixingMatrix);
    ASSERT_EQ(matterSolver2->getAntiNeutrino(), true);
}

TEST(Propagator /*unused*/, SetterErrors)
{

    Tensor badEnergiesWrongSize = Tensor::ones({10, 1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor badEnergiesWrongType = Tensor::ones({10}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor badMassesWrongSize = Tensor::ones({1, 1, 3}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor badMassesWrongShape = Tensor::ones({1, 4}, dtypes::kFloat, dtypes::kCPU, false);

    Tensor diagonal = Tensor({1.0, 1.0, 1.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor badMixingMatrixWrongSize = Tensor::diag(diagonal).unsqueeze(0).unsqueeze(0);

    diagonal = Tensor({1.0, 1.0, 1.0, 1.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor badMixingMatrixWrongShape = Tensor::diag(diagonal).unsqueeze(0);

    Propagator propagator = Propagator(/*nGenerations=*/3);

    EXPECT_THROW(propagator.setMasses(badMassesWrongSize), std::invalid_argument);
    EXPECT_THROW(propagator.setMasses(badMassesWrongShape), std::invalid_argument);
    EXPECT_THROW(propagator.setEnergies(badEnergiesWrongSize), std::invalid_argument);
    EXPECT_THROW(propagator.setEnergies(badEnergiesWrongType), std::invalid_argument);
    EXPECT_THROW(propagator.setMixingMatrix(badMixingMatrixWrongSize), std::invalid_argument);
    EXPECT_THROW(propagator.setMixingMatrix(badMixingMatrixWrongShape), std::invalid_argument);
}

TEST(Propagator /*unused*/, invalidConfigErrors)
{

    Tensor masses = Tensor::ones({1, 3});
    Tensor energies = Tensor::ones({10}, dtypes::kComplexFloat);
    Tensor diagonal = Tensor({1.0, 1.0, 1.0}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor mixingMatrix = Tensor::diag(diagonal).unsqueeze(0);

    Propagator propagator = Propagator(/*nGenerations=*/3);

    EXPECT_THROW((void)propagator.calculateProbs(), std::runtime_error);

    propagator.setMasses(masses);

    EXPECT_THROW((void)propagator.calculateProbs(), std::runtime_error);

    propagator.setEnergies(energies);

    EXPECT_THROW((void)propagator.calculateProbs(), std::runtime_error);

    propagator.setMixingMatrix(mixingMatrix);

    EXPECT_NO_THROW((void)propagator.calculateProbs());
}

// NOLINTEND(readability-function-cognitive-complexity)

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)