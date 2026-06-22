#include <tests/test-matter-solver.hpp>

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
TEST(BaseMatterSolver /*unused*/, SettersGetters)
{

    Tensor energies = Tensor::ones({10, 1}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor masses = Tensor::ones({1, 3}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor diagonal = Tensor({1.0, 1.0, 1.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor mixingMatrix = Tensor::diag(diagonal).unsqueeze(0);

    DummyMatterSolver matterSolver = DummyMatterSolver(3, false);
    matterSolver.setEnergies(energies);
    matterSolver.setMasses(masses);
    matterSolver.setMixingMatrix(mixingMatrix);
    matterSolver.setAntiNeutrino(true);
    ;

    ASSERT_EQ(matterSolver.getEnergies(), energies);
    ASSERT_EQ(matterSolver.getMasses(), masses);
    ASSERT_EQ(matterSolver.getMixingMatrix(), mixingMatrix);
    ASSERT_EQ(matterSolver.getAntiNeutrino(), true);
}

TEST(BaseMatterSolver /*unused*/, SetterErrors)
{

    Tensor badEnergies = Tensor::ones({10}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor badMasses = Tensor::ones({3}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor diagonal = Tensor({1.0, 1.0, 1.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor badMixingMatrix = Tensor::diag(diagonal);

    DummyMatterSolver matterSolver = DummyMatterSolver(3, false);

    EXPECT_THROW(matterSolver.setMasses(badMasses), std::invalid_argument);
    EXPECT_THROW(matterSolver.setEnergies(badEnergies), std::invalid_argument);
    EXPECT_THROW(matterSolver.setMixingMatrix(badMixingMatrix), std::invalid_argument);
}

TEST(ConstDensityMatterSolver /*unused*/, SetterErrors)
{

    Tensor badEnergies = Tensor::ones({10}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor badMasses = Tensor::ones({3}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor diagonal = Tensor({1.0, 1.0, 1.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor badMixingMatrix = Tensor::diag(diagonal);

    ConstDensityMatterSolver matterSolver = ConstDensityMatterSolver(3);

    EXPECT_THROW(matterSolver.setMasses(badMasses), std::invalid_argument);
    EXPECT_THROW(matterSolver.setEnergies(badEnergies), std::invalid_argument);
    EXPECT_THROW(matterSolver.setMixingMatrix(badMixingMatrix), std::invalid_argument);
}

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)