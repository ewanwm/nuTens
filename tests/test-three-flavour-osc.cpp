#include <gtest/gtest.h> // NOLINT
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/pmns-matrix.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <tests/barger-propagator.hpp>

using namespace nuTens;
using namespace nuTens::testing;

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
class ThreeFlavourOscillations : public gtest::TestWithParam<float>
{
  protected:
    // NOLINTBEGIN(cppcoreguidelines-non-private-member-variables-in-classes)
    float theta12 = NAN;

    float theta23 = 0.23 * M_PI;
    float theta13 = 0.13 * M_PI;
    float deltaCP = 0.25 * M_PI;

    float mass1 = 0.0;
    float mass2 = 0.008 * units::eV * units::eV;
    float mass3 = 0.01 * units::eV * units::eV;

    float energy = 0.5 * units::GeV;
    float baseline = 295.0 * units::km;
    float density = 2.6;

    float tolerance = 1e-5;
    Tensor masses;
    Tensor energies;
    // NOLINTEND(cppcoreguidelines-non-private-member-variables-in-classes)

    // set up common values to use across tests
    void SetUp()
    {

        masses = Tensor({mass1, mass2, mass3}, dtypes::kComplexDouble).addBatchDim();
        energies = Tensor({energy}, dtypes::kComplexDouble).addBatchDim();
    }

    void testConstDensity(bool antiNu)
    {

        // get parameterised theta value
        theta12 = GetParam();

        NT_INFO("\n#### const density test for theta12 = {} ####", theta12);

        // set up the barger propagator
        ThreeFlavourBarger<> bargerProp{};
        bargerProp.setParams(mass1, mass2, mass3, theta12, theta13, theta23, deltaCP, baseline, density, antiNu);

        NT_INFO("alpha():  {}", bargerProp.calculateAlpha(energy));
        NT_INFO("beta():   {}", bargerProp.calculateBeta(energy));
        NT_INFO("gamma():  {}", bargerProp.calculateGamma(energy));
        NT_INFO("");

        // construct the mixing matrix for current theta value
        PMNSmatrix pmns;
        pmns.setParameterValues(theta12, theta13, theta23, deltaCP);
        Tensor pmnsTensor = pmns.build();

        NT_INFO("Re[PMNS]:\n{}", pmns.build().real().toString());
        NT_INFO("Im[PMNS]:\n{}", pmns.build().imag().toString());

        // set up the matter solver
        Propagator tensorPropagator(3, baseline);
        auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(3, density);

        // set up the propagator
        tensorPropagator.setMatterSolver(tensorSolver);
        tensorPropagator.setMixingMatrix(pmns.build());
        tensorPropagator.setMasses(masses);
        tensorPropagator.setAntiNeutrino(antiNu);
        tensorPropagator.setEnergies(energies);

        Tensor eigenVals;
        Tensor eigenVecs;

        tensorSolver->calculateEigenvalues(eigenVecs, eigenVals);

        // first check that the effective dM^2 from the tensor solver is what we
        // expect
        auto calcV1 = eigenVals.getValue<float>({0, 0}) * 2.0 * energy;
        auto calcV2 = eigenVals.getValue<float>({0, 1}) * 2.0 * energy;
        auto calcV3 = eigenVals.getValue<float>({0, 2}) * 2.0 * energy;

        // print out the effective masses from both methods
        NT_INFO("Effective masses:");
        NT_INFO("M1: tensor solver: {:.7f} :: barger: {:.7f}", calcV1, bargerProp.calculateEffectiveM2(energy, 0));
        NT_INFO("M2: tensor solver: {:.7f} :: barger: {:.7f}", calcV2, bargerProp.calculateEffectiveM2(energy, 1));
        NT_INFO("M3: tensor solver: {:.7f} :: barger: {:.7f}", calcV3, bargerProp.calculateEffectiveM2(energy, 2));
        NT_INFO("");
        NT_INFO("Effective dM^2's:");
        NT_INFO("dM^2_21: tensor solver: {:.7f} :: barger: {:.7f}", calcV2 - calcV1,
                bargerProp.calculateEffectiveM2(energy, 1) - bargerProp.calculateEffectiveM2(energy, 0));
        NT_INFO("dM^2_32: tensor solver: {:.7f} :: barger: {:.7f}", calcV3 - calcV2,
                bargerProp.calculateEffectiveM2(energy, 2) - bargerProp.calculateEffectiveM2(energy, 1));
        NT_INFO("dM^2_31: tensor solver: {:.7f} :: barger: {:.7f}", calcV3 - calcV1,
                bargerProp.calculateEffectiveM2(energy, 2) - bargerProp.calculateEffectiveM2(energy, 0));

        // print out the hamiltonians from both methods
        Tensor hamiltonianTensor = tensorSolver->getHamiltonian();
        NT_INFO("#########################################################################");
        NT_INFO("Hamiltonian:");
        NT_INFO(
            "[0,0] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 0, 0}),
            hamiltonianTensor.imag().getValue<float>({0, 0, 0}), bargerProp.getHamiltonianElement(energy, 0, 0).real(),
            bargerProp.getHamiltonianElement(energy, 0, 0).imag());
        NT_INFO(
            "[0,1] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 0, 1}),
            hamiltonianTensor.imag().getValue<float>({0, 0, 1}), bargerProp.getHamiltonianElement(energy, 0, 1).real(),
            bargerProp.getHamiltonianElement(energy, 0, 1).imag());
        NT_INFO(
            "[0,2] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 0, 2}),
            hamiltonianTensor.imag().getValue<float>({0, 0, 2}), bargerProp.getHamiltonianElement(energy, 0, 2).real(),
            bargerProp.getHamiltonianElement(energy, 0, 2).imag());
        NT_INFO("");
        NT_INFO(
            "[1,0] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 1, 0}),
            hamiltonianTensor.imag().getValue<float>({0, 1, 0}), bargerProp.getHamiltonianElement(energy, 1, 0).real(),
            bargerProp.getHamiltonianElement(energy, 1, 0).imag());
        NT_INFO(
            "[1,1] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 1, 1}),
            hamiltonianTensor.imag().getValue<float>({0, 1, 1}), bargerProp.getHamiltonianElement(energy, 1, 1).real(),
            bargerProp.getHamiltonianElement(energy, 1, 1).imag());
        NT_INFO(
            "[1,2] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 1, 2}),
            hamiltonianTensor.imag().getValue<float>({0, 1, 2}), bargerProp.getHamiltonianElement(energy, 1, 2).real(),
            bargerProp.getHamiltonianElement(energy, 1, 2).imag());
        NT_INFO("");
        NT_INFO(
            "[2,0] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 2, 0}),
            hamiltonianTensor.imag().getValue<float>({0, 2, 0}), bargerProp.getHamiltonianElement(energy, 2, 0).real(),
            bargerProp.getHamiltonianElement(energy, 2, 0).imag());
        NT_INFO(
            "[2,1] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 2, 1}),
            hamiltonianTensor.imag().getValue<float>({0, 2, 1}), bargerProp.getHamiltonianElement(energy, 2, 1).real(),
            bargerProp.getHamiltonianElement(energy, 2, 1).imag());
        NT_INFO(
            "[2,2] :: tensor solver ({}, {}i) :: barger ({}, {}i)", hamiltonianTensor.real().getValue<float>({0, 2, 2}),
            hamiltonianTensor.imag().getValue<float>({0, 2, 2}), bargerProp.getHamiltonianElement(energy, 2, 2).real(),
            bargerProp.getHamiltonianElement(energy, 2, 2).imag());
        NT_INFO("#########################################################################");

        // print put the probabilities obtained via both methods
        Tensor probabilities = tensorPropagator.calculateProbs();
        NT_INFO("#########################################################################");
        NT_INFO("Oscillation probabilities:");
        NT_INFO("[0,0] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 0, 0}),
                bargerProp.calculateProb(energy, 0, 0));
        NT_INFO("[0,1] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 0, 1}),
                bargerProp.calculateProb(energy, 0, 1));
        NT_INFO("[0,2] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 0, 2}),
                bargerProp.calculateProb(energy, 0, 2));
        NT_INFO("");
        NT_INFO("[1,0] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 1, 0}),
                bargerProp.calculateProb(energy, 1, 0));
        NT_INFO("[1,1] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 1, 1}),
                bargerProp.calculateProb(energy, 1, 1));
        NT_INFO("[1,2] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 1, 2}),
                bargerProp.calculateProb(energy, 1, 2));
        NT_INFO("");
        NT_INFO("[2,0] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 2, 0}),
                bargerProp.calculateProb(energy, 2, 0));
        NT_INFO("[2,1] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 2, 1}),
                bargerProp.calculateProb(energy, 2, 1));
        NT_INFO("[2,2] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 2, 2}),
                bargerProp.calculateProb(energy, 2, 2));
        NT_INFO("#########################################################################");

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 2}), bargerProp.calculateProb(energy, 0, 2), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 2}), bargerProp.calculateProb(energy, 1, 2), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 0}), bargerProp.calculateProb(energy, 2, 0), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 1}), bargerProp.calculateProb(energy, 2, 1), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 2}), bargerProp.calculateProb(energy, 2, 2), tolerance);
    }

    void testVacuum(bool antiNu)
    {

        // get parameterised theta value
        float theta12 = GetParam();

        NT_INFO("\n#### vacuum test for theta12 = {} ####", theta12);

        // set up the barger propagator
        ThreeFlavourBarger<> bargerProp{};
        bargerProp.setParams(mass1, mass2, mass3, theta12, theta13, theta23, deltaCP, baseline, density = -999.9, antiNu);

        NT_INFO("alpha():  {}", bargerProp.calculateAlpha(energy));
        NT_INFO("beta():   {}", bargerProp.calculateBeta(energy));
        NT_INFO("gamma():  {}", bargerProp.calculateGamma(energy));
        NT_INFO("");

        // construct the mixing matrix for current theta value
        PMNSmatrix pmns;
        pmns.setParameterValues(theta12, theta13, theta23, deltaCP);
        Tensor pmnsTensor = pmns.build();

        NT_INFO("Re[PMNS]:\n{}", pmns.build().real().toString());
        NT_INFO("Im[PMNS]:\n{}", pmns.build().imag().toString());

        // set up the matter solver
        Propagator tensorPropagator(3, baseline);

        // set up the propagator
        tensorPropagator.setMixingMatrix(pmns.build());
        tensorPropagator.setMasses(masses);
        tensorPropagator.setAntiNeutrino(antiNu);
        tensorPropagator.setEnergies(energies);

        // print put the probabilities obtained via both methods
        Tensor probabilities = tensorPropagator.calculateProbs();
        NT_INFO("#########################################################################");
        NT_INFO("Oscillation probabilities:");
        NT_INFO("[0,0] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 0, 0}),
                bargerProp.calculateProb(energy, 0, 0));
        NT_INFO("[0,1] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 0, 1}),
                bargerProp.calculateProb(energy, 0, 1));
        NT_INFO("[0,2] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 0, 2}),
                bargerProp.calculateProb(energy, 0, 2));
        NT_INFO("");
        NT_INFO("[1,0] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 1, 0}),
                bargerProp.calculateProb(energy, 1, 0));
        NT_INFO("[1,1] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 1, 1}),
                bargerProp.calculateProb(energy, 1, 1));
        NT_INFO("[1,2] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 1, 2}),
                bargerProp.calculateProb(energy, 1, 2));
        NT_INFO("");
        NT_INFO("[2,0] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 2, 0}),
                bargerProp.calculateProb(energy, 2, 0));
        NT_INFO("[2,1] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 2, 1}),
                bargerProp.calculateProb(energy, 2, 1));
        NT_INFO("[2,2] :: tensor solver {:.4f} :: barger {:.4f}", probabilities.getValue<float>({0, 2, 2}),
                bargerProp.calculateProb(energy, 2, 2));
        NT_INFO("#########################################################################");

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 2}), bargerProp.calculateProb(energy, 0, 2), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 2}), bargerProp.calculateProb(energy, 1, 2), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 0}), bargerProp.calculateProb(energy, 2, 0), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 1}), bargerProp.calculateProb(energy, 2, 1), tolerance);
        ASSERT_NEAR(probabilities.getValue<float>({0, 2, 2}), bargerProp.calculateProb(energy, 2, 2), tolerance);
    }
};

// test const density matter oscillations
TEST_P(ThreeFlavourOscillations /*unused*/, ConstDensityOscProbsNu /*unused*/)
{

    testConstDensity(/*antiNu=*/false);
}

// test const density matter oscillations for anti-neutrinos
TEST_P(ThreeFlavourOscillations /*unused*/, ConstDensityOscProbsAntiNu /*unused*/)
{

    testConstDensity(/*antiNu=*/true);
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

INSTANTIATE_TEST_CASE_P(OscProb, ThreeFlavourOscillations,
                        ::testing::Values(-M_PI, -0.8 * M_PI, -0.6 * M_PI, -0.4 * M_PI, -0.2 * M_PI, 0.0, 0.2 * M_PI,
                                          0.4 * M_PI, 0.6 * M_PI, 0.8 * M_PI, M_PI));

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)