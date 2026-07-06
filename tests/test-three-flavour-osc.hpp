#pragma once

#include <gtest/gtest.h> // NOLINT
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/pmns-matrix.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <tests/barger-propagator.hpp>
#include <tests/printing.hpp>
#include <tests/utils.hpp>

using namespace nuTens;
using namespace nuTens::testing;

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
class ThreeFlavourOscillations : public gtest::TestWithParam<std::tuple<float, dtypes::deviceType>>
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
    void SetUp() override
    {
        dtypes::deviceType device = std::get<1>(GetParam());

        // skip GPU tests if no GPU available
        SKIP_GPU(device);

        masses = Tensor({mass1, mass2, mass3}, dtypes::kComplexDouble).addBatchDim().device(device);
        energies = Tensor({energy}, dtypes::kComplexDouble).addBatchDim().device(device);
    }

    // cognitive complexity is heavily inflated by the gtest macros
    // but they don't actually decrease readability
    // NOLINTBEGIN(readability-function-cognitive-complexity)
    void testConstDensityEvals(bool antiNu)
    {

        // get parameterised theta value
        theta12 = std::get<0>(GetParam());
        dtypes::deviceType device = std::get<1>(GetParam());

        // set up the barger propagator

        // linter seems to struggle with recogising this type and thinks it is an int
        // and always thinks it is uninitialised
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ThreeFlavourBarger<> bargerProp{};
        bargerProp.setMass1(mass1)
            .setMass2(mass2)
            .setMass3(mass3)
            .setTheta12(theta12)
            .setTheta13(theta13)
            .setTheta23(theta23)
            .setDeltaCP(deltaCP)
            .setBaseline(baseline)
            .setDensity(density)
            .setAntiNeutrino(antiNu);

        // construct the mixing matrix for current theta value
        PMNSmatrix pmns(device);
        pmns.setTheta12(theta12).setTheta13(theta13).setTheta23(theta23).setDeltaCP(deltaCP);
        Tensor pmnsTensor = pmns.build();

        // set up the matter solver
        Propagator tensorPropagator = Propagator(3, device).setBaseline(baseline);
        auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(3, device);
        tensorSolver->setDensity(density);

        // set up the propagator
        tensorPropagator.setMatterSolver(tensorSolver);
        tensorPropagator.setMixingMatrix(pmns.build());
        tensorPropagator.setMasses(masses);
        tensorPropagator.setAntiNeutrino(antiNu);
        tensorPropagator.setEnergies(energies);

        BaseMatterSolver::EigenvalTensor eigenVals;
        BaseMatterSolver::EigenvecTensor eigenVecs;

        tensorSolver->calculateEigenvalues(eigenVecs, eigenVals);

        // first check that the effective dM^2 from the tensor solver is what we
        // expect
        auto calcV1 = eigenVals.getValue<float>({0, 0}) * 2.0 * energy;
        auto calcV2 = eigenVals.getValue<float>({0, 1}) * 2.0 * energy;
        auto calcV3 = eigenVals.getValue<float>({0, 2}) * 2.0 * energy;

        // Compare effective masses from both methods
        NT_INFO("M1: tensor solver: {:.7f} :: barger: {:.7f}", calcV1, bargerProp.calculateEffectiveM2(energy, 0));
        NT_INFO("M2: tensor solver: {:.7f} :: barger: {:.7f}", calcV2, bargerProp.calculateEffectiveM2(energy, 1));
        NT_INFO("M3: tensor solver: {:.7f} :: barger: {:.7f}", calcV3, bargerProp.calculateEffectiveM2(energy, 2));

        // Effective dM^2's
        ASSERT_NEAR(calcV2 - calcV1,
                    bargerProp.calculateEffectiveM2(energy, 1) - bargerProp.calculateEffectiveM2(energy, 0), tolerance);
        ASSERT_NEAR(calcV3 - calcV2,
                    bargerProp.calculateEffectiveM2(energy, 2) - bargerProp.calculateEffectiveM2(energy, 1), tolerance);
        ASSERT_NEAR(calcV3 - calcV1,
                    bargerProp.calculateEffectiveM2(energy, 2) - bargerProp.calculateEffectiveM2(energy, 0), tolerance);
    }

    void testConstDensityHamiltonian(bool antiNu)
    {

        // get parameterised theta value
        theta12 = std::get<0>(GetParam());
        dtypes::deviceType device = std::get<1>(GetParam());

        NT_INFO("\n#### const density test for theta12 = {} ####", theta12);

        // set up the barger propagator

        // linter seems to struggle with recogising this type and thinks it is an int
        // and always thinks it is uninitialised
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ThreeFlavourBarger<> bargerProp{};
        bargerProp.setMass1(mass1)
            .setMass2(mass2)
            .setMass3(mass3)
            .setTheta12(theta12)
            .setTheta13(theta13)
            .setTheta23(theta23)
            .setDeltaCP(deltaCP)
            .setBaseline(baseline)
            .setDensity(density)
            .setAntiNeutrino(antiNu);

        // construct the mixing matrix for current theta value
        PMNSmatrix pmns(device);
        pmns.setTheta12(theta12).setTheta13(theta13).setTheta23(theta23).setDeltaCP(deltaCP);
        Tensor pmnsTensor = pmns.build();

        // set up the matter solver
        Propagator tensorPropagator = Propagator(3, device).setBaseline(baseline);
        auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(3, device);
        tensorSolver->setDensity(density);

        // set up the propagator
        tensorPropagator.setMatterSolver(tensorSolver);
        tensorPropagator.setMixingMatrix(pmns.build());
        tensorPropagator.setMasses(masses);
        tensorPropagator.setAntiNeutrino(antiNu);
        tensorPropagator.setEnergies(energies);

        // compare the hamiltonians from both methods
        Tensor hamiltonianTensor = tensorSolver->getHamiltonian();
        ASSERT_NEAR(hamiltonianTensor.real().getValue<float>({0, 0, 0}),
                    bargerProp.getHamiltonianElement(energy, 0, 0).real(), tolerance);
        ASSERT_NEAR(hamiltonianTensor.imag().getValue<float>({0, 0, 0}),
                    bargerProp.getHamiltonianElement(energy, 0, 0).imag(), tolerance);

        ASSERT_NEAR(hamiltonianTensor.real().getValue<float>({0, 0, 1}),
                    bargerProp.getHamiltonianElement(energy, 0, 1).real(), tolerance);
        ASSERT_NEAR(hamiltonianTensor.imag().getValue<float>({0, 0, 1}),
                    bargerProp.getHamiltonianElement(energy, 0, 1).imag(), tolerance);

        ASSERT_NEAR(hamiltonianTensor.real().getValue<float>({0, 0, 2}),
                    bargerProp.getHamiltonianElement(energy, 0, 2).real(), tolerance);
        ASSERT_NEAR(hamiltonianTensor.imag().getValue<float>({0, 0, 2}),
                    bargerProp.getHamiltonianElement(energy, 0, 2).imag(), tolerance);

        ASSERT_NEAR(hamiltonianTensor.real().getValue<float>({0, 1, 0}),
                    bargerProp.getHamiltonianElement(energy, 1, 0).real(), tolerance);
        ASSERT_NEAR(hamiltonianTensor.imag().getValue<float>({0, 1, 0}),
                    bargerProp.getHamiltonianElement(energy, 1, 0).imag(), tolerance);

        ASSERT_NEAR(hamiltonianTensor.real().getValue<float>({0, 1, 1}),
                    bargerProp.getHamiltonianElement(energy, 1, 1).real(), tolerance);
        ASSERT_NEAR(hamiltonianTensor.imag().getValue<float>({0, 1, 1}),
                    bargerProp.getHamiltonianElement(energy, 1, 1).imag(), tolerance);

        ASSERT_NEAR(hamiltonianTensor.real().getValue<float>({0, 1, 2}),
                    bargerProp.getHamiltonianElement(energy, 1, 2).real(), tolerance);
        ASSERT_NEAR(hamiltonianTensor.imag().getValue<float>({0, 1, 2}),
                    bargerProp.getHamiltonianElement(energy, 1, 2).imag(), tolerance);

        ASSERT_NEAR(hamiltonianTensor.real().getValue<float>({0, 2, 0}),
                    bargerProp.getHamiltonianElement(energy, 2, 0).real(), tolerance);
        ASSERT_NEAR(hamiltonianTensor.imag().getValue<float>({0, 2, 0}),
                    bargerProp.getHamiltonianElement(energy, 2, 0).imag(), tolerance);

        ASSERT_NEAR(hamiltonianTensor.real().getValue<float>({0, 2, 1}),
                    bargerProp.getHamiltonianElement(energy, 2, 1).real(), tolerance);
        ASSERT_NEAR(hamiltonianTensor.imag().getValue<float>({0, 2, 1}),
                    bargerProp.getHamiltonianElement(energy, 2, 1).imag(), tolerance);

        ASSERT_NEAR(hamiltonianTensor.real().getValue<float>({0, 2, 2}),
                    bargerProp.getHamiltonianElement(energy, 2, 2).real(), tolerance);
        ASSERT_NEAR(hamiltonianTensor.imag().getValue<float>({0, 2, 2}),
                    bargerProp.getHamiltonianElement(energy, 2, 2).imag(), tolerance);
    }

    void testConstDensityOscillations(bool antiNu)
    {

        // get parameterised theta value
        theta12 = std::get<0>(GetParam());
        dtypes::deviceType device = std::get<1>(GetParam());

        NT_INFO("\n#### const density test for theta12 = {} ####", theta12);

        // set up the barger propagator

        // linter seems to struggle with recogising this type and thinks it is an int
        // and always thinks it is uninitialised
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ThreeFlavourBarger<> bargerProp{};
        bargerProp.setMass1(mass1)
            .setMass2(mass2)
            .setMass3(mass3)
            .setTheta12(theta12)
            .setTheta13(theta13)
            .setTheta23(theta23)
            .setDeltaCP(deltaCP)
            .setBaseline(baseline)
            .setDensity(density)
            .setAntiNeutrino(antiNu);

        NT_INFO("alpha():  {}", bargerProp.calculateAlpha(energy));
        NT_INFO("beta():   {}", bargerProp.calculateBeta(energy));
        NT_INFO("gamma():  {}", bargerProp.calculateGamma(energy));
        NT_INFO("");

        // construct the mixing matrix for current theta value
        PMNSmatrix pmns(device);
        pmns.setTheta12(theta12).setTheta13(theta13).setTheta23(theta23).setDeltaCP(deltaCP);
        Tensor pmnsTensor = pmns.build();

        NT_INFO("Re[PMNS]:\n{}", pmns.build().real().toString());
        NT_INFO("Im[PMNS]:\n{}", pmns.build().imag().toString());

        // set up the matter solver
        Propagator tensorPropagator = Propagator(3, device).setBaseline(baseline);
        auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(3, device);
        tensorSolver->setDensity(density);

        // set up the propagator
        tensorPropagator.setMatterSolver(tensorSolver);
        tensorPropagator.setMixingMatrix(pmns.build());
        tensorPropagator.setMasses(masses);
        tensorPropagator.setAntiNeutrino(antiNu);
        tensorPropagator.setEnergies(energies);

        // print put the probabilities obtained via both methods
        Tensor probabilities = tensorPropagator.calculateProbs();

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
        theta12 = std::get<0>(GetParam());
        dtypes::deviceType device = std::get<1>(GetParam());

        NT_INFO("\n#### vacuum test for theta12 = {} ####", theta12);

        // set up the barger propagator

        // linter seems to struggle with recogising this type and thinks it is an int
        // and always thinks it is uninitialised
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ThreeFlavourBarger<> bargerProp{};
        bargerProp.setMass1(mass1)
            .setMass2(mass2)
            .setMass3(mass3)
            .setTheta12(theta12)
            .setTheta13(theta13)
            .setTheta23(theta23)
            .setDeltaCP(deltaCP)
            .setBaseline(baseline)
            .setDensity(density = -999.9)
            .setAntiNeutrino(antiNu);

        NT_INFO("alpha():  {}", bargerProp.calculateAlpha(energy));
        NT_INFO("beta():   {}", bargerProp.calculateBeta(energy));
        NT_INFO("gamma():  {}", bargerProp.calculateGamma(energy));
        NT_INFO("");

        // construct the mixing matrix for current theta value
        PMNSmatrix pmns(device);
        pmns.setTheta12(theta12).setTheta13(theta13).setTheta23(theta23).setDeltaCP(deltaCP);
        Tensor pmnsTensor = pmns.build();

        NT_INFO("Re[PMNS]:\n{}", pmns.build().real().toString());
        NT_INFO("Im[PMNS]:\n{}", pmns.build().imag().toString());

        // set up the matter solver
        Propagator tensorPropagator = Propagator(3, device).setBaseline(baseline);

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
    // NOLINTEND(readability-function-cognitive-complexity)
};

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)