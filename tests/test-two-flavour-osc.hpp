#pragma once

#include <gtest/gtest.h> // NOLINT
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <tests/barger-propagator.hpp>
#include <tests/printing.hpp>
#include <tests/utils.hpp>

using namespace nuTens;
using namespace nuTens::testing;

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)

class TwoFlavourOscillations : public gtest::TestWithParam<std::tuple<float, dtypes::deviceType>>
{
  protected:
    // NOLINTBEGIN(cppcoreguidelines-non-private-member-variables-in-classes)
    float theta = NAN;

    float mass1 = 0.0;
    float mass2 = 0.008 * units::eV * units::eV;
    float energy = 0.5 * units::GeV;
    float baseline = 295.0 * units::km;
    float density = 2.6;
    float tolerance = 1e-5;
    Tensor masses = Tensor();
    Tensor energies = Tensor();
    // NOLINTEND(cppcoreguidelines-non-private-member-variables-in-classes)

    // set up common values to use across tests
    void SetUp() override
    {

        dtypes::deviceType device = std::get<1>(GetParam());
        // skip if no GPU available
        SKIP_GPU(device);

        masses = Tensor({mass1, mass2}, dtypes::kComplexFloat).addBatchDim().device(device);

        energies = Tensor::ones({1, 1}, dtypes::kComplexFloat).requiresGrad(false).device(device);
        energies.setValue({0, 0}, energy);
    }

    // cognitive complexity is heavily inflated by the gtest macros
    // but they don't actually decrease readability
    // NOLINTBEGIN(readability-function-cognitive-complexity)
    void testVacuum(bool antiNu)
    {
        // get parameterised theta value
        float theta = std::get<0>(GetParam());
        dtypes::deviceType device = std::get<1>(GetParam());

        std::cout << "\n#### vacuum test for theta = " << theta << " ####" << std::endl;

        Propagator tensorPropagator = Propagator(2, device).setBaseline(baseline).setAntiNeutrino(antiNu);
        tensorPropagator.setMasses(masses);

        // will use this for baseline for comparisons

        // linter seems to struggle with recogising this type and thinks it is an int
        // and always thinks it is uninitialised
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        TwoFlavourBarger<> bargerProp{};

        bargerProp.setMass1(mass1).setMass2(mass2).setTheta(theta).setBaseline(baseline).setAntiNeutrino(antiNu);

        // construct the mixing matrix for current theta value
        Tensor PMNS = Tensor::ones({1, 2, 2}, dtypes::kComplexFloat).requiresGrad(false).device(device);
        PMNS.setValue({0, 0, 0}, std::cos(theta));
        PMNS.setValue({0, 0, 1}, -std::sin(theta));
        PMNS.setValue({0, 1, 0}, std::sin(theta));
        PMNS.setValue({0, 1, 1}, std::cos(theta));

        tensorPropagator.setMixingMatrix(PMNS);

        tensorPropagator.setEnergies(energies);

        Tensor probabilities = tensorPropagator.calculateProbs();

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0), tolerance);
    }

    void testConstDensity(bool antiNu)
    {

        // get parameterised theta value
        float theta = std::get<0>(GetParam());
        dtypes::deviceType device = std::get<1>(GetParam());

        std::cout << "\n#### const density test for theta = " << theta << " ####" << std::endl;

        Propagator tensorPropagator = Propagator(2, device).setBaseline(baseline);
        auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(2, device);
        tensorSolver->setDensity(density);

        // linter seems to struggle with recogising this type and thinks it is an int
        // and always thinks it is uninitialised
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        TwoFlavourBarger<> bargerProp{};

        bargerProp.setMass1(mass1)
            .setMass2(mass2)
            .setTheta(theta)
            .setBaseline(baseline)
            .setDensity(density)
            .setAntiNeutrino(antiNu);

        std::cout << "lMatter():   " << bargerProp.lMatter() << std::endl;
        std::cout << "ang:         " << bargerProp.calculateEffectiveAngle(energy) << std::endl;
        std::cout << "dm2:         " << bargerProp.calculateEffectiveDm2(energy) << std::endl;
        std::cout << "off-diag:    " << bargerProp.calculateProb(energy, 0, 1) << std::endl << std::endl;

        // construct the mixing matrix for current theta value
        Tensor PMNS = Tensor::ones({1, 2, 2}, dtypes::kComplexFloat).requiresGrad(false).device(device);
        PMNS.setValue({0, 0, 0}, std::cos(theta));
        PMNS.setValue({0, 0, 1}, std::sin(theta));
        PMNS.setValue({0, 1, 0}, -std::sin(theta));
        PMNS.setValue({0, 1, 1}, std::cos(theta));
        PMNS.requiresGrad(true);

        tensorPropagator.setMatterSolver(tensorSolver);
        tensorPropagator.setMixingMatrix(PMNS);
        tensorPropagator.setMasses(masses);
        tensorPropagator.setAntiNeutrino(antiNu);

        tensorPropagator.setEnergies(energies);

        BaseMatterSolver::EigenvalTensor eigenVals;
        BaseMatterSolver::EigenvecTensor eigenVecs;

        tensorSolver->calculateEigenvalues(eigenVecs, eigenVals);

        // first check that the effective dM^2 from the tensor solver is what we
        // expect
        std::cout << "tensorSolver eigenvals: " << std::endl;
        std::cout << eigenVals << std::endl;
        auto calcV1 = eigenVals.getValue<float>({0, 0});
        auto calcV2 = eigenVals.getValue<float>({0, 1});
        float effDm2 = (calcV1 - calcV2) * 2.0 * energy;

        ASSERT_NEAR(effDm2, bargerProp.calculateEffectiveDm2(energy), tolerance);

        // now check the actual mixing matrix entries
        Tensor PMNSeff = Tensor::matmul(PMNS, eigenVecs);

        ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 0, 0})), std::abs(bargerProp.getPMNSelement(energy, 0, 0)),
                    tolerance);

        ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 1, 1})), std::abs(bargerProp.getPMNSelement(energy, 1, 1)),
                    tolerance);

        ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 0, 1})), std::abs(bargerProp.getPMNSelement(energy, 0, 1)),
                    tolerance);

        ASSERT_NEAR(std::abs(PMNSeff.getValue<float>({0, 1, 0})), std::abs(bargerProp.getPMNSelement(energy, 1, 0)),
                    tolerance);

        Tensor probabilities = tensorPropagator.calculateProbs();

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1), tolerance);

        ASSERT_NEAR(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0), tolerance);
    }
    // NOLINTEND(readability-function-cognitive-complexity)
};

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)