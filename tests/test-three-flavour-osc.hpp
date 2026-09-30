#pragma once

#include <gtest/gtest.h> // NOLINT
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/pmns-matrix.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <nuTens/testing/barger-propagator.hpp>
#include <nuTens/testing/printing.hpp>
#include <nuTens/testing/utils.hpp>

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
        energies = Tensor({energy}, dtypes::kComplexDouble).device(device);
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

void testBatching(dtypes::scalarType scalarType, dtypes::deviceType deviceType, bool antiNu)
{

    float theta12 = 0.12 * M_PI;
    float theta13 = 0.13 * M_PI;
    float deltaCP = 0.25 * M_PI;

    float mass1 = 0.0;
    float mass2 = 0.008 * units::eV * units::eV;
    float mass3 = 0.01 * units::eV * units::eV;

    float energy = 0.5 * units::GeV;
    float baseline = 295.0 * units::km;
    float density = 2.6;

    Tensor massTensor = Tensor::zeros({10, 3}, scalarType, deviceType);
    auto theta23s = std::vector<float>(10);
    auto theta12s = std::vector<float>(10);
    auto theta13s = std::vector<float>(10);
    auto deltaCPs = std::vector<float>(10);

    // linter seems to struggle with recogising this type and thinks it is an int
    // and always thinks it is uninitialised
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ThreeFlavourBarger<> bargerProp{};
    bargerProp.setMass1(mass1)
        .setMass2(mass2)
        .setMass3(mass3)
        .setTheta12(theta12)
        .setTheta13(theta13)
        .setDeltaCP(deltaCP)
        .setBaseline(baseline)
        .setDensity(density)
        .setAntiNeutrino(antiNu);

    auto bargerProbs = std::array<std::array<std::array<float, 3>, 3>, 10>();

    for (int iTheta = 0; iTheta < 10; iTheta++)
    {
        for (int iLep = 0; iLep < 3; iLep++)
        {
            for (int jLep = 0; jLep < 3; jLep++)
            {
                float theta23 = -M_PI + 2.0 * M_PI * (float)iTheta / (float)10;

                // calculate prob using the barger propagator
                bargerProp.setTheta23(theta23);
                bargerProbs[iTheta][iLep][jLep] = bargerProp.calculateProb(energy, iLep, jLep);

                // add the theta value to the list
                theta23s[iTheta] = theta23;

                // push back other values
                massTensor.setValue({iTheta, 0}, mass1);
                massTensor.setValue({iTheta, 1}, mass2);
                massTensor.setValue({iTheta, 2}, mass3);
                deltaCPs[iTheta] = deltaCP;
                theta12s[iTheta] = theta12;
                theta13s[iTheta] = theta13;
            }
        }
    }

    std::cout << "done with barger" << std::endl;

    // now calculate osc probs using nuTens propagator
    PMNSmatrix pmns(/*device=*/deviceType, /*batchSize=*/10);
    std::cout << "created PMNSmatrix" << std::endl;
    pmns.setTheta12(theta12s).setTheta13(theta13s).setTheta23(theta23s).setDeltaCP(deltaCPs);
    std::cout << "set PMNSmatrix values" << std::endl;
    Tensor pmnsTensor = pmns.build();

    std::cout << "got pmns matrix" << std::endl;

    // set up the matter solver
    Propagator tensorPropagator = Propagator(3, deviceType, /*batchSize=*/10).setBaseline(baseline);

    std::cout << "made propagator" << std::endl;

    auto tensorSolver = std::make_shared<ConstDensityMatterSolver>(3, deviceType, 10);

    std::cout << "made matter solver" << std::endl;
    tensorSolver->setDensity(density);

    // set up the propagator
    tensorPropagator.setMatterSolver(tensorSolver);
    tensorPropagator.setMixingMatrix(pmns.build());
    std::cout << "set mixing matrix" << std::endl;
    tensorPropagator.setMasses(massTensor);
    tensorPropagator.setAntiNeutrino(antiNu);

    auto energies = Tensor({energy}, dtypes::kComplexFloat, deviceType);
    tensorPropagator.setEnergies(energies);

    auto oscProbs = tensorPropagator.calculateProbs();

    std::cout << "osc probs: " << oscProbs << std::endl;

    for (int iTheta = 0; iTheta < 10; iTheta++)
    {
        for (int iLep = 0; iLep < 3; iLep++)
        {
            for (int jLep = 0; jLep < 3; jLep++)
            {
                ASSERT_NEAR(oscProbs.getValue<float>({iTheta, iLep, jLep}), bargerProbs[iTheta][iLep][jLep], 1e-6);
            }
        }
    }
}

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)