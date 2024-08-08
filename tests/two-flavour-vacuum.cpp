#include <nuTens/propagator/propagator.hpp>
#include <tests/barger-propagator.hpp>
#include <tests/test-utils.hpp>

using namespace Testing;

int main()
{

    NT_PROFILE_BEGINSESSION("two-flavour-vacuum-test");

    NT_PROFILE();

    float m1 = 0.1;
    float m2 = 0.5;
    float energy = 1.0;
    float baseline = 0.5;

    Tensor masses = Tensor({m1, m2}, NTdtypes::kFloat).addBatchDim().requiresGrad(true);

    Tensor energies = Tensor::ones({1, 1}, NTdtypes::kFloat).requiresGrad(false);
    energies.setValue({0, 0}, energy);
    energies.requiresGrad(true);

    Propagator tensorPropagator(2, baseline);
    tensorPropagator.setMasses(masses);

    // will use this for baseline for comparisons
    TwoFlavourBarger bargerProp{};

    // test that Propagator gives expected oscillation probabilites for a range
    // of thetas
    for (int i = 0; i < 20; i++)
    {
        float theta = (-1.0 + 2.0 * (float)i / 20.0) * 0.49 * M_PI;

        bargerProp.setParams(m1, m2, theta, baseline);

        // construct the PMNS matrix for current theta value
        Tensor PMNS = Tensor::ones({1, 2, 2}, NTdtypes::kComplexFloat).requiresGrad(false);
        PMNS.setValue({0, 0, 0}, std::cos(theta));
        PMNS.setValue({0, 0, 1}, -std::sin(theta));
        PMNS.setValue({0, 1, 0}, std::sin(theta));
        PMNS.setValue({0, 1, 1}, std::cos(theta));
        PMNS.requiresGrad(true);

        tensorPropagator.setPMNS(PMNS);

        Tensor probabilities = tensorPropagator.calculateProbs(energies);

        TEST_EXPECTED(probabilities.getValue<float>({0, 0, 0}), bargerProp.calculateProb(energy, 0, 0),
                      "probability for alpha == beta == 0", 0.00001)

        TEST_EXPECTED(probabilities.getValue<float>({0, 1, 1}), bargerProp.calculateProb(energy, 1, 1),
                      "probability for alpha == beta == 1", 0.00001)

        TEST_EXPECTED(probabilities.getValue<float>({0, 0, 1}), bargerProp.calculateProb(energy, 0, 1),
                      "probability for alpha == 0, beta == 1", 0.00001)

        TEST_EXPECTED(probabilities.getValue<float>({0, 1, 0}), bargerProp.calculateProb(energy, 1, 0),
                      "probability for alpha == 1, beta == 0", 0.00001)
    }

    NT_PROFILE_ENDSESSION();
}