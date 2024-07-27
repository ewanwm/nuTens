#include <iostream>
#include <tests/barger-propagator.hpp>
#include <tests/test-utils.hpp>

// who tests the testers???

using namespace Testing;

int main()
{
    float baseline = 500.0;

    TwoFlavourBarger bargerProp{};

    // ##########################################################
    // ## Test vacuum propagations for some fixed param values ##
    // ##########################################################

    // check that we get no vacuum oscillations when theta == 0 for a range of
    // energies
    bargerProp.setParams(/*m1=*/1.0, /*m2=*/2.0, /*theta=*/0.0, baseline);
    for (int i = 0; i < 100; i++)
    {
        float energy = (float)i * 10.0;

        if (relativeDiff(bargerProp.calculateProb(energy, 0, 0), 1.0) > 0.00001 ||
            relativeDiff(bargerProp.calculateProb(energy, 1, 1), 1.0) > 0.00001 ||
            relativeDiff(bargerProp.calculateProb(energy, 0, 1), 0.0) > 0.00001 ||
            relativeDiff(bargerProp.calculateProb(energy, 1, 0), 0.0) > 0.00001)
        {
            std::cerr << "ERROR: probabilities for theta == 0 should be "
                         "identity matrix"
                      << std::endl;
            std::cerr << "       This was not the case for energy = " << energy << std::endl;
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
            return 1;
        }
    }

    // check that we get no vacuum oscillations when m1 == m2 for a range of
    // energies
    bargerProp.setParams(/*m1=*/1.0, /*m2=*/1.0, /*theta=*/M_PI / 3.0, baseline);
    for (int i = 0; i < 100; i++)
    {
        float energy = (float)i * 10.0;

        if (relativeDiff(bargerProp.calculateProb(energy, 0, 0), 1.0) > 0.00001 ||
            relativeDiff(bargerProp.calculateProb(energy, 1, 1), 1.0) > 0.00001 ||
            relativeDiff(bargerProp.calculateProb(energy, 0, 1), 0.0) > 0.00001 ||
            relativeDiff(bargerProp.calculateProb(energy, 1, 0), 0.0) > 0.00001)
        {
            std::cerr << "ERROR: probabilities for m1 == m2 should be identity matrix" << std::endl;
            std::cerr << "       This was not the case for energy = " << energy << std::endl;
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
            return 1;
        }
    }

    // now check for fixed parameters values against externally calculated values

    // theta = pi/8, m1 = 1, m2 = 2, E = 3, L = 4
    // => prob_(alpha != beta) = sin^2(Pi/4) * sin^2(1) = 0.35403670913
    //    prob_(alpha == beta) =      1 - 0.35403670913 = 0.64596329086

    bargerProp.setParams(/*m1=*/1.0, /*m2=*/2.0, /*theta=*/M_PI / 8.0,
                         /*baseline=*/4.0);

    TEST_EXPECTED(bargerProp.calculateProb(3.0, 0, 0), 0.64596329086, "probability for alpha == beta == 0", 0.00001)

    TEST_EXPECTED(bargerProp.calculateProb(3.0, 1, 1), 0.64596329086, "probability for alpha == beta == 1", 0.00001)

    TEST_EXPECTED(bargerProp.calculateProb(3.0, 0, 1), 0.35403670913, "probability for alpha == 0, beta == 1", 0.00001)

    TEST_EXPECTED(bargerProp.calculateProb(3.0, 1, 0), 0.35403670913, "probability for alpha == 1, beta == 0", 0.00001)

    // ##############################################################
    // ## Now test matter propagations for some fixed param values ##
    // ##############################################################

    // theta = 0.24pi, m1 = 1.5, m2 = 1.7, E = 5, L = 6, density = 7
    // lv = 4pi * E / dm^2 = 98.1747704247
    // lm = 2pi / ( sqrt(2) * G * density ) = 5882.49338759
    // gamma = atan( sin( 2theta ) / (cos( 2theta ) - lv / lm) ) / 2.0
    //       = atan(21.648602992) / 2 = 0.762318391 rad
    // dM2 = dm^2 * sqrt( 1 - 2 * (lv / lm) * cos(2theta) + (lv / lm)^2)
    //     = 0.63941819056
    //
    // => prob_(alpha != beta) = sin^2(2*gamma) * sin^2((L / E) * dM2/4 )
    //                         = 0.03627048316
    //    prob_(alpha == beta) =      1 - 0.03627048316 = 0.96372951684

    bargerProp.setParams(/*m1=*/1.7, /*m2=*/1.5, /*theta=*/0.24 * M_PI,
                         /*baseline=*/6.0, /*density=*/7.0);

    TEST_EXPECTED(bargerProp.lv(5.0), 98.1747704247, "vacuum osc length", 0.00001)

    TEST_EXPECTED(bargerProp.lm(), 5882.49338759, "matter osc length", 0.00001)

    TEST_EXPECTED(bargerProp.calculateEffectiveAngle(5.0), 0.762318391, "effective mixing angle", 0.00001)

    TEST_EXPECTED(bargerProp.calculateEffectiveDm2(5.0), 0.63941819056, "effective m^2 diff", 0.00001)

    TEST_EXPECTED(bargerProp.calculateProb(5.0, 0, 0), 0.96372951684, "probability for alpha == beta == 0", 0.00001)

    TEST_EXPECTED(bargerProp.calculateProb(5.0, 1, 1), 0.96372951684, "probability for alpha == beta == 1", 0.00001)

    TEST_EXPECTED(bargerProp.calculateProb(5.0, 0, 1), 0.03627048316, "probability for alpha == 0, beta == 1", 0.00001)

    TEST_EXPECTED(bargerProp.calculateProb(5.0, 1, 0), 0.03627048316, "probability for alpha == 1, beta == 0", 0.00001)
}