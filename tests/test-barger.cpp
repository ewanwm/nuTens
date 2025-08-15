#include <iostream>
#include <nuTens/propagator/units.hpp>
#include <tests/barger-propagator.hpp>

#include <gtest/gtest.h>

// who tests the testers???

using namespace nuTens;
using namespace nuTens::testing;

TEST(TwoFlavourBargerPropTest, zeroThetaNoOscTest) {
  
    constexpr float baseline = 500.0 * units::km;

    TwoFlavourBarger bargerProp{};

    // ##########################################################
    // ## Test vacuum propagations for some fixed param values ##
    // ##########################################################

    // check that we get no vacuum oscillations when theta == 0 for a range of
    // energies
    bargerProp.setParams(/*m1=*/1.0, /*m2=*/2.0, /*theta=*/0.0, baseline);
    
    for (int iEnergy = 1; iEnergy < 100; iEnergy++)
    {
        float energy = (float)iEnergy * units::GeV / 10.0;

        EXPECT_EQ(bargerProp.calculateProb(energy, 0, 0), 1.0);
        EXPECT_EQ(bargerProp.calculateProb(energy, 1, 1), 1.0);
        EXPECT_EQ(bargerProp.calculateProb(energy, 0, 1), 0.0);
        EXPECT_EQ(bargerProp.calculateProb(energy, 1, 0), 0.0);
    }
}

TEST(TwoFlavourBargerPropTest, zeroDmsqNoOscTest) {
  
    constexpr float baseline = 500.0 * units::km;

    TwoFlavourBarger bargerProp{};

    // ##########################################################
    // ## Test vacuum propagations for some fixed param values ##
    // ##########################################################

    // check that we get no vacuum oscillations when theta == 0 for a range of
    // energies
    bargerProp.setParams(/*m1=*/1.0, /*m2=*/1.0, /*theta=*/M_PI / 4.0, baseline);
    
    for (int iEnergy = 1; iEnergy < 100; iEnergy++)
    {
        float energy = (float)iEnergy * units::GeV / 10.0;

        EXPECT_EQ(bargerProp.calculateProb(energy, 0, 0), 1.0);
        EXPECT_EQ(bargerProp.calculateProb(energy, 1, 1), 1.0);
        EXPECT_EQ(bargerProp.calculateProb(energy, 0, 1), 0.0);
        EXPECT_EQ(bargerProp.calculateProb(energy, 1, 0), 0.0);
    }
}

TEST(TwoFlavourBargerPropTest, fixedValuesTest) {
  
    TwoFlavourBarger bargerProp{};

    // now check for fixed parameters values against externally calculated values

    // theta = pi/8, m1 = 1, m2 = 2, E = 3, L = 4
    // => prob_(alpha != beta) = sin^2(Pi/4) * sin^2(1) = 0.35403670913
    //    prob_(alpha == beta) =      1 - 0.35403670913 = 0.64596329086

    bargerProp.setParams(/*m1=*/1.0, /*m2=*/2.0, /*theta=*/M_PI / 8.0,
                         /*baseline=*/4.0);

    ASSERT_NEAR(bargerProp.calculateProb(3.0, 0, 0), 0.64596329086, 1e-5);

    ASSERT_NEAR(bargerProp.calculateProb(3.0, 1, 1), 0.64596329086, 1e-5);

    ASSERT_NEAR(bargerProp.calculateProb(3.0, 0, 1), 0.35403670913, 1e-5);

    ASSERT_NEAR(bargerProp.calculateProb(3.0, 1, 0), 0.35403670913, 1e-5);


    // ##############################################################
    // ## Now test matter propagations for some fixed param values ##
    // ##############################################################

    // theta = 0.24, m1 = 0.04eV, m2 = 0.001eV, E = 1GeV, L = 500km, density = 2
    // lv = 4pi * E / dm^2 = 7.8588934e+12 
    // lm = 2pi / ( sqrt(2) * G * density ) = 2.0588727e+13 
    // gamma = atan( sin( 2theta ) / (cos( 2theta ) - lv / lm) ) / 2.0
    //       = atan(0.91389598537 ) / 2 = 0.370219805 rad
    // dM2 = dm^2 * sqrt( 1 - 2 * (lv / lm) * cos(2theta) + (lv / lm)^2)
    //     = 0.00109453
    //
    // => prob_(alpha != beta) = sin^2(2*gamma) * sin^2((L / E) * dM2/4 )
    //                         = 0.186410
    //    prob_(alpha == beta) =      1 - 0.186410  = 0.81359

    bargerProp.setParams(/*m1=*/0.04, /*m2=*/0.001, /*theta=*/0.24,
                         /*baseline=*/500.0 * units::km, /*density=*/2.0);

    ASSERT_NEAR(bargerProp.lv(1.0 * units::GeV), 7.8588934e+12, 1e6) << "vacuum osc length";

    ASSERT_NEAR(bargerProp.lm(), 2.0588727e+13, 1e6) <<  "matter osc length";

    ASSERT_NEAR(bargerProp.calculateEffectiveAngle(1.0 * units::GeV), 0.370219805, 0.00001) << "effective mixing angle";

    ASSERT_NEAR(bargerProp.calculateEffectiveDm2(1.0 * units::GeV), 0.00109453, 0.00001) << "effective m^2 diff";

    ASSERT_NEAR(bargerProp.calculateProb(1.0 * units::GeV, 0, 0), 0.81359, 0.00001) << "probability for alpha == beta == 0";

    ASSERT_NEAR(bargerProp.calculateProb(1.0 * units::GeV, 1, 1), 0.81359, 0.00001) << "probability for alpha == beta == 1";

    ASSERT_NEAR(bargerProp.calculateProb(1.0 * units::GeV, 0, 1), 0.186410, 0.00001) << "probability for alpha == 0, beta == 1";

    ASSERT_NEAR(bargerProp.calculateProb(1.0 * units::GeV, 1, 0), 0.186410, 0.00001) << "probability for alpha == 1, beta == 0";
}