#include <iostream>
#include <nuTens/propagator/units.hpp>
#include <tests/barger-propagator.hpp>

#include <gtest/gtest.h>

// who tests the testers???

using namespace nuTens;
using namespace nuTens::testing;

TEST(TwoFlavourBargerPropTest, zeroThetaNoOscTest) {
  
    constexpr float baseline = 5.0e12;

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
  
    constexpr float baseline = 5.0e12;

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

    // theta = pi/8, dm^2 = 0.01 eV E = 1 GeV L = 100 km
    // => prob_(alpha != beta) = sin^2(2 theta) * sin^2( 1.27 * dm^2 * L / E [ eV^2 km / GeV] )
    //
    //                         = sin^2(Pi/4) * sin^2( 1.27 * 0.01 * 100 / 1 ) = 0.4561088222
    // 
    //    prob_(alpha == beta) =      1 - 0.4561088222 = 0.5438911778

    bargerProp.setParams(/*m1=*/0.0, /*m2=*/0.1, /*theta=*/M_PI / 8.0,
                         /*baseline=*/100.0 * units::km );

    ASSERT_NEAR(bargerProp.calculateProb(1.0 * units::GeV, 0, 0), 0.5438911778, 1e-3);

    ASSERT_NEAR(bargerProp.calculateProb(1.0 * units::GeV, 1, 1), 0.5438911778, 1e-3);

    ASSERT_NEAR(bargerProp.calculateProb(1.0 * units::GeV, 0, 1), 0.4561088222, 1e-3);

    ASSERT_NEAR(bargerProp.calculateProb(1.0 * units::GeV, 1, 0), 0.4561088222, 1e-3);


    // ##############################################################
    // ## Now test matter propagations for some fixed param values ##
    // ##############################################################

    // theta = 0.24, m1 = 0.04eV, m2 = 0.001eV, E = 1GeV, L = 250 km, density = 2
    // lv = 4pi * E / dm^2 = 7.8588934e+12 
    // lm = 2pi / ( sqrt(2) * G * density ) = 4.1177454e+13 
    // gamma = atan( sin( 2theta ) / (cos( 2theta ) - lv / lm) ) / 2.0
    //       = atan(0.663342 ) / 2 = 0.292848614 rad
    // dM2 = dm^2 * sqrt( 1 - 2 * (lv / lm) * cos(2theta) + (lv / lm)^2)
    //     = 0.001599 * sqrt ( 1 - 2 * 0.19085428 * 0.8869949 +  0.19085428 ^2)
    //     = 0.001335765
    //
    // => prob_(alpha != beta) = sin^2(2*gamma) * sin^2( 1.27 (L / E) * dM2 )
    //                         = sin^2( 2 * 0.292848614 ) * sin^2( 1.27 * 250 / 1 * 0.001599)
    //                         = 0.0517436
    //    prob_(alpha == beta) =      1 - 0.0517436  = 0.9482564

    bargerProp.setParams(/*m1=*/0.04, /*m2=*/0.001, /*theta=*/0.24,
                         /*baseline=*/250 * units::km, /*density=*/2.0);

    ASSERT_NEAR(bargerProp.lv(1.0e9), 7.8588934e+12, 1e6) << "vacuum osc length";

    ASSERT_NEAR(bargerProp.lm(), 4.1177454e+13 , 1e6) <<  "matter osc length";

    ASSERT_NEAR(bargerProp.calculateEffectiveAngle(1.0e9), 0.292848614, 0.00001) << "effective mixing angle";

    ASSERT_NEAR(bargerProp.calculateEffectiveDm2(1.0e9), 0.001335765, 0.00001) << "effective m^2 diff";

    ASSERT_NEAR(bargerProp.calculateProb(1.0e9, 0, 0), 0.9482564, 1e-3) << "probability for alpha == beta == 0";

    ASSERT_NEAR(bargerProp.calculateProb(1.0e9, 1, 1), 0.9482564, 1e-3) << "probability for alpha == beta == 1";

    ASSERT_NEAR(bargerProp.calculateProb(1.0e9, 0, 1), 0.0517436, 1e-3) << "probability for alpha == 0, beta == 1";

    ASSERT_NEAR(bargerProp.calculateProb(1.0e9, 1, 0), 0.0517436, 1e-3) << "probability for alpha == 1, beta == 0";
}