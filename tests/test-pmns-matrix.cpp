#include <gtest/gtest.h> // NOLINT
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <iostream>
#include <nuTens/propagator/pmns-matrix.hpp>

// magic numbers are fine for testing!
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
using namespace nuTens;

class PMNSmatrixTest : public gtest::TestWithParam<float>
{

  protected:
    // NOLINTBEGIN(cppcoreguidelines-non-private-member-variables-in-classes)
    float theta12 = 1.2 * M_PI;
    float theta23 = 2.3 * M_PI;
    float theta13 = 1.3 * M_PI;
    float deltaCP = 0.5 * M_PI;

    PMNSmatrix matrix;

    Tensor matrixTensor;
    // NOLINTEND(cppcoreguidelines-non-private-member-variables-in-classes)

    // set up common values to use across tests
    void SetUp()
    {

        matrix.setParameterValues(theta12, theta13, theta23, deltaCP);

        matrixTensor = matrix.build();
    }
};

TEST_F(PMNSmatrixTest /*unused*/, FixedValuesTest_Ue1 /*unused*/)
{

    ASSERT_EQ(matrixTensor.getValue<float>({0, 0, 0}), std::cos(theta12) * std::cos(theta13));
}

TEST_F(PMNSmatrixTest /*unused*/, FixedValuesTest_Ue2 /*unused*/)
{

    ASSERT_EQ(matrixTensor.getValue<float>({0, 0, 1}), std::sin(theta12) * std::cos(theta13));
}

TEST_F(PMNSmatrixTest /*unused*/, FixedValuesTest_Ue3 /*unused*/)
{

    std::complex<float> Ue3 = std::sin(theta13) * std::exp(std::complex<float>(0.0, -1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0, 0, 2}), Ue3);
}

TEST_F(PMNSmatrixTest /*unused*/, FixedValuesTest_Um1 /*unused*/)
{

    std::complex<float> Um1 =
        -std::sin(theta12) * std::cos(theta23) -
        std::cos(theta12) * std::sin(theta23) * std::sin(theta13) * std::exp(std::complex<float>(0.0, 1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0, 1, 0}), Um1);
}

TEST_F(PMNSmatrixTest /*unused*/, FixedValuesTest_Um2 /*unused*/)
{

    std::complex<float> Um2 =
        std::cos(theta12) * std::cos(theta23) -
        std::sin(theta12) * std::sin(theta23) * std::sin(theta13) * std::exp(std::complex<float>(0.0, 1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0, 1, 1}), Um2);
}

TEST_F(PMNSmatrixTest /*unused*/, FixedValuesTest_Um3 /*unused*/)
{

    ASSERT_EQ(matrixTensor.getValue<float>({0, 1, 2}), std::sin(theta23) * std::cos(theta13));
}

TEST_F(PMNSmatrixTest /*unused*/, FixedValuesTest_Ut1 /*unused*/)
{

    std::complex<float> Ut1 =
        std::sin(theta12) * std::sin(theta23) -
        std::cos(theta12) * std::cos(theta23) * std::sin(theta13) * std::exp(std::complex<float>(0.0, 1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0, 2, 0}), Ut1);
}

TEST_F(PMNSmatrixTest /*unused*/, FixedValuesTest_Ut2 /*unused*/)
{

    std::complex<float> Ut2 =
        -std::cos(theta12) * std::sin(theta23) -
        std::sin(theta12) * std::cos(theta23) * std::sin(theta13) * std::exp(std::complex<float>(0.0, 1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0, 2, 1}), Ut2);
}

TEST_F(PMNSmatrixTest /*unused*/, FixedValuesTest_Ut3 /*unused*/)
{

    ASSERT_EQ(matrixTensor.getValue<float>({0, 2, 2}), std::cos(theta23) * std::cos(theta13));
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)