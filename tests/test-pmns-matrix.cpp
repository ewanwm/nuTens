#include <gtest/gtest.h>
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <iostream>
#include <nuTens/propagator/pmns-matrix.hpp>

#include <gtest/gtest.h>

using namespace nuTens;

class PMNSmatrixTest :public gtest::TestWithParam<float> {

  protected:

    float theta12;
    float theta23;
    float theta13;
    float deltaCP;

    PMNSmatrix matrix;
    
    Tensor matrixTensor;

    // set up common values to use across tests
    void SetUp() {

        theta12 = 1.2  * M_PI;
        theta23 = 2.3  * M_PI;
        theta13 = 1.3  * M_PI;
        deltaCP = 0.5  * M_PI;

        matrix.setParameterValues(theta12, theta13, theta23, deltaCP);

        matrixTensor = matrix.build();

    }
};

TEST_F(PMNSmatrixTest, FixedValuesTest_Ue1) {

    ASSERT_EQ(matrixTensor.getValue<float>({0,0,0}), std::cos(theta12) * std::cos(theta13));

}

TEST_F(PMNSmatrixTest, FixedValuesTest_Ue2) {

    ASSERT_EQ(matrixTensor.getValue<float>({0,0,1}), std::sin(theta12) * std::cos(theta13));

}

TEST_F(PMNSmatrixTest, FixedValuesTest_Ue3) {

    std::complex<float> Ue3 = std::sin(theta13) * std::exp(std::complex<float>(0.0, -1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0,0,2}), Ue3);

}

TEST_F(PMNSmatrixTest, FixedValuesTest_Um1) {

    std::complex<float> Um1 = -std::sin(theta12) * std::cos(theta23) - std::cos(theta12) * std::sin(theta23) * std::sin(theta13) * std::exp(std::complex<float>(0.0, 1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0,1,0}), Um1);

}

TEST_F(PMNSmatrixTest, FixedValuesTest_Um2) {

    std::complex<float> Um2 = std::cos(theta12) * std::cos(theta23) - std::sin(theta12) * std::sin(theta23) * std::sin(theta13) * std::exp(std::complex<float>(0.0, 1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0,1,1}), Um2);

}

TEST_F(PMNSmatrixTest, FixedValuesTest_Um3) {

    ASSERT_EQ(matrixTensor.getValue<float>({0,1,2}), std::sin(theta23) * std::cos(theta13));

}

TEST_F(PMNSmatrixTest, FixedValuesTest_Ut1) {

    std::complex<float> Ut1 = std::sin(theta12) * std::sin(theta23) - std::cos(theta12) * std::cos(theta23) * std::sin(theta13) * std::exp(std::complex<float>(0.0, 1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0,2,0}), Ut1);

}

TEST_F(PMNSmatrixTest, FixedValuesTest_Ut2) {

    std::complex<float> Ut2 = -std::cos(theta12) * std::sin(theta23) - std::sin(theta12) * std::cos(theta23) * std::sin(theta13) * std::exp(std::complex<float>(0.0, 1.0) * deltaCP);
    ASSERT_EQ(matrixTensor.getValue<std::complex<float>>({0,2,1}), Ut2);

}

TEST_F(PMNSmatrixTest, FixedValuesTest_Ut3) {

    ASSERT_EQ(matrixTensor.getValue<float>({0,2,2}), std::cos(theta23) * std::cos(theta13));

}
