#include <tests/test-pmns-matrix.hpp>

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
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
    void SetUp() override
    {

        matrix.setTheta12(theta12).setTheta13(theta13).setTheta23(theta23).setDeltaCP(deltaCP);

        matrixTensor = matrix.build();
    }
};

TEST_F(PMNSmatrixTest /*unused*/, testParameterSetting /*unused*/)
{
    ASSERT_EQ(theta12, matrix.getTheta12Tensor().getValue<float>());
    ASSERT_EQ(theta13, matrix.getTheta13Tensor().getValue<float>());
    ASSERT_EQ(theta23, matrix.getTheta23Tensor().getValue<float>());
    ASSERT_EQ(deltaCP, matrix.getDeltaCPTensor().getValue<float>());
}

TEST_F(PMNSmatrixTest /*unused*/, testGPU /*unused*/)
{

    // skip this test if there is no GPU available
    SKIP_GPU(dtypes::kGPU);

    PMNSmatrix matrixGPU =
        PMNSmatrix(dtypes::kGPU).setTheta12(theta12).setTheta13(theta13).setTheta23(theta23).setDeltaCP(deltaCP);

    Tensor matrixTensorGPU = matrixGPU.build();

    ASSERT_NEAR(matrixTensorGPU.real().getValue<double>({0, 0, 0}), matrixTensor.real().getValue<double>({0, 0, 0}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.real().getValue<double>({0, 0, 1}), matrixTensor.real().getValue<double>({0, 0, 1}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.real().getValue<double>({0, 0, 2}), matrixTensor.real().getValue<double>({0, 0, 2}),
                1e-5);

    ASSERT_NEAR(matrixTensorGPU.real().getValue<double>({0, 1, 0}), matrixTensor.real().getValue<double>({0, 1, 0}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.real().getValue<double>({0, 1, 1}), matrixTensor.real().getValue<double>({0, 1, 1}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.real().getValue<double>({0, 1, 2}), matrixTensor.real().getValue<double>({0, 1, 2}),
                1e-5);

    ASSERT_NEAR(matrixTensorGPU.real().getValue<double>({0, 2, 0}), matrixTensor.real().getValue<double>({0, 2, 0}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.real().getValue<double>({0, 2, 1}), matrixTensor.real().getValue<double>({0, 2, 1}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.real().getValue<double>({0, 2, 2}), matrixTensor.real().getValue<double>({0, 2, 2}),
                1e-5);

    ASSERT_NEAR(matrixTensorGPU.imag().getValue<double>({0, 0, 0}), matrixTensor.imag().getValue<double>({0, 0, 0}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.imag().getValue<double>({0, 0, 1}), matrixTensor.imag().getValue<double>({0, 0, 1}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.imag().getValue<double>({0, 0, 2}), matrixTensor.imag().getValue<double>({0, 0, 2}),
                1e-5);

    ASSERT_NEAR(matrixTensorGPU.imag().getValue<double>({0, 1, 0}), matrixTensor.imag().getValue<double>({0, 1, 0}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.imag().getValue<double>({0, 1, 1}), matrixTensor.imag().getValue<double>({0, 1, 1}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.imag().getValue<double>({0, 1, 2}), matrixTensor.imag().getValue<double>({0, 1, 2}),
                1e-5);

    ASSERT_NEAR(matrixTensorGPU.imag().getValue<double>({0, 2, 0}), matrixTensor.imag().getValue<double>({0, 2, 0}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.imag().getValue<double>({0, 2, 1}), matrixTensor.imag().getValue<double>({0, 2, 1}),
                1e-5);
    ASSERT_NEAR(matrixTensorGPU.imag().getValue<double>({0, 2, 2}), matrixTensor.imag().getValue<double>({0, 2, 2}),
                1e-5);
}

TEST_F(PMNSmatrixTest /*unused*/, CachingSameResultTest /*unused*/)
{

    PMNSmatrix cacheMatrix;
    cacheMatrix.setTheta12(theta12).setTheta13(theta13).setTheta23(theta23).setDeltaCP(deltaCP);

    // copy resulting tensor
    Tensor cachedMatrixTensor = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat, dtypes::kCPU, false);
    cachedMatrixTensor.setValue({"..."}, cacheMatrix.build());

    // make sure we get the same result again without changing parameter values
    Tensor newMatrix = cacheMatrix.build();
    ASSERT_EQ(newMatrix, cachedMatrixTensor);
}

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

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)