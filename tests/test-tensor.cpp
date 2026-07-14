
#include <tests/test-tensor.hpp>

/*
    Do some very basic tests of tensor functionality
    e.g. test that complex matrices work as expected, 1+1 == 2 etc.
*/

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
using namespace nuTens;

// check creation of tensors
TEST(Tensor /*unused*/, TensorCreationFloatCPU /*unused*/)
{
    testTensorCreation<float>(dtypes::kFloat, dtypes::kCPU);
}

TEST(Tensor /*unused*/, TensorCreationDoubleCPU /*unused*/)
{
    testTensorCreation<double>(dtypes::kDouble, dtypes::kCPU);
}

TEST(Tensor /*unused*/, TensorCreationComplexFloatCPU /*unused*/)
{
    testTensorCreation<std::complex<float>>(dtypes::kComplexFloat, dtypes::kCPU);
}

TEST(Tensor /*unused*/, TensorCreationComplexDoubleCPU /*unused*/)
{
    testTensorCreation<std::complex<double>>(dtypes::kComplexDouble, dtypes::kCPU);
}

TEST(Tensor /*unused*/, ComplexTensorCreationComplexFloatCPU /*unused*/)
{
    testComplexTensorCreation<std::complex<float>>(dtypes::kComplexFloat, dtypes::kCPU);
}

// check equality operators
TEST(Tensor /*unused*/, EqualityOperators /*unused*/)
{
    testEqualityOperators(dtypes::kFloat, dtypes::kCPU);
}

// test manipulation of elements of tensor
TEST(Tensor /*unused*/, ElementMapipulationFloat /*unused*/)
{
    testElementManipulation<float>(dtypes::kFloat, dtypes::kCPU);
}
TEST(Tensor /*unused*/, ElementMapipulationDouble /*unused*/)
{
    testElementManipulation<double>(dtypes::kDouble, dtypes::kCPU);
}

// check some basic arithmetic
TEST(Tensor /*unused*/, simpleArithmeticFloatCPU /*unused*/)
{
    testArithmeticFloatType<float>(dtypes::kFloat, dtypes::kCPU);
}

TEST(Tensor /*unused*/, simpleArithmeticDoubleCPU /*unused*/)
{
    testArithmeticFloatType<double>(dtypes::kDouble, dtypes::kCPU);
}

// check some basic arithmetic
TEST(Tensor /*unused*/, simpleArithmeticComplexFloatCPU /*unused*/)
{
    testArithmeticComplexType<float>(dtypes::kComplexFloat, dtypes::kCPU);
}

// check some basic arithmetic
TEST(Tensor /*unused*/, simpleArithmeticComplexDoubleCPU /*unused*/)
{
    testArithmeticComplexType<double>(dtypes::kComplexDouble, dtypes::kCPU);
}

TEST(Tensor /*unused*/, SummationFloatCPU /*unused*/)
{
    testSummation<float>(dtypes::kFloat, dtypes::kCPU);
}

// check standard functions of real tensors
TEST(Tensor /*unused*/, StandardFunctionsFloatCPU /*unused*/)
{
    testStandardFunctions<float>(dtypes::kFloat, dtypes::kCPU);
}

// test matrix operations for real tensor
TEST(Tensor /*unused*/, MatrixFloatCPU /*unused*/)
{
    testMatrixOperations<float>(dtypes::kFloat, dtypes::kCPU);
}

TEST(Tensor /*unused*/, eigFloatCPU /*unused*/)
{
    testEig<float>(dtypes::kFloat, dtypes::kCPU);
}

TEST(Tensor /*unused*/, eighFloatCPU /*unused*/)
{
    testEigh<float>(dtypes::kFloat, dtypes::kCPU);
}

TEST(Tensor /*unused*/, eigvalsFloatCPU /*unused*/)
{
    testEigVals<float>(dtypes::kFloat, dtypes::kCPU);
}

TEST(Tensor /*unused*/, eigvalshFloatCPU /*unused*/)
{
    testEigValsh<float>(dtypes::kFloat, dtypes::kCPU);
}

TEST(Tensor /*unused*/, AccessedTensor1D /*unused*/)
{

    auto tensor = AccessedTensor<float, 1, dtypes::kCPU>::zeros({3}, false);

    tensor.setValue({1}, 1.0F);

    ASSERT_EQ(tensor.getValue(1), 1.0F);
    ASSERT_EQ(tensor.getValue<float>({1}), 1.0F);
}

TEST(Tensor /*unused*/, AccessedTensor2D /*unused*/)
{

    auto tensor = AccessedTensor<float, 2, dtypes::kCPU>::zeros({3, 3}, false);

    tensor.setValue({1, 1}, 2.0);
    ASSERT_EQ(tensor.getValue(1, 1), 2.0F);
    ASSERT_EQ(tensor.getValue<float>({1, 1}), 2.0F);
}

TEST(Tensor /*unused*/, AccessedTensor3D /*unused*/)
{

    auto tensor = AccessedTensor<float, 3, dtypes::kCPU>::zeros({3, 3, 3}, false);

    tensor.setValue({1, 1, 1}, 3.0);
    ASSERT_EQ(tensor.getValue(1, 1, 1), 3.0F);
    ASSERT_EQ(tensor.getValue<float>({1, 1, 1}), 3.0F);
}

TEST(Tensor /*unused*/, AccessedTensorDimCheck /*unused*/)
{

    auto tensor1D = AccessedTensor<float, 1, dtypes::kCPU, true>::zeros({3}, false);
    auto tensor2D = AccessedTensor<float, 2, dtypes::kCPU, true>::zeros({3, 3}, false);
    auto tensor3D = AccessedTensor<float, 3, dtypes::kCPU, true>::zeros({3, 3, 3}, false);

    // check it works with correct num of dimensions
    tensor1D.setValue({1}, 1.23);
    ASSERT_EQ(tensor1D.getValue(1), 1.23F);

    // check it doesn't work with wrong number of dimensions
    EXPECT_THROW(tensor1D.setValue({1, 1}, 1.23), std::invalid_argument);
    EXPECT_THROW(tensor2D.setValue({1, 1, 1}, 1.23), std::invalid_argument);
    EXPECT_THROW(tensor3D.setValue({1, 1, 1, 1}, 1.23), std::invalid_argument);
}

// Test arithmetic overrides
TEST(Tensor /*unused*/, addOverride /*unused*/)
{

    Tensor one = Tensor::ones({2}, dtypes::kFloat, dtypes::kCPU, false);

    ASSERT_EQ((one + one), Tensor::add(one, one));
}

TEST(Tensor /*unused*/, multiplyOverride /*unused*/)
{

    Tensor two = Tensor({2, 2}, dtypes::kFloat, dtypes::kCPU, false);

    ASSERT_EQ((two * two), Tensor::mul(two, two));
}

TEST(Tensor /*unused*/, scaleOverride /*unused*/)
{

    Tensor two = Tensor({2, 2}, dtypes::kFloat, dtypes::kCPU, false);

    ASSERT_EQ((two * 2.0), Tensor::scale(two, 2.0));
}

TEST(Tensor /*unused*/, divOverride /*unused*/)
{

    Tensor one = Tensor::ones({2}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor two = Tensor({2, 2}, dtypes::kFloat, dtypes::kCPU, false);

    ASSERT_EQ((one / two), Tensor::div(one, two));
}

TEST(Tensor /*unused*/, batchDim /*unused*/)
{

    Tensor tensor = Tensor::ones({3, 3, 3, 3}, dtypes::kFloat, dtypes::kCPU, false);

    ASSERT_TRUE(!tensor.getHasBatchDim());

    tensor.addBatchDim();

    ASSERT_TRUE(tensor.getHasBatchDim());
    ASSERT_EQ(tensor.getNdim(), 5);

    // doing again should have no effect
    tensor.addBatchDim();

    ASSERT_TRUE(tensor.getHasBatchDim());
    ASSERT_EQ(tensor.getNdim(), 5);
}

// test NoGrad guard
TEST(Tensor /*unused*/, testNoGrad /*unused*/)
{
    testNoGrad<float>(dtypes::kFloat, dtypes::kCPU);
}
// test basic derivatives
TEST(Tensor /*unused*/, testDerivativesBasicScalarFloat /*unused*/)
{
    testDerivativesBasicScalarReal<float>(dtypes::kFloat, dtypes::kCPU);
}
TEST(Tensor /*unused*/, testDerivativesBasicScalarDouble /*unused*/)
{
    testDerivativesBasicScalarReal<double>(dtypes::kDouble, dtypes::kCPU);
}
TEST(Tensor /*unused*/, testDerivativesBasicScalarComplexFloat /*unused*/)
{
    testDerivativesBasicScalarComplex<float>(dtypes::kComplexFloat, dtypes::kCPU);
}
TEST(Tensor /*unused*/, testDerivativesBasicScalarComplexDouble /*unused*/)
{
    testDerivativesBasicScalarComplex<double>(dtypes::kComplexDouble, dtypes::kCPU);
}

// test basic tensor derivatives
TEST(Tensor /*unused*/, testDerivativesBasicTensorFloat /*unused*/)
{
    testDerivativesBasicTensorReal<float>(dtypes::kFloat, dtypes::kCPU);
}
TEST(Tensor /*unused*/, testDerivativesBasicTensorDouble /*unused*/)
{
    testDerivativesBasicTensorReal<double>(dtypes::kDouble, dtypes::kCPU);
}
TEST(Tensor /*unused*/, testDerivativesBasicTensorComplexFloat /*unused*/)
{
    testDerivativesBasicTensorComplex<float>(dtypes::kComplexFloat, dtypes::kCPU);
}
TEST(Tensor /*unused*/, testDerivativesBasicTensorComplexDouble /*unused*/)
{
    testDerivativesBasicTensorComplex<double>(dtypes::kComplexDouble, dtypes::kCPU);
}

// test derivatives of some standard functions
TEST(Tensor /*unused*/, testDerivativesStandardFunctionsFloat /*unused*/)
{
    testDerivativesStandardFunctions(dtypes::kFloat, dtypes::kCPU);
}
TEST(Tensor /*unused*/, testDerivativesStandardFunctionsTensorDouble /*unused*/)
{
    testDerivativesStandardFunctions(dtypes::kDouble, dtypes::kCPU);
}

TEST(Tensor /*unused*/, GetVariantValue /*unused*/)
{

    Tensor floatTensor = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU);
    auto variantFloat = floatTensor.getVariantValue({0});
    ASSERT_TRUE(std::holds_alternative<float>(variantFloat));

    Tensor doubleTensor = Tensor::ones({1}, dtypes::kDouble, dtypes::kCPU);
    auto variantDouble = doubleTensor.getVariantValue({0});
    ASSERT_TRUE(std::holds_alternative<double>(variantDouble));

    Tensor complexFloatTensor = Tensor::ones({1}, dtypes::kComplexFloat, dtypes::kCPU);
    auto variantComplexFloat = complexFloatTensor.getVariantValue({0});
    ASSERT_TRUE(std::holds_alternative<std::complex<float>>(variantComplexFloat));

    Tensor complexDoubleTensor = Tensor::ones({1}, dtypes::kComplexDouble, dtypes::kCPU);
    auto variantComplexDouble = complexDoubleTensor.getVariantValue({0});
    ASSERT_TRUE(std::holds_alternative<std::complex<double>>(variantComplexDouble));
}

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)