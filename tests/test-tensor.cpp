
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

    // test the dedicated complex tensor builder function
    Tensor complex =
        Tensor::TensorComplex({std::complex<float>(1.234, 5.678)}, dtypes::kComplexFloat, dtypes::kCPU, false);

    ASSERT_EQ(complex.getValue<std::complex<float>>(), std::complex<float>(1.234, 5.678));
}

TEST(Tensor /*unused*/, TensorCreationComplexDoubleCPU /*unused*/)
{
    testTensorCreation<std::complex<double>>(dtypes::kComplexDouble, dtypes::kCPU);
}

// check equality operators
TEST(Tensor /*unused*/, EqualityOperators /*unused*/)
{

    Tensor one = Tensor({1.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor two = Tensor({2.0}, dtypes::kFloat, dtypes::kCPU, false);

    ASSERT_TRUE(one == one);
    ASSERT_TRUE(one != two);
}

// test manipulation of elements of tensor
TEST(Tensor /*unused*/, ElementMapipulation /*unused*/)
{

    auto tensorFloat = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);

    tensorFloat.setValue({0, 0}, 0.0);
    tensorFloat.setValue({0, 1}, 1.0);

    tensorFloat.setValue({1, 0}, 2.0);
    tensorFloat.setValue({1, 1}, 3.0);

    std::cout << "Test matrix: \n" << tensorFloat << std::endl;

    // test slicing
    Tensor slice = tensorFloat.getValues({1, "..."});
    ASSERT_EQ(slice.getValue<float>({0}), 2.0);
    ASSERT_EQ(slice.getValue<float>({1}), 3.0);

    tensorFloat.dType(dtypes::kDouble);
    ASSERT_EQ(tensorFloat.getValue<double>({0, 0}), 0.0);
    ASSERT_EQ(tensorFloat.getValue<double>({0, 1}), 1.0);
}

// check some basic arithmetic
TEST(Tensor /*unused*/, simpleArithmeticFloat /*unused*/)
{
    testArithmeticFloatType<float>(dtypes::kFloat);
}

TEST(Tensor /*unused*/, simpleArithmeticDouble /*unused*/)
{
    testArithmeticFloatType<double>(dtypes::kDouble);
}

// check some basic arithmetic
TEST(Tensor /*unused*/, simpleArithmeticComplexFloat /*unused*/)
{
    testArithmeticComplexType<float>(dtypes::kComplexFloat);
}

// check some basic arithmetic
TEST(Tensor /*unused*/, simpleArithmeticComplexDouble /*unused*/)
{
    testArithmeticComplexType<double>(dtypes::kComplexDouble);
}

TEST(Tensor /*unused*/, Summation /*unused*/)
{

    Tensor tensor = Tensor::ones({3, 3}, dtypes::kFloat, dtypes::kCPU, false);

    ASSERT_EQ(tensor.sum().getValue<float>(), 9.0);

    Tensor sum = tensor.sum({1});

    ASSERT_EQ(sum.getValue<float>({0}), 3.0);
    ASSERT_EQ(sum.getValue<float>({1}), 3.0);
    ASSERT_EQ(sum.getValue<float>({2}), 3.0);

    Tensor cumsum = tensor.cumsum(1);

    ASSERT_EQ(cumsum.getValue<float>({0, 0}), 1.0);
    ASSERT_EQ(cumsum.getValue<float>({0, 1}), 2.0);
    ASSERT_EQ(cumsum.getValue<float>({0, 2}), 3.0);
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

// check standard functions of real tensors
TEST(Tensor /*unused*/, StandardFunctionsFloat /*unused*/)
{

    float theta = 1.234;
    Tensor thetaTensor = Tensor({theta}, dtypes::kComplexFloat, dtypes::kCPU, false);

    ASSERT_EQ(Tensor::sin(thetaTensor).getValue<float>(), std::sin(theta));
    ASSERT_EQ(Tensor::cos(thetaTensor).getValue<float>(), std::cos(theta));
    ASSERT_EQ(Tensor::exp(thetaTensor).getValue<float>(), std::exp(theta));
}

// check inplace functions
TEST(Tensor /*unused*/, InPlaceMatmul /*unused*/)
{

    // test matrix multiplication
    Tensor tensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    tensor.setValue({0, 0}, 0.0);
    tensor.setValue({0, 1}, 1.0);

    tensor.setValue({1, 0}, 2.0);
    tensor.setValue({1, 1}, 3.0);

    Tensor otherTensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    otherTensor.setValue({0, 0}, 0.0);
    otherTensor.setValue({0, 1}, 1.0);

    otherTensor.setValue({1, 0}, 2.0);
    otherTensor.setValue({1, 1}, 3.0);

    tensor.matmul_(otherTensor);

    // test matrix multiplication
    ASSERT_EQ(tensor.getValue<float>({0, 0}), 2.0);
    ASSERT_EQ(tensor.getValue<float>({0, 1}), 3.0);
    ASSERT_EQ(tensor.getValue<float>({1, 0}), 6.0);
    ASSERT_EQ(tensor.getValue<float>({1, 1}), 11.0);
}

TEST(Tensor /*unused*/, InPlaceMul /*unused*/)
{

    // test matrix multiplication
    Tensor tensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    tensor.setValue({0, 0}, 0.0);
    tensor.setValue({0, 1}, 1.0);

    tensor.setValue({1, 0}, 2.0);
    tensor.setValue({1, 1}, 3.0);

    Tensor otherTensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    otherTensor.setValue({0, 0}, 0.0);
    otherTensor.setValue({0, 1}, 1.0);

    otherTensor.setValue({1, 0}, 2.0);
    otherTensor.setValue({1, 1}, 3.0);

    tensor.mul_(otherTensor);

    // test matrix multiplication
    ASSERT_EQ(tensor.getValue<float>({0, 0}), 0.0);
    ASSERT_EQ(tensor.getValue<float>({0, 1}), 1.0);
    ASSERT_EQ(tensor.getValue<float>({1, 0}), 4.0);
    ASSERT_EQ(tensor.getValue<float>({1, 1}), 9.0);
}

TEST(Tensor /*unused*/, InPlaceDiv /*unused*/)
{

    // test matrix multiplication
    Tensor tensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    tensor.setValue({0, 0}, 1.0);
    tensor.setValue({0, 1}, 2.0);

    tensor.setValue({1, 0}, 3.0);
    tensor.setValue({1, 1}, 4.0);

    Tensor otherTensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    otherTensor.setValue({0, 0}, 1.0);
    otherTensor.setValue({0, 1}, 2.0);

    otherTensor.setValue({1, 0}, 3.0);
    otherTensor.setValue({1, 1}, 4.0);

    tensor.div_(otherTensor);

    // test matrix multiplication
    ASSERT_EQ(tensor.getValue<float>({0, 0}), 1.0);
    ASSERT_EQ(tensor.getValue<float>({0, 1}), 1.0);
    ASSERT_EQ(tensor.getValue<float>({1, 0}), 1.0);
    ASSERT_EQ(tensor.getValue<float>({1, 1}), 1.0);
}

TEST(Tensor /*unused*/, InPlaceScale /*unused*/)
{

    // test matrix multiplication
    Tensor tensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    tensor.setValue({0, 0}, 1.0);
    tensor.setValue({0, 1}, 2.0);

    tensor.setValue({1, 0}, 3.0);
    tensor.setValue({1, 1}, 4.0);

    tensor.scale_(2.0);

    // test matrix multiplication
    ASSERT_EQ(tensor.getValue<float>({0, 0}), 2.0);
    ASSERT_EQ(tensor.getValue<float>({0, 1}), 4.0);
    ASSERT_EQ(tensor.getValue<float>({1, 0}), 6.0);
    ASSERT_EQ(tensor.getValue<float>({1, 1}), 8.0);
}

TEST(Tensor /*unused*/, InPlaceScaleComplex /*unused*/)
{

    // test matrix multiplication
    Tensor tensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    tensor.setValue({0, 0}, 1.0);
    tensor.setValue({0, 1}, 2.0);

    tensor.setValue({1, 0}, 3.0);
    tensor.setValue({1, 1}, 4.0);

    tensor.scale_(std::complex<float>(2.0, 2.0));

    // test matrix multiplication
    ASSERT_EQ(tensor.getValue<std::complex<float>>({0, 0}), std::complex<float>(2.0, 2.0));
    ASSERT_EQ(tensor.getValue<std::complex<float>>({0, 1}), std::complex<float>(4.0, 4.0));
    ASSERT_EQ(tensor.getValue<std::complex<float>>({1, 0}), std::complex<float>(6.0, 6.0));
    ASSERT_EQ(tensor.getValue<std::complex<float>>({1, 1}), std::complex<float>(8.0, 8.0));
}

TEST(Tensor /*unused*/, InPlacePow /*unused*/)
{

    // test matrix multiplication
    Tensor tensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    tensor.setValue({0, 0}, 1.0);
    tensor.setValue({0, 1}, 2.0);

    tensor.setValue({1, 0}, 3.0);
    tensor.setValue({1, 1}, 4.0);

    tensor.pow_(2.0);

    // test matrix multiplication
    ASSERT_EQ(tensor.getValue<float>({0, 0}), 1.0);
    ASSERT_EQ(tensor.getValue<float>({0, 1}), 4.0);
    ASSERT_EQ(tensor.getValue<float>({1, 0}), 9.0);
    ASSERT_EQ(tensor.getValue<float>({1, 1}), 16.0);
}

TEST(Tensor /*unused*/, InPlaceComplexPow /*unused*/)
{

    // proof of eulers identity
    Tensor euler = Tensor({static_cast<float>(std::exp(1.0))}, dtypes::kComplexFloat, dtypes::kCPU, false);
    euler.pow_(std::complex<float>(0.0, M_PI));

    std::complex<float> testVal = euler.getValue<std::complex<float>>();
    ASSERT_NEAR(testVal.real(), -1.0, 1e-6);
    ASSERT_NEAR(testVal.imag(), 0.0, 1e-6);
}

TEST(Tensor /*unused*/, InPlaceExp /*unused*/)
{

    // test matrix multiplication
    Tensor tensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    tensor.setValue({0, 0}, 1.0);
    tensor.setValue({0, 1}, 2.0);

    tensor.setValue({1, 0}, 3.0);
    tensor.setValue({1, 1}, 4.0);

    tensor.exp_();

    // test matrix multiplication
    ASSERT_NEAR(tensor.getValue<float>({0, 0}), std::exp(1.0), 1e-4);
    ASSERT_NEAR(tensor.getValue<float>({0, 1}), std::exp(2.0), 1e-4);
    ASSERT_NEAR(tensor.getValue<float>({1, 0}), std::exp(3.0), 1e-4);
    ASSERT_NEAR(tensor.getValue<float>({1, 1}), std::exp(4.0), 1e-4);
}

TEST(Tensor /*unused*/, InPlaceTranspose /*unused*/)
{

    // test matrix multiplication
    Tensor tensor = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    tensor.setValue({0, 0}, 1.0);
    tensor.setValue({0, 1}, 2.0);

    tensor.setValue({1, 0}, 3.0);
    tensor.setValue({1, 1}, 4.0);

    tensor.transpose_(0, 1);

    // test matrix multiplication
    ASSERT_EQ(tensor.getValue<float>({0, 0}), 1.0);
    ASSERT_EQ(tensor.getValue<float>({0, 1}), 3.0);
    ASSERT_EQ(tensor.getValue<float>({1, 0}), 2.0);
    ASSERT_EQ(tensor.getValue<float>({1, 1}), 4.0);
}

// test matrix operations for real tensor
TEST(Tensor /*unused*/, MatrixFloat /*unused*/)
{

    auto tensorFloat = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    auto eye = Tensor::eye(2, dtypes::kFloat, dtypes::kCPU, false);

    tensorFloat.setValue({0, 0}, 0.0);
    tensorFloat.setValue({0, 1}, 1.0);

    tensorFloat.setValue({1, 0}, 2.0);
    tensorFloat.setValue({1, 1}, 3.0);

    std::cout << "Test matrix: \n" << tensorFloat << std::endl;

    // test matrix multiplication
    Tensor squared = Tensor::matmul(tensorFloat, tensorFloat);
    ASSERT_EQ(squared.getValue<float>({0, 0}), 2.0);
    ASSERT_EQ(squared.getValue<float>({0, 1}), 3.0);
    ASSERT_EQ(squared.getValue<float>({1, 0}), 6.0);
    ASSERT_EQ(squared.getValue<float>({1, 1}), 11.0);

    // test multiplication by identity matrix
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<float>({0, 0}), 0.0);
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<float>({0, 1}), 1.0);
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<float>({1, 0}), 2.0);
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<float>({1, 1}), 3.0);

    // test matrix addition
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<float>({0, 0}), 0.0);
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<float>({0, 1}), 2.0);
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<float>({1, 0}), 4.0);
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<float>({1, 1}), 6.0);

    // test transpose
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0, 1)).getValue<float>({0, 0}), 0.0);
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0, 1)).getValue<float>({1, 0}), 1.0);
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0, 1)).getValue<float>({0, 1}), 2.0);
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0, 1)).getValue<float>({1, 1}), 3.0);

    // test outer product of two vectors
    Tensor vec1 = Tensor::zeros({2}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor vec2 = Tensor::zeros({2}, dtypes::kFloat, dtypes::kCPU, false);

    vec1.setValue({0}, 1.0);
    vec1.setValue({1}, 2.0);
    vec2.setValue({0}, 3.0);
    vec2.setValue({1}, 4.0);

    Tensor outer = Tensor::outer(vec1, vec2);

    ASSERT_EQ(outer.getValue<float>({0, 0}), 3.0);
    ASSERT_EQ(outer.getValue<float>({0, 1}), 4.0);
    ASSERT_EQ(outer.getValue<float>({1, 0}), 6.0);
    ASSERT_EQ(outer.getValue<float>({1, 1}), 8.0);
}

// get eigenvalues of matrix
// ------
// | 2 1 |
// | 1 2 |
// ------
// which are 1 and 3
// with eigenvectors
// v_1 = [1, -1]
// v_3 = [1, 1 ]

TEST(Tensor /*unused*/, eig /*unused*/)
{

    Tensor evals = Tensor::zeros({2}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor evecs = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor mat = Tensor::ones({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    mat.setValue({0, 0}, 2.0);
    mat.setValue({0, 1}, 1.0);
    mat.setValue({1, 0}, 1.0);
    mat.setValue({1, 1}, 2.0);

    Tensor::eig(mat, evals, evecs);

    ASSERT_EQ(evals.getValue<float>({0}), 3.0);
    ASSERT_EQ(evals.getValue<float>({1}), 1.0);

    ASSERT_EQ(evecs.getValue<float>({0, 0}), evecs.getValue<float>({1, 0}));
    ASSERT_EQ(evecs.getValue<float>({0, 1}), -evecs.getValue<float>({1, 1}));
}

TEST(Tensor /*unused*/, eigh /*unused*/)
{

    Tensor evals = Tensor::zeros({2}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor evecs = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor mat = Tensor::ones({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    mat.setValue({0, 0}, 2.0);
    mat.setValue({0, 1}, 1.0);
    mat.setValue({1, 0}, 1.0);
    mat.setValue({1, 1}, 2.0);

    Tensor::eigh(mat, evals, evecs);

    ASSERT_EQ(evals.getValue<float>({0}), 1.0);
    ASSERT_EQ(evals.getValue<float>({1}), 3.0);

    ASSERT_EQ(evecs.getValue<float>({0, 0}), -evecs.getValue<float>({1, 0}));
    ASSERT_EQ(evecs.getValue<float>({0, 1}), evecs.getValue<float>({1, 1}));
}

TEST(Tensor /*unused*/, eigvals /*unused*/)
{

    Tensor evals = Tensor::zeros({2}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor mat = Tensor::ones({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    mat.setValue({0, 0}, 2.0);
    mat.setValue({0, 1}, 1.0);
    mat.setValue({1, 0}, 1.0);
    mat.setValue({1, 1}, 2.0);

    Tensor::eigvals(mat, evals);

    ASSERT_EQ(evals.getValue<float>({0}), 3.0);
    ASSERT_EQ(evals.getValue<float>({1}), 1.0);
}

TEST(Tensor /*unused*/, eigvalsh /*unused*/)
{

    Tensor evals = Tensor::zeros({2}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor mat = Tensor::ones({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    mat.setValue({0, 0}, 2.0);
    mat.setValue({0, 1}, 1.0);
    mat.setValue({1, 0}, 1.0);
    mat.setValue({1, 1}, 2.0);

    Tensor::eigvalsh(mat, evals);

    ASSERT_EQ(evals.getValue<float>({0}), 1.0);
    ASSERT_EQ(evals.getValue<float>({1}), 3.0);
}

TEST(Tensor /*unused*/, AccessedTensor1D /*unused*/)
{

    auto tensor = AccessedTensor<float, 1, dtypes::kCPU>::zeros({3}, false);

    tensor.setValue(1.0, 1);

    ASSERT_EQ(tensor.getValue(1), 1.0);
}

TEST(Tensor /*unused*/, AccessedTensor2D /*unused*/)
{

    auto tensor = AccessedTensor<float, 2, dtypes::kCPU>::zeros({3, 3}, false);

    tensor.setValue(2.0, 1, 1);

    ASSERT_EQ(tensor.getValue(1, 1), 2.0);
}

TEST(Tensor /*unused*/, AccessedTensor3D /*unused*/)
{

    auto tensor = AccessedTensor<float, 3, dtypes::kCPU>::zeros({3, 3, 3}, false);

    tensor.setValue(3.0, 1, 1, 1);

    ASSERT_EQ(tensor.getValue(1, 1, 1), 3.0);
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

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)