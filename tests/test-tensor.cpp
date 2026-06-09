
#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>

#include <complex>
#include <gtest/gtest.h>

/*
    Do some very basic tests of tensor functionality
    e.g. test that complex matrices work as expected, 1+1 == 2 etc.
*/

using namespace nuTens;

// check creation of tensors
TEST(Tensor, TensorCreationFloat)
{

    Tensor zero = Tensor::zeros({1}, dtypes::kFloat, dtypes::kCPU, false);
    std::cout << "zero tensor: " << zero << std::endl;
    ASSERT_EQ(zero.getValue<float>(), 0.0);

    Tensor one = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU, false);
    std::cout << "one tensor: " << one << std::endl;
    ASSERT_EQ(one.getValue<float>(), 1.0);

    Tensor three = Tensor({3.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(three.getValue<float>(), 3.0);

    Tensor rand = Tensor::rand({1}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_LE(rand.getValue<float>(), 1.0);
    ASSERT_GE(rand.getValue<float>(), 0.0);

    Tensor diagonal = Tensor({0.0, 1.0, 2.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor diagTensor = Tensor::diag(diagonal);
    std::cout << "diagonal tensor: \n" << diagTensor << std::endl;
    ASSERT_EQ(diagTensor.getValue<float>({0, 0}), 0.0);
    ASSERT_EQ(diagTensor.getValue<float>({1, 1}), 1.0);
    ASSERT_EQ(diagTensor.getValue<float>({2, 2}), 2.0);

    ASSERT_EQ(diagTensor.getValue<float>({0, 1}), 0.0);
    ASSERT_EQ(diagTensor.getValue<float>({0, 2}), 0.0);
    ASSERT_EQ(diagTensor.getValue<float>({1, 0}), 0.0);
    ASSERT_EQ(diagTensor.getValue<float>({1, 2}), 0.0);
    ASSERT_EQ(diagTensor.getValue<float>({2, 0}), 0.0);
    ASSERT_EQ(diagTensor.getValue<float>({2, 1}), 0.0);

    Tensor eye = Tensor::eye(2, dtypes::kFloat, dtypes::kCPU, false);
    std::cout << "identity tensor: " << eye << std::endl;
    ASSERT_EQ(eye.getValue<float>({0, 0}), 1.0);
    ASSERT_EQ(eye.getValue<float>({1, 1}), 1.0);
    ASSERT_EQ(eye.getValue<float>({0, 1}), 0.0);
    ASSERT_EQ(eye.getValue<float>({1, 0}), 0.0);
}

// check equality operators
TEST(Tensor, EqualityOperators)
{

    Tensor one = Tensor({1.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor two = Tensor({2.0}, dtypes::kFloat, dtypes::kCPU, false);

    ASSERT_TRUE(one == one);
    ASSERT_TRUE(one != two);
}

// test manipulation of elements of tensor
TEST(Tensor, ElementMapipulation)
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
TEST(Tensor, simpleArithmeticFloat)
{

    // test simple addition
    Tensor one = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ((one + one).getValue<float>(), 2.0);
    ASSERT_EQ((one - one).getValue<float>(), 0.0);

    // test multiplication of tensors
    Tensor ten = Tensor({10.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor five = Tensor({5.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ((ten / five).getValue<float>(), 2.0);
    ASSERT_EQ((ten * five).getValue<float>(), 50.0);
    ASSERT_EQ(Tensor::pow(ten, 2.0).getValue<float>(), 100.0);

    // test sqrt
    Tensor four = Tensor({4.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(Tensor::pow(four, 0.5).getValue<float>(), 2.0);

    // test scaling by float
    ASSERT_NEAR((one * 1.234).getValue<float>(), 1.234, 1e-6);
    ASSERT_NEAR((one / 2.0).getValue<float>(), 0.5, 1e-6);
    ASSERT_NEAR((1.234 * one).getValue<float>(), 1.234, 1e-6);

    // addition of float
    ASSERT_EQ((one + 1.0).getValue<float>(), 2.0);
    ASSERT_EQ((one - 1.0).getValue<float>(), 0.0);
    ASSERT_EQ((1.0 + one).getValue<float>(), 2.0);
    ASSERT_EQ((1.0 - one).getValue<float>(), 0.0);

    // negation
    ASSERT_EQ((-one).getValue<float>(), -1.0);
}

// check some basic arithmetic
TEST(Tensor, simpleArithmeticComplexFloat)
{

    // test addition for complex value with real component
    Tensor one = Tensor::ones({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    ASSERT_EQ((one + one).getValue<std::complex<float>>(), std::complex<float>(2.0, 0.0));
    ASSERT_EQ((one - one).getValue<std::complex<float>>(), std::complex<float>(0.0, 0.0));

    // check that sqrt -1 = i
    ASSERT_EQ((Tensor::pow(-one, 0.5)).getValue<std::complex<float>>(), std::complex<float>(0.0, -1.0));

    // imag unit to use in testing
    Tensor imag = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    imag.setValue({0}, std::complex<float>(0.0, 1.0));

    // check that i^2 = -1
    ASSERT_EQ((Tensor::pow(imag, 2.0)).getValue<std::complex<float>>(), std::complex<float>(-1.0, 0.0));

    // test addition
    ASSERT_EQ((one + imag).getValue<std::complex<float>>(), std::complex<float>(1.0, 1.0));

    // test multiplication by real scalar
    Tensor ten = Tensor({10.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor five = Tensor({5.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(Tensor::div(imag, five).getValue<std::complex<float>>(), std::complex<float>(0.0, 0.2));
    ASSERT_EQ(Tensor::mul(imag, five).getValue<std::complex<float>>(), std::complex<float>(0.0, 5.0));

    // test scaling by real float
    ASSERT_EQ(Tensor::scale(imag, 1.234f).getValue<std::complex<float>>(), std::complex<float>(0.0, 1.234));

    // test scaling by complex float
    ASSERT_EQ(Tensor::scale(imag, std::complex<float>(1.0, 1.0)).getValue<std::complex<float>>(),
              std::complex<float>(-1.0, 1.0));

    // test complex operations
    ASSERT_EQ(imag.imag().getValue<float>(), 1.0);
    ASSERT_EQ(imag.real().getValue<float>(), 0.0);
    ASSERT_EQ((one + imag).conj(), (one - imag));

    // proof of eulers identity
    Tensor euler = Tensor({static_cast<float>(std::exp(1.0))}, dtypes::kComplexFloat, dtypes::kCPU, false);
    std::complex<float> testVal = Tensor::pow(euler, std::complex<float>(0.0, M_PI)).getValue<std::complex<float>>();
    ASSERT_NEAR(testVal.real(), -1.0, 1e-6);
    ASSERT_NEAR(testVal.imag(), 0.0, 1e-6);

    // other complex operations
    ASSERT_NEAR(imag.angle().getValue<float>(), M_PI / 2.0, 1e-5);
    ASSERT_NEAR(imag.abs().getValue<float>(), 1.0, 1e-5);
}

TEST(Tensor, Summation)
{

    Tensor tensor = Tensor::ones({3, 3}, dtypes::kFloat, dtypes::kCPU, false);

    ASSERT_EQ(tensor.sum(tensor).getValue<float>(), 9.0);

    Tensor sum = tensor.sum({1});

    ASSERT_EQ(sum.getValue<float>({0}), 3.0);
    ASSERT_EQ(sum.getValue<float>({1}), 3.0);
    ASSERT_EQ(sum.getValue<float>({2}), 3.0);

    Tensor cumsum = tensor.cumsum(1);

    ASSERT_EQ(cumsum.getValue<float>({0, 0}), 1.0);
    ASSERT_EQ(cumsum.getValue<float>({0, 1}), 2.0);
    ASSERT_EQ(cumsum.getValue<float>({0, 2}), 3.0);
}

TEST(Tensor, GetVariantValue)
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
TEST(Tensor, StandardFunctionsFloat)
{

    float theta = 1.234;
    Tensor thetaTensor = Tensor({theta}, dtypes::kComplexFloat, dtypes::kCPU, false);

    ASSERT_EQ(Tensor::sin(thetaTensor).getValue<float>(), std::sin(theta));
    ASSERT_EQ(Tensor::cos(thetaTensor).getValue<float>(), std::cos(theta));
    ASSERT_EQ(Tensor::exp(thetaTensor).getValue<float>(), std::exp(theta));
}

// check inplace functions
TEST(Tensor, InPlaceMatmul)
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

TEST(Tensor, InPlaceMul)
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

TEST(Tensor, InPlaceDiv)
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

TEST(Tensor, InPlaceScale)
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

TEST(Tensor, InPlaceScaleComplex)
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

TEST(Tensor, InPlacePow)
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

TEST(Tensor, InPlaceTranspose)
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
TEST(Tensor, MatrixFloat)
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

TEST(Tensor, AccessedTensor1D)
{

    auto tensor = AccessedTensor<float, 1, dtypes::kCPU>::zeros({3}, false);

    tensor.setValue(1.0, 1);

    ASSERT_EQ(tensor.getValue(1), 1.0);
}

TEST(Tensor, AccessedTensor2D)
{

    auto tensor = AccessedTensor<float, 2, dtypes::kCPU>::zeros({3, 3}, false);

    tensor.setValue(2.0, 1, 1);

    ASSERT_EQ(tensor.getValue(1, 1), 2.0);
}

TEST(Tensor, AccessedTensor3D)
{

    auto tensor = AccessedTensor<float, 3, dtypes::kCPU>::zeros({3, 3, 3}, false);

    tensor.setValue(3.0, 1, 1, 1);

    ASSERT_EQ(tensor.getValue(1, 1, 1), 3.0);
}
