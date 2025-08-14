
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
TEST(Tensor, TensorCreationFloat) {

    Tensor zero = Tensor::zeros({1}, dtypes::kFloat, dtypes::kCPU, false);
    std::cout << "zero tensor: " << zero << std::endl;
    ASSERT_EQ(zero.getValue<float>(), 0.0);

    Tensor one = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU, false);
    std::cout << "one tensor: " << one << std::endl;
    ASSERT_EQ(one.getValue<float>(), 1.0);

    Tensor three = Tensor({3.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(three.getValue<float>(), 3.0);

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
    ASSERT_EQ(eye.getValue<float>({0,0}), 1.0);
    ASSERT_EQ(eye.getValue<float>({1,1}), 1.0);
    ASSERT_EQ(eye.getValue<float>({0,1}), 0.0);
    ASSERT_EQ(eye.getValue<float>({1,0}), 0.0);
}

// test manipulation of elements of tensor
TEST(Tensor, ElementMapipulation) {

    auto tensorFloat = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);

    tensorFloat.setValue({0,0}, 0.0);
    tensorFloat.setValue({0,1}, 1.0);
    
    tensorFloat.setValue({1,0}, 2.0);
    tensorFloat.setValue({1,1}, 3.0);

    std::cout << "Test matrix: \n" << tensorFloat << std::endl;

    // test slicing
    Tensor slice = tensorFloat.getValues({1, "..."});
    ASSERT_EQ(slice.getValue<float>({0}), 2.0);
    ASSERT_EQ(slice.getValue<float>({1}), 3.0);

}

// check some basic arithmetic
TEST(Tensor, simpleArithmeticFloat) {

    // test simple addition
    Tensor one = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ((one + one).getValue<float>(), 2.0);
    ASSERT_EQ((one - one).getValue<float>(), 0.0);

    // test multiplication of scalars
    Tensor ten  = Tensor({10.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor five = Tensor({5.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(Tensor::div(ten, five).getValue<float>(), 2.0);
    ASSERT_EQ(Tensor::mul(ten, five).getValue<float>(), 50.0);
    ASSERT_EQ(Tensor::pow(ten, 2.0).getValue<float>(), 100.0);

    // test sqrt
    Tensor four = Tensor({4.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(Tensor::pow(four, 0.5).getValue<float>(), 2.0);

    // test scaling by float
    ASSERT_NEAR(Tensor::scale(one, 1.234).getValue<float>(), 1.234, 1e-6);
}

// check some basic arithmetic
TEST(Tensor, simpleArithmeticComplexFloat) {

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
    Tensor ten  = Tensor({10.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor five = Tensor({5.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(Tensor::div(imag, five).getValue<std::complex<float>>(), std::complex<float>(0.0, 0.2));
    ASSERT_EQ(Tensor::mul(imag, five).getValue<std::complex<float>>(), std::complex<float>(0.0, 5.0));

    // test scaling by real float
    ASSERT_EQ(Tensor::scale(imag, 1.234f).getValue<std::complex<float>>(), std::complex<float>(0.0, 1.234));

    // test scaling by complex float
    ASSERT_EQ(Tensor::scale(imag, std::complex<float>(1.0, 1.0)).getValue<std::complex<float>>(), std::complex<float>(-1.0, 1.0));

    // test complex operations
    ASSERT_EQ(imag.imag().getValue<float>(), 1.0);
    ASSERT_EQ(imag.real().getValue<float>(), 0.0);
    ASSERT_EQ((one + imag).conj(), (one - imag));

}

// check standard functions of real tensors
TEST(Tensor, StandardFunctionsFloat) {

    float theta = 1.234;
    Tensor thetaTensor = Tensor({theta}, dtypes::kComplexFloat, dtypes::kCPU, false);
    
    ASSERT_EQ(Tensor::sin(thetaTensor).getValue<float>(), std::sin(theta));
    ASSERT_EQ(Tensor::cos(thetaTensor).getValue<float>(), std::cos(theta));
    ASSERT_EQ(Tensor::exp(thetaTensor).getValue<float>(), std::exp(theta));

}

// test matrix operations for real tensor
TEST(Tensor, MatrixFloat) {

    auto tensorFloat = Tensor::zeros({2, 2}, dtypes::kFloat, dtypes::kCPU, false);
    auto eye = Tensor::eye(2, dtypes::kFloat, dtypes::kCPU, false);

    tensorFloat.setValue({0,0}, 0.0);
    tensorFloat.setValue({0,1}, 1.0);
    
    tensorFloat.setValue({1,0}, 2.0);
    tensorFloat.setValue({1,1}, 3.0);

    std::cout << "Test matrix: \n" << tensorFloat << std::endl;

    // test matrix multiplication
    Tensor squared = Tensor::matmul(tensorFloat, tensorFloat);
    ASSERT_EQ(squared.getValue<float>({0,0}), 2.0);
    ASSERT_EQ(squared.getValue<float>({0,1}), 3.0);
    ASSERT_EQ(squared.getValue<float>({1,0}), 6.0);
    ASSERT_EQ(squared.getValue<float>({1,1}), 11.0);

    // test multiplication by identity matrix
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<float>({0,0}), 0.0);
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<float>({0,1}), 1.0);
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<float>({1,0}), 2.0);
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<float>({1,1}), 3.0);

    // test matrix addition
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<float>({0,0}), 0.0);
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<float>({0,1}), 2.0);
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<float>({1,0}), 4.0);
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<float>({1,1}), 6.0);

    // test transpose
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0,1)).getValue<float>({0,0}), 0.0);
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0,1)).getValue<float>({1,0}), 1.0);
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0,1)).getValue<float>({0,1}), 2.0);
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0,1)).getValue<float>({1,1}), 3.0);

    // test outer product of two vectors
    Tensor vec1 = Tensor::zeros({2}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor vec2 = Tensor::zeros({2}, dtypes::kFloat, dtypes::kCPU, false);

    vec1.setValue({0}, 1.0);
    vec1.setValue({1}, 2.0);
    vec2.setValue({0}, 3.0);
    vec2.setValue({1}, 4.0);

    Tensor outer = Tensor::outer(vec1, vec2);
    
    ASSERT_EQ(outer.getValue<float>({0,0}), 3.0);
    ASSERT_EQ(outer.getValue<float>({0,1}), 4.0);
    ASSERT_EQ(outer.getValue<float>({1,0}), 6.0);
    ASSERT_EQ(outer.getValue<float>({1,1}), 8.0);
    
}
