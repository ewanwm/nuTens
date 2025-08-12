
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
    ASSERT_EQ(zero.getValue<float>(), 0.0);

    Tensor one = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(one.getValue<float>(), 1.0);

    Tensor three = Tensor({3.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(three.getValue<float>(), 3.0);

    Tensor diagonal = Tensor({0.0, 1.0, 2.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor diagTensor = Tensor::diag(diagonal);
    ASSERT_EQ(diagTensor.getValue<float>({0}), 0.0);
    ASSERT_EQ(diagTensor.getValue<float>({1}), 1.0);
    ASSERT_EQ(diagTensor.getValue<float>({2}), 2.0);

    Tensor eye = Tensor::eye(2, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(eye.getValue<float>({0,0}), 1.0);
    ASSERT_EQ(eye.getValue<float>({1,1}), 1.0);
    ASSERT_EQ(eye.getValue<float>({0,1}), 0.0);
    ASSERT_EQ(eye.getValue<float>({1,0}), 0.0);
}


// check some basic arithmetic
TEST(Tensor, simpleArithmeticFloat) {

    Tensor one = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ((one + one).getValue<float>(), 2.0);
    ASSERT_EQ((one - one).getValue<float>(), 0.0);

    Tensor ten  = Tensor({10.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor five = Tensor({5.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(Tensor::div(ten, five).getValue<float>(), 2.0);
    ASSERT_EQ(Tensor::mul(ten, five).getValue<float>(), 50.0);
    ASSERT_EQ(Tensor::pow(ten, 2.0).getValue<float>(), 100.0);

    Tensor four = Tensor({4.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(Tensor::pow(ten, 0.5).getValue<float>(), 2.0);

    ASSERT_EQ(Tensor::scale(one, 1.234).getValue<float>(), 1.234);
}

// check some basic arithmetic
TEST(Tensor, simpleArithmeticComplexFloat) {

    Tensor one = Tensor::ones({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    ASSERT_EQ((one + one).getValue<std::complex<float>>(), std::complex<float>(2.0, 0.0));
    ASSERT_EQ((one - one).getValue<std::complex<float>>(), std::complex<float>(0.0, 0.0));

}

int main()
{
    NT_PROFILE_BEGINSESSION("tensor-basic-test");

    NT_PROFILE();

    std::cout << "Tensor library: " << Tensor::getTensorLibrary() << std::endl;

    std::cout << "########################################" << std::endl;
    std::cout << "Float: " << std::endl;
    auto tensorFloat = AccessedTensor<double, 2, dtypes::kCPU>::zeros({3, 3}).requiresGrad(false);
    tensorFloat.setValue(0.0, 0, 0);
    tensorFloat.setValue(1.0, 0, 1);
    tensorFloat.setValue(2.0, 0, 2);
    
    tensorFloat.setValue(3.0, 1, 0);
    tensorFloat.setValue(4.0, 1, 1);
    tensorFloat.setValue(5.0, 1, 2);
    
    tensorFloat.setValue(6.0, 2, 0);
    tensorFloat.setValue(7.0, 2, 1);
    tensorFloat.setValue(8.0, 2, 2);
    std::cout << "tensor: " << std::endl << tensorFloat << std::endl;
    std::cout << "Middle value: " << tensorFloat.getValue(1, 1) << std::endl;
    std::cout << "tensorFloat({'...', 1}) = " << tensorFloat.getValues({1, "..."}) << std::endl;

    Tensor realSquared = Tensor::matmul(tensorFloat, tensorFloat);
    std::cout << "Squared: " << std::endl;
    std::cout << realSquared << std::endl;
    std::cout << "########################################" << std::endl << std::endl;

    std::cout << "########################################" << std::endl;
    std::cout << "Complex float: " << std::endl;
    Tensor tensorComplex = Tensor::zeros({3, 3}, dtypes::kComplexFloat).requiresGrad(false);
    tensorComplex.setValue({0, 0}, std::complex<float>(0.0J));
    tensorComplex.setValue({0, 1}, std::complex<float>(1.0J));
    tensorComplex.setValue({0, 2}, std::complex<float>(2.0J));

    tensorComplex.setValue({1, 0}, std::complex<float>(3.0J));
    tensorComplex.setValue({1, 1}, std::complex<float>(4.0J));
    tensorComplex.setValue({1, 2}, std::complex<float>(5.0J));

    tensorComplex.setValue({2, 0}, std::complex<float>(6.0J));
    tensorComplex.setValue({2, 1}, std::complex<float>(7.0J));
    tensorComplex.setValue({2, 2}, std::complex<float>(8.0J));

    std::cout << "real: " << std::endl << tensorComplex.real() << std::endl;
    std::cout << "imag: " << std::endl << tensorComplex.imag() << std::endl << std::endl;

    std::cout << "Complex conjugate: " << std::endl;
    std::cout << "real: " << std::endl << tensorComplex.conj().real() << std::endl;
    std::cout << "imag: " << std::endl << tensorComplex.conj().imag() << std::endl << std::endl;

    if (tensorComplex.imag() != -tensorComplex.conj().imag())
    {
        std::cerr << std::endl;
        std::cerr << "ERROR: Im(complex.conj()) != - Im(complex) " << std::endl;
        std::cerr << std::endl;
        return 1;
    }

    Tensor imagSquared = Tensor::matmul(tensorComplex, tensorComplex);
    std::cout << "Squared: " << std::endl;
    std::cout << imagSquared << std::endl;
    std::cout << "########################################" << std::endl << std::endl;

    // check if the real matrix squared is equal to the -ve of the imaginary one
    // squared
    if (realSquared != -imagSquared.real())
    {
        std::cerr << std::endl;
        std::cerr << "real**2 != -imaginary**2" << std::endl;
        std::cerr << std::endl;
        return 1;
    }

    Tensor ones = Tensor::ones({3, 3}, dtypes::kFloat);
    Tensor twos = ones + ones;

    std::cout << "ones + ones: " << std::endl;
    std::cout << twos << std::endl;

    // check that adding works
    if (twos.getValue<float>({1, 1}) != 2.0)
    {
        std::cerr << std::endl;
        std::cerr << "ERROR: 1 + 1 != 2 !!!!" << std::endl;
        std::cerr << std::endl;
        return 1;
    }

    // ######### test some of the basic autograd functionality ###########

    // first just a simple test of scaling by a constant factor
    Tensor ones_scaleTest = Tensor::ones({2, 2}).dType(dtypes::kFloat).requiresGrad(true);
    Tensor threes = Tensor::scale(ones_scaleTest, 3.0).sum();
    threes.backward();
    Tensor grad = ones_scaleTest.grad();
    std::cout << "Gradient of 2x2 ones multiplied by 3: " << std::endl;
    std::cout << grad << std::endl << std::endl;

    if ((grad.getValue<float>({0, 0}) != 3.0) || (grad.getValue<float>({0, 1}) != 3.0) ||
        (grad.getValue<float>({1, 0}) != 3.0) || (grad.getValue<float>({1, 1}) != 3.0))
    {
        std::cerr << std::endl;
        std::cerr << "ERROR: unexpected gradient when scaling by constant!!!!" << std::endl;
        std::cerr << std::endl;
        return 1;
    }

    Tensor complexGradTest = Tensor::zeros({2, 2}, dtypes::kComplexFloat).requiresGrad(false);
    complexGradTest.setValue({0, 0}, std::complex<float>(0.0 + 0.0J));
    complexGradTest.setValue({0, 1}, std::complex<float>(0.0 + 1.0J));
    complexGradTest.setValue({1, 0}, std::complex<float>(1.0 + 0.0J));
    complexGradTest.setValue({1, 1}, std::complex<float>(1.0 + 1.0J));
    complexGradTest.requiresGrad(true);

    Tensor complexGradSquared = Tensor::matmul(complexGradTest, complexGradTest).sum();
    std::cout << "sum(complexTest **2): " << std::endl;
    std::cout << complexGradSquared.real().getValue<float>() << " + " << complexGradSquared.imag().getValue<float>()
              << "i" << std::endl;
    complexGradSquared.backward();
    std::cout << "complex test gradient: " << std::endl;
    std::cout << "  Real: " << std::endl;
    std::cout << complexGradTest.grad().real() << std::endl;
    std::cout << "  Imag: " << std::endl;
    std::cout << complexGradTest.grad().imag() << std::endl << std::endl;

    NT_PROFILE_ENDSESSION();
}