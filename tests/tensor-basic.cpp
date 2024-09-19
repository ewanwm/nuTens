
#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>

/*
    Do some very basic tests of tensor functionality
    e.g. test that complex matrices work as expected, 1+1 == 2 etc.
*/

int main()
{
    NT_PROFILE_BEGINSESSION("tensor-basic-test");

    NT_PROFILE();

    std::cout << "Tensor library: " << Tensor::getTensorLibrary() << std::endl;

    std::cout << "########################################" << std::endl;
    std::cout << "Float: " << std::endl;
    Tensor tensorFloat = Tensor::zeros({3, 3}, NTdtypes::kDouble).dType(NTdtypes::kFloat).device(NTdtypes::kCPU);
    tensorFloat.setValue({0, 0}, 0.0);
    tensorFloat.setValue({0, 1}, 1.0);
    tensorFloat.setValue({0, 2}, 2.0);

    tensorFloat.setValue({1, 0}, 3.0);
    tensorFloat.setValue({1, 1}, 4.0);
    tensorFloat.setValue({1, 2}, 5.0);

    tensorFloat.setValue({2, 0}, 6.0);
    tensorFloat.setValue({2, 1}, 7.0);
    tensorFloat.setValue({2, 2}, 8.0);
    std::cout << "real: " << std::endl << tensorFloat.real() << std::endl;
    std::cout << "Middle value: " << tensorFloat.getValue<float>({1, 1}) << std::endl;
    std::cout << "tensorFloat({'...', 1}) = " << tensorFloat.getValues({1, "..."}) << std::endl;

    Tensor realSquared = Tensor::matmul(tensorFloat, tensorFloat);
    std::cout << "Squared: " << std::endl;
    std::cout << realSquared << std::endl;
    std::cout << "########################################" << std::endl << std::endl;

    std::cout << "########################################" << std::endl;
    std::cout << "Complex float: " << std::endl;
    Tensor tensorComplex = Tensor::zeros({3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);
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

    Tensor ones = Tensor::ones({3, 3}, NTdtypes::kFloat);
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
    Tensor ones_scaleTest = Tensor::ones({2, 2}).dType(NTdtypes::kFloat).requiresGrad(true);
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

    Tensor complexGradTest = Tensor::zeros({2, 2}, NTdtypes::kComplexFloat);
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