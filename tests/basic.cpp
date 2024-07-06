#include <nuTens/tensors/tensor.hpp>
#include <nuTens/tensors/dtypes.hpp>
#include <complex.h>

/*
    Do some very basic tests of tensor functionality
    e.g. test that complex matrices work as expected, 1+1 == 2 etc.
*/

int main(){
    std::cout << "Tensor library: " << Tensor::getTensorLibrary() << std::endl;

    std::cout << "########################################" << std::endl;
    std::cout << "Float: " << std::endl;
    Tensor tensorFloat;
    tensorFloat.zeros({3, 3}, NTdtypes::kDouble).dType(NTdtypes::kFloat).device(NTdtypes::kCPU);
    tensorFloat.setValue({0,0}, 0.0);
    tensorFloat.setValue({0,1}, 1.0);
    tensorFloat.setValue({0,2}, 2.0);
    
    tensorFloat.setValue({1,0}, 3.0);
    tensorFloat.setValue({1,1}, 4.0);
    tensorFloat.setValue({1,2}, 5.0);
    
    tensorFloat.setValue({2,0}, 6.0);
    tensorFloat.setValue({2,1}, 7.0);
    tensorFloat.setValue({2,2}, 8.0);
    std::cout << "real: " << std::endl << tensorFloat.real() << std::endl;
    std::cout << "Middle value: " << tensorFloat.getValue<float>({1,1}) << std::endl;
    
    Tensor realSquared = Tensor::matmul(tensorFloat, tensorFloat);
    std::cout << "Squared: " << std::endl;
    std::cout << realSquared << std::endl;
    std::cout << "########################################" << std::endl << std::endl;
    
    
    std::cout << "########################################" << std::endl;
    std::cout << "Complex float: " << std::endl;
    Tensor tensorComplex;
    tensorComplex.zeros({3, 3}, NTdtypes::kComplexFloat);
    tensorComplex.setValue({0,0}, std::complex<float>(0.0j));
    tensorComplex.setValue({0,1}, std::complex<float>(1.0j));
    tensorComplex.setValue({0,2}, std::complex<float>(2.0j));
    
    tensorComplex.setValue({1,0}, std::complex<float>(3.0j));
    tensorComplex.setValue({1,1}, std::complex<float>(4.0j));
    tensorComplex.setValue({1,2}, std::complex<float>(5.0j));
    
    tensorComplex.setValue({2,0}, std::complex<float>(6.0j));
    tensorComplex.setValue({2,1}, std::complex<float>(7.0j));
    tensorComplex.setValue({2,2}, std::complex<float>(8.0j));

    std::cout << "real: " << std::endl << tensorComplex.real() << std::endl;
    std::cout << "imag: " << std::endl << tensorComplex.imag() << std::endl << std::endl;
    
    Tensor imagSquared = Tensor::matmul(tensorComplex, tensorComplex);
    std::cout << "Squared: " << std::endl;
    std::cout << imagSquared << std::endl;
    std::cout << "########################################" << std::endl << std::endl;

    // check if the real matrix squared is equal to the -ve of the imaginary one squared
    if( realSquared != - imagSquared.real()){
        std::cerr << std::endl;
        std::cerr << "real**2 != -imaginary**2" << std::endl;
        std::cerr << std::endl;
        return 1;
    }

    Tensor ones;
    ones.ones({3,3}, NTdtypes::kFloat);
    Tensor twos = ones + ones;

    std::cout << "ones + ones: " << std::endl;
    std::cout << twos << std::endl;

    // check that adding works
    if( twos.getValue<float>({1, 1}) != 2.0 ){
        std::cerr << std::endl;
        std::cerr << "ERROR: 1 + 1 != 2 !!!!" << std::endl;
        std::cerr << std::endl;
        return 1;
    }

}