#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>

#include <complex>
#include <gtest/gtest.h>

using namespace nuTens;

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)

// cognitive complexity is heavily inflated by the gtest macros
// but they don't actually decrease readability
// NOLINTBEGIN(readability-function-cognitive-complexity)

template <typename T> void testTensorCreation(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{
    Tensor zero = Tensor::zeros({1}, dtype, deviceType, false);
    std::cout << "zero tensor: " << zero << std::endl;
    ASSERT_EQ(zero.getValue<T>(), T(0.0));

    // test making tensor using setter functions
    Tensor zeroSetters = Tensor::zeros({1}).dType(dtype).device(deviceType).requiresGrad(false);
    ASSERT_EQ(zeroSetters.getValue<T>(), T(0.0));

    Tensor one = Tensor::ones({1}, dtype, deviceType, false);
    std::cout << "one tensor: " << one << std::endl;
    ASSERT_EQ(one.getValue<T>(), T(1.0));

    Tensor three = Tensor({3.0}, dtype, deviceType, false);
    ASSERT_EQ(three.getValue<T>(), T(3.0));

    Tensor rand = Tensor::rand({1}, dtype, deviceType, false);
    if ((dtype == dtypes::kComplexDouble) || (dtype == dtypes::kComplexFloat))
    {
        ASSERT_LE(rand.abs().getValue<float>(), 2.0);
        ASSERT_GE(rand.abs().getValue<float>(), 0.0);
    }
    else
    {
        ASSERT_LE(rand.getValue<float>(), 1.0);
        ASSERT_GE(rand.getValue<float>(), 0.0);
    }

    Tensor diagonal = Tensor({0.0, 1.0, 2.0}, dtype, deviceType, false);
    Tensor diagTensor = Tensor::diag(diagonal);
    std::cout << "diagonal tensor: \n" << diagTensor << std::endl;
    ASSERT_EQ(diagTensor.getValue<T>({0, 0}), T(0.0));
    ASSERT_EQ(diagTensor.getValue<T>({1, 1}), T(1.0));
    ASSERT_EQ(diagTensor.getValue<T>({2, 2}), T(2.0));

    ASSERT_EQ(diagTensor.getValue<T>({0, 1}), T(0.0));
    ASSERT_EQ(diagTensor.getValue<T>({0, 2}), T(0.0));
    ASSERT_EQ(diagTensor.getValue<T>({1, 0}), T(0.0));
    ASSERT_EQ(diagTensor.getValue<T>({1, 2}), T(0.0));
    ASSERT_EQ(diagTensor.getValue<T>({2, 0}), T(0.0));
    ASSERT_EQ(diagTensor.getValue<T>({2, 1}), T(0.0));

    Tensor eye = Tensor::eye(2, dtype, deviceType, false);
    std::cout << "identity tensor: " << eye << std::endl;
    ASSERT_EQ(eye.getValue<T>({0, 0}), T(1.0));
    ASSERT_EQ(eye.getValue<T>({1, 1}), T(1.0));
    ASSERT_EQ(eye.getValue<T>({0, 1}), T(0.0));
    ASSERT_EQ(eye.getValue<T>({1, 0}), T(0.0));
}

template <typename T> void testArithmeticFloatType(const dtypes::scalarType dtype)
{

    // test simple addition
    Tensor one = Tensor::ones({1}, dtype, dtypes::kCPU, false);
    ASSERT_EQ((one + one).getValue<T>(), 2.0);
    ASSERT_EQ((one - one).getValue<T>(), 0.0);

    // test multiplication of tensors
    Tensor ten = Tensor({10.0}, dtype, dtypes::kCPU, false);
    Tensor five = Tensor({5.0}, dtype, dtypes::kCPU, false);
    ASSERT_EQ((ten / five).getValue<T>(), 2.0);
    ASSERT_EQ((ten * five).getValue<T>(), 50.0);
    ASSERT_EQ(Tensor::square(ten).getValue<T>(), 100.0);

    // test sqrt
    Tensor four = Tensor({4.0}, dtype, dtypes::kCPU, false);
    ASSERT_EQ(Tensor::sqrt(four).getValue<T>(), 2.0);

    // test scaling by float
    ASSERT_NEAR((one * 1.234).getValue<T>(), 1.234, 1e-6);
    ASSERT_NEAR((one / 2.0).getValue<T>(), 0.5, 1e-6);
    ASSERT_NEAR((1.234 * one).getValue<T>(), 1.234, 1e-6);

    // addition of float
    ASSERT_EQ((one + 1.0).getValue<T>(), 2.0);
    ASSERT_EQ((one - 1.0).getValue<T>(), 0.0);
    ASSERT_EQ((1.0 + one).getValue<T>(), 2.0);
    ASSERT_EQ((1.0 - one).getValue<T>(), 0.0);

    // negation
    ASSERT_EQ((-one).getValue<T>(), -1.0);
}

template <typename T> void testArithmeticComplexType(const dtypes::scalarType dtype)
{

    // the complex type used for this test
    typedef std::complex<T> complexType;

    // test addition for complex value with real component
    Tensor one = Tensor::ones({1}, dtype, dtypes::kCPU, false);
    ASSERT_EQ((one + one).getValue<complexType>(), complexType(2.0, 0.0));
    ASSERT_EQ((one - one).getValue<complexType>(), complexType(0.0, 0.0));

    // check that sqrt -1 = i
    ASSERT_EQ((Tensor::pow(-one, 0.5)).getValue<complexType>(), complexType(0.0, -1.0));

    // imag unit to use in testing
    Tensor imag = Tensor::zeros({1}, dtype, dtypes::kCPU, false);
    imag.setValue({0}, complexType(0.0, 1.0));

    // check that i^2 = -1
    ASSERT_EQ((Tensor::square(imag)).getValue<complexType>(), complexType(-1.0, 0.0));

    // test addition
    ASSERT_EQ((one + imag).getValue<complexType>(), complexType(1.0, 1.0));

    // test multiplication by real scalar
    Tensor ten = Tensor({10.0}, dtypes::kFloat, dtypes::kCPU, false);
    Tensor five = Tensor({5.0}, dtypes::kFloat, dtypes::kCPU, false);
    ASSERT_EQ(Tensor::div(imag, five).getValue<complexType>(), complexType(0.0, 0.2));
    ASSERT_EQ(Tensor::mul(imag, five).getValue<complexType>(), complexType(0.0, 5.0));

    // test scaling by real float
    Tensor scaled = Tensor::scale(imag, (T)1.234);
    ASSERT_EQ(scaled.getValue<complexType>(), complexType(0.0, 1.234));

    // test scaling by complex float
    scaled = Tensor::scale(imag, complexType(1.0, 1.0));
    ASSERT_EQ(scaled.getValue<complexType>(), complexType(-1.0, 1.0));

    // test complex operations
    ASSERT_EQ(imag.imag().getValue<T>(), 1.0);
    ASSERT_EQ(imag.real().getValue<T>(), 0.0);
    ASSERT_EQ((one + imag).conj(), (one - imag));

    // proof of eulers identity
    Tensor euler = Tensor({static_cast<T>(std::exp(T(1.0)))}, dtype, dtypes::kCPU, false);
    Tensor exp = Tensor::pow(euler, complexType(0.0, M_PI));
    complexType testVal = exp.getValue<complexType>();
    ASSERT_NEAR(testVal.real(), -1.0, 1e-6);
    ASSERT_NEAR(testVal.imag(), 0.0, 1e-6);

    // other complex operations
    ASSERT_NEAR(imag.angle().getValue<T>(), M_PI / 2.0, 1e-5);
    ASSERT_NEAR(imag.abs().getValue<T>(), 1.0, 1e-5);
}

template <typename T>
void testDerivativesBasicScalarReal(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{

    T grad = 1.234;
    Tensor one = Tensor::ones({1}, dtype, deviceType, true);
    Tensor result = one * grad;
    result.backward();

    ASSERT_EQ(one.grad().getValue<T>(), grad);
}

template <typename T>
void testDerivativesBasicScalarComplex(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{

    typedef std::complex<T> complexType;

    complexType grad = complexType(1.234, 5.678);
    Tensor one = Tensor::ones({1}, dtype, deviceType, true);
    /// @todo test * override when added for complex scalars
    Tensor result = Tensor::scale(one, grad);

    result.real().backward();
    Tensor realGradTensor = one.grad();

    ASSERT_EQ(realGradTensor.getValue<complexType>().real(), grad.real());
    ASSERT_EQ(realGradTensor.getValue<complexType>().imag(), -grad.imag());

    // recompute the result to reset the gradient
    one = Tensor::ones({1}, dtype, deviceType, true);
    result = Tensor::scale(one, grad);

    result.imag().backward();
    Tensor imagGradTensor = one.grad();

    ASSERT_EQ(imagGradTensor.getValue<complexType>().real(), grad.imag());
    ASSERT_EQ(imagGradTensor.getValue<complexType>().imag(), grad.real());
}

template <typename T>
void testDerivativesBasicTensorReal(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{

    Tensor grad = Tensor({1.234}, dtype, deviceType, false);
    Tensor one = Tensor::ones({1}, dtype, deviceType, true);
    Tensor result = one * grad;

    result.backward();

    ASSERT_EQ(one.grad(), grad);
    EXPECT_THROW((void)grad.grad(), std::runtime_error);
}

template <typename T>
void testDerivativesBasicTensorComplex(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{

    typedef std::complex<T> complexType;

    Tensor grad = Tensor::scale(Tensor::ones({1}, dtype, deviceType, false), complexType(1.234, 5.678));

    // test derivative of real part of product
    Tensor one = Tensor::ones({1}, dtype, deviceType, true);
    Tensor result = (one * grad).real();
    result.backward();

    Tensor gradTensor = one.grad();

    ASSERT_EQ(gradTensor.conj(), grad);

    // test derivative of imaginary part of product
    one = Tensor::ones({1}, dtype, deviceType, true);
    result = (one * grad).imag();
    result.backward();

    gradTensor = one.grad();

    ASSERT_EQ(Tensor::scale(gradTensor.conj(), complexType(0.0, 1.0)), grad);
}

void testDerivativesStandardFunctions(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{

    Tensor tensor = Tensor({1.2345}, dtype, deviceType, true);

    Tensor exp = Tensor::exp(tensor);
    exp.backward();

    ASSERT_EQ(Tensor::exp(tensor), tensor.grad());
    tensor.zeroGrad();

    Tensor cos = Tensor::cos(tensor);
    cos.backward();

    ASSERT_EQ(-Tensor::sin(tensor), tensor.grad());
    tensor.zeroGrad();

    Tensor square = Tensor::square(tensor);
    square.backward();

    ASSERT_EQ(2.0 * tensor, tensor.grad());
    tensor.zeroGrad();
}

// NOLINTEND(readability-function-cognitive-complexity)

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)