#include <nuTens/tensors/autograd.hpp>
#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <nuTens/testing/utils.hpp>

#include <complex>
#include <gtest/gtest.h>

using namespace nuTens;

// magic numbers are fine for testing!
// NOLINTBEGIN(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)

// cognitive complexity is heavily inflated by the gtest macros
// but they don't actually decrease readability
// NOLINTBEGIN(readability-function-cognitive-complexity)

template <typename T>
void testElementManipulation(const dtypes::scalarType &dtype, const dtypes::deviceType &deviceType)
{

    auto tensorFloat = Tensor::zeros({2, 2}, dtype, deviceType, false);

    tensorFloat.setValue({0, 0}, (T)0.0);
    tensorFloat.setValue({0, 1}, (T)1.0);

    tensorFloat.setValue({1, 0}, (T)2.0);
    tensorFloat.setValue({1, 1}, (T)3.0);

    std::cout << "Test matrix: \n" << tensorFloat << std::endl;

    // test slicing
    Tensor slice = tensorFloat.getValues({1, "..."});
    ASSERT_EQ(slice.getValue<T>({0}), 2.0);
    ASSERT_EQ(slice.getValue<T>({1}), 3.0);
}

template <typename T> void testTensorCreation(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{
    Tensor uninit;
    ASSERT_FALSE(uninit.isInitialised());

    uninit = Tensor::zeros({10}, dtype, deviceType, false);
    ASSERT_TRUE(uninit.isInitialised());

    Tensor zero = Tensor::zeros({1}, dtype, deviceType, false);
    std::cout << "zero tensor: " << zero << std::endl;
    ASSERT_EQ(zero.getValue<T>(), T(0.0));

    std::vector<long int> shape = zero.getShape();
    ASSERT_EQ(shape.size(), 1);
    ASSERT_EQ(shape[0], 1);

    // test making tensor using setter functions
    Tensor zeroSetters = Tensor::zeros({1}).dType(dtype).device(deviceType).requiresGrad(false);
    ASSERT_EQ(zeroSetters.getValue<T>(), T(0.0));

    Tensor one = Tensor::ones({1}, dtype, deviceType, false);
    std::cout << "one tensor: " << one << std::endl;
    ASSERT_EQ(one.getValue<T>(), T(1.0));

    Tensor three = Tensor({3.0}, dtype, deviceType, false);
    ASSERT_EQ(three.getValue<T>(), T(3.0));

    Tensor rand = Tensor::rand({1}, dtype, deviceType, false);

    float randUpperLimit = 1.0;
    if ((dtype == dtypes::kComplexDouble) || (dtype == dtypes::kComplexFloat))
    {
        randUpperLimit = 2.0;
    }

    ASSERT_LE(rand.abs().getValue<float>(), randUpperLimit);
    ASSERT_GE(rand.abs().getValue<float>(), 0.0);

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

template <typename T> void testComplexTensorCreation(dtypes::scalarType dtype, dtypes::deviceType deviceType)
{

    // test the dedicated complex tensor builder function
    Tensor complex = Tensor::TensorComplex({T(1.234, 5.678)}, dtype, deviceType, false);

    ASSERT_EQ(complex.getValue<T>(), T(1.234, 5.678));
}

void testEqualityOperators(dtypes::scalarType dtype, dtypes::deviceType deviceType)
{
    Tensor one = Tensor({1.0}, dtype, deviceType, false);
    Tensor two = Tensor({2.0}, dtype, deviceType, false);

    ASSERT_TRUE(one == one);
    ASSERT_TRUE(one != two);
}

template <typename T> void testStandardFunctions(dtypes::scalarType dtype, dtypes::deviceType deviceType)
{
    float theta = 1.234;
    Tensor thetaTensor = Tensor({theta}, dtype, deviceType, false);

    ASSERT_EQ(Tensor::sin(thetaTensor).getValue<float>(), std::sin(theta));
    ASSERT_EQ(Tensor::cos(thetaTensor).getValue<float>(), std::cos(theta));
    ASSERT_EQ(Tensor::exp(thetaTensor).getValue<float>(), std::exp(theta));
    ASSERT_EQ(Tensor::log(thetaTensor).getValue<float>(), std::log(theta));
}

template <typename T> void testSummation(const dtypes::scalarType dtype, dtypes::deviceType deviceType)
{

    Tensor tensor = Tensor::ones({3, 3}, dtype, deviceType, false);

    ASSERT_EQ(tensor.sum().getValue<T>(), 9.0);

    Tensor sum = tensor.sum({1});

    ASSERT_EQ(sum.getValue<T>({0}), 3.0);
    ASSERT_EQ(sum.getValue<T>({1}), 3.0);
    ASSERT_EQ(sum.getValue<T>({2}), 3.0);

    Tensor cumsum = tensor.cumsum(1);

    ASSERT_EQ(cumsum.getValue<T>({0, 0}), 1.0);
    ASSERT_EQ(cumsum.getValue<T>({0, 1}), 2.0);
    ASSERT_EQ(cumsum.getValue<T>({0, 2}), 3.0);
}

template <typename T> void testMatrixOperations(dtypes::scalarType dtype, dtypes::deviceType deviceType)
{
    auto tensorFloat = Tensor::zeros({2, 2}, dtype, deviceType, false);
    auto eye = Tensor::eye(2, dtype, deviceType, false);

    tensorFloat.setValue({0, 0}, 0.0);
    tensorFloat.setValue({0, 1}, 1.0);

    tensorFloat.setValue({1, 0}, 2.0);
    tensorFloat.setValue({1, 1}, 3.0);

    std::cout << "Test matrix: \n" << tensorFloat << std::endl;

    // test matrix multiplication
    Tensor squared = Tensor::matmul(tensorFloat, tensorFloat);
    ASSERT_EQ(squared.getValue<T>({0, 0}), 2.0);
    ASSERT_EQ(squared.getValue<T>({0, 1}), 3.0);
    ASSERT_EQ(squared.getValue<T>({1, 0}), 6.0);
    ASSERT_EQ(squared.getValue<T>({1, 1}), 11.0);

    // test multiplication by identity matrix
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<T>({0, 0}), 0.0);
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<T>({0, 1}), 1.0);
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<T>({1, 0}), 2.0);
    ASSERT_EQ(Tensor::matmul(eye, tensorFloat).getValue<T>({1, 1}), 3.0);

    // test matrix addition
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<T>({0, 0}), 0.0);
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<T>({0, 1}), 2.0);
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<T>({1, 0}), 4.0);
    ASSERT_EQ((tensorFloat + tensorFloat).getValue<T>({1, 1}), 6.0);

    // test transpose
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0, 1)).getValue<T>({0, 0}), 0.0);
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0, 1)).getValue<T>({1, 0}), 1.0);
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0, 1)).getValue<T>({0, 1}), 2.0);
    ASSERT_EQ((Tensor::transpose(tensorFloat, 0, 1)).getValue<T>({1, 1}), 3.0);

    // test outer product of two vectors
    Tensor vec1 = Tensor::zeros({2}, dtype, deviceType, false);
    Tensor vec2 = Tensor::zeros({2}, dtype, deviceType, false);

    vec1.setValue({0}, 1.0);
    vec1.setValue({1}, 2.0);
    vec2.setValue({0}, 3.0);
    vec2.setValue({1}, 4.0);

    Tensor outer = Tensor::outer(vec1, vec2);

    ASSERT_EQ(outer.getValue<T>({0, 0}), 3.0);
    ASSERT_EQ(outer.getValue<T>({0, 1}), 4.0);
    ASSERT_EQ(outer.getValue<T>({1, 0}), 6.0);
    ASSERT_EQ(outer.getValue<T>({1, 1}), 8.0);
}

template <typename T> void testArithmeticFloatType(const dtypes::scalarType dtype, dtypes::deviceType deviceType)
{

    // test simple addition
    Tensor one = Tensor::ones({1}, dtype, deviceType, false);
    ASSERT_EQ((one + one).getValue<T>(), 2.0);
    ASSERT_EQ((one - one).getValue<T>(), 0.0);

    ASSERT_EQ(Tensor::add(one, one).getValue<T>(), 2.0);
    ASSERT_EQ(Tensor::add(one, -one).getValue<T>(), 0.0);

    // test multiplication of tensors
    Tensor ten = Tensor({10.0}, dtype, deviceType, false);
    Tensor five = Tensor({5.0}, dtype, deviceType, false);
    ASSERT_EQ((ten / five).getValue<T>(), 2.0);
    ASSERT_EQ((ten * five).getValue<T>(), 50.0);
    ASSERT_EQ(Tensor::square(ten).getValue<T>(), 100.0);

    // test sqrt
    Tensor four = Tensor({4.0}, dtype, deviceType, false);
    ASSERT_EQ(Tensor::sqrt(four).getValue<T>(), 2.0);

    // test scaling by float
    Tensor val = (one * (T)1.234);
    ASSERT_NEAR(val.getValue<T>(), 1.234, 1e-6);
    val = (one / (T)2.0);
    ASSERT_NEAR(val.getValue<T>(), 0.5, 1e-6);
    val = ((T)1.234 * one);
    ASSERT_NEAR(val.getValue<T>(), 1.234, 1e-6);

    // test dividing bt float
    val = Tensor::div(ten, (T)5.0);
    ASSERT_EQ(val.getValue<T>(), 2.0);
    val = (ten / (T)5.0);
    ASSERT_EQ(val.getValue<T>(), 2.0);

    // addition of float
    val = Tensor::add(one, (T)1.0);
    ASSERT_EQ(val.getValue<T>(), 2.0);
    val = (one + (T)1.0);
    ASSERT_EQ(val.getValue<T>(), 2.0);
    val = (one - (T)1.0);
    ASSERT_EQ(val.getValue<T>(), 0.0);
    val = ((T)1.0 + one);
    ASSERT_EQ(val.getValue<T>(), 2.0);
    val = ((T)1.0 - one);
    ASSERT_EQ(val.getValue<T>(), 0.0);

    // negation
    ASSERT_EQ((-one).getValue<T>(), -1.0);
}

template <typename T> void testArithmeticComplexType(const dtypes::scalarType dtype, dtypes::deviceType deviceType)
{

    // the complex type used for this test
    typedef std::complex<T> complexType;
    Tensor testTensor;

    // test addition for complex value with real component
    Tensor one = Tensor::ones({1}, dtype, deviceType, false);
    ASSERT_EQ((one + one).getValue<complexType>(), complexType(2.0, 0.0));
    ASSERT_EQ((one - one).getValue<complexType>(), complexType(0.0, 0.0));

    // check that sqrt -1 = i
    Tensor sqrtNegOneTensor = Tensor::sqrt(-one);
    ASSERT_NEAR(sqrtNegOneTensor.real().getValue<T>(), 0.0, 1e-6);
    ASSERT_NEAR(sqrtNegOneTensor.imag().getValue<T>(), -1.0, 1e-6);

    // imag unit to use in testing
    Tensor imag = Tensor::zeros({1}, dtype, deviceType, false);
    imag.setValue({0}, complexType(0.0, 1.0));

    // check that i^2 = -1
    ASSERT_EQ((Tensor::square(imag)).getValue<complexType>(), complexType(-1.0, 0.0));

    // test addition
    ASSERT_EQ((one + imag).getValue<complexType>(), complexType(1.0, 1.0));
    testTensor = (one + complexType(0.0, 1.0));
    ASSERT_EQ(testTensor.getValue<complexType>(), complexType(1.0, 1.0));

    // test multiplication by real scalar
    Tensor ten = Tensor({10.0}, dtypes::kFloat, deviceType, false);
    Tensor five = Tensor({5.0}, dtypes::kFloat, deviceType, false);
    ASSERT_EQ(Tensor::div(imag, five).getValue<complexType>(), complexType(0.0, 0.2));
    ASSERT_EQ(Tensor::mul(imag, five).getValue<complexType>(), complexType(0.0, 5.0));
    testTensor = Tensor::div(imag, complexType(5.0, 0.0));
    ASSERT_EQ(testTensor.getValue<complexType>(), complexType(0.0, 0.2));

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
    Tensor euler = Tensor({std::exp(1.0F)}, dtype, deviceType, false);
    Tensor exp = Tensor::pow(euler, complexType(0.0, M_PI));
    complexType testVal = exp.getValue<complexType>();
    ASSERT_NEAR(testVal.real(), -1.0, 1e-6);
    ASSERT_NEAR(testVal.imag(), 0.0, 1e-6);

    // other complex operations
    ASSERT_NEAR(imag.angle().getValue<T>(), M_PI / 2.0, 1e-5);
    ASSERT_NEAR(imag.abs().getValue<T>(), 1.0, 1e-5);
}

template <typename T> void testNoGrad(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{

    auto noGradGuard = autograd::NoGrad();

    T grad = 1.234;
    Tensor one = Tensor::ones({1}, dtype, deviceType, true);
    Tensor result = one * grad;

    EXPECT_ANY_THROW(result.backward());
    EXPECT_ANY_THROW(autograd::grad(result, one));
}

template <typename T>
void testAutogradNoRequiresGradFail(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{

    T grad = 1.234;
    Tensor one = Tensor::ones({1}, dtype, deviceType, false);
    Tensor result = one * grad;

    EXPECT_ANY_THROW(result.backward());
    EXPECT_ANY_THROW(autograd::grad(result, one));
}

template <typename T>
void testDerivativesBasicScalarReal(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{

    T grad = 1.234;
    Tensor one = Tensor::ones({1}, dtype, deviceType, true);
    Tensor result = one * grad;
    result.backward();

    ASSERT_EQ(one.grad().getValue<T>(), grad);
    ASSERT_EQ(autograd::grad(result, one), one.grad());
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
    ASSERT_EQ(autograd::grad(result, one).conj(), grad);

    // test derivative of imaginary part of product
    one = Tensor::ones({1}, dtype, deviceType, true);
    result = (one * grad).imag();
    result.backward();

    gradTensor = one.grad();

    ASSERT_EQ(Tensor::scale(gradTensor.conj(), complexType(0.0, 1.0)), grad);
    ASSERT_EQ(Tensor::scale(autograd::grad(result, one).conj(), complexType(0.0, 1.0)), grad);
}

void testDerivativesStandardFunctions(const dtypes::scalarType dtype, const dtypes::deviceType deviceType)
{

    Tensor tensor = Tensor({1.2345}, dtype, deviceType, true);

    Tensor exp = Tensor::exp(tensor);
    exp.backward();

    ASSERT_EQ(Tensor::exp(tensor), tensor.grad());
    ASSERT_EQ(Tensor::exp(tensor), autograd::grad(exp, tensor));
    tensor.zeroGrad();

    Tensor cos = Tensor::cos(tensor);
    cos.backward();

    ASSERT_EQ(-Tensor::sin(tensor), tensor.grad());
    ASSERT_EQ(-Tensor::sin(tensor), autograd::grad(cos, tensor));
    tensor.zeroGrad();

    Tensor square = Tensor::square(tensor);
    square.backward();

    ASSERT_EQ(2.0 * tensor, tensor.grad());
    ASSERT_EQ(2.0 * tensor, autograd::grad(square, tensor));
    tensor.zeroGrad();
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

template <typename T> void testEig(dtypes::scalarType dtype, dtypes::deviceType deviceType)
{
    Tensor evals = Tensor::zeros({2}, dtype, deviceType, false);
    Tensor evecs = Tensor::zeros({2, 2}, dtype, deviceType, false);
    Tensor mat = Tensor::ones({2, 2}, dtype, deviceType, false);
    mat.setValue({0, 0}, 2.0);
    mat.setValue({0, 1}, 1.0);
    mat.setValue({1, 0}, 1.0);
    mat.setValue({1, 1}, 2.0);

    Tensor::eig(mat, evals, evecs);

    ASSERT_EQ(evals.getValue<T>({0}), 3.0);
    ASSERT_EQ(evals.getValue<T>({1}), 1.0);

    ASSERT_EQ(evecs.getValue<T>({0, 0}), evecs.getValue<T>({1, 0}));
    ASSERT_EQ(evecs.getValue<T>({0, 1}), -evecs.getValue<T>({1, 1}));
}

template <typename T> void testEigh(dtypes::scalarType dtype, dtypes::deviceType deviceType)
{
    Tensor evals = Tensor::zeros({2}, dtype, deviceType, false);
    Tensor evecs = Tensor::zeros({2, 2}, dtype, deviceType, false);
    Tensor mat = Tensor::ones({2, 2}, dtype, deviceType, false);
    mat.setValue({0, 0}, 2.0);
    mat.setValue({0, 1}, 1.0);
    mat.setValue({1, 0}, 1.0);
    mat.setValue({1, 1}, 2.0);

    Tensor::eigh(mat, evals, evecs);

    ASSERT_EQ(evals.getValue<T>({0}), 1.0);
    ASSERT_EQ(evals.getValue<T>({1}), 3.0);

    ASSERT_EQ(evecs.getValue<T>({0, 0}), -evecs.getValue<T>({1, 0}));
    ASSERT_EQ(evecs.getValue<T>({0, 1}), evecs.getValue<T>({1, 1}));
}

template <typename T> void testEigVals(dtypes::scalarType dtype, dtypes::deviceType deviceType)
{
    Tensor evals = Tensor::zeros({2}, dtype, deviceType, false);
    Tensor mat = Tensor::ones({2, 2}, dtype, deviceType, false);
    mat.setValue({0, 0}, 2.0);
    mat.setValue({0, 1}, 1.0);
    mat.setValue({1, 0}, 1.0);
    mat.setValue({1, 1}, 2.0);

    Tensor::eigvals(mat, evals);

    ASSERT_EQ(evals.getValue<T>({0}), 3.0);
    ASSERT_EQ(evals.getValue<T>({1}), 1.0);
}

template <typename T> void testEigValsh(dtypes::scalarType dtype, dtypes::deviceType deviceType)
{
    Tensor evals = Tensor::zeros({2}, dtype, deviceType, false);
    Tensor mat = Tensor::ones({2, 2}, dtype, deviceType, false);
    mat.setValue({0, 0}, 2.0);
    mat.setValue({0, 1}, 1.0);
    mat.setValue({1, 0}, 1.0);
    mat.setValue({1, 1}, 2.0);

    Tensor::eigvalsh(mat, evals);

    ASSERT_EQ(evals.getValue<T>({0}), 1.0);
    ASSERT_EQ(evals.getValue<T>({1}), 3.0);
}

// NOLINTEND(readability-function-cognitive-complexity)

// NOLINTEND(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
