// pybind11 stuff
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

// nuTens stuff
#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>

namespace py = pybind11;

void initTensor(py::module &);
void initPropagator(py::module &);
void initDtypes(py::module &);

// initialise the top level module "pyNuTens"
PYBIND11_MODULE(pyNuTens, m)
{

    initTensor(m);
    initPropagator(m);
    initDtypes(m);
}

void initTensor(py::module &m)
{
    auto m_tensor = m.def_submodule("tensor");

    py::class_<Tensor>(m_tensor, "Tensor")
        .def(py::init()) // <- default constructor
        .def(py::init<std::vector<float>, NTdtypes::scalarType, NTdtypes::deviceType, bool>())

        // Tensor creation functions
        .def("eye", &Tensor::eye, "Create a tensor initialised with an identity matrix")
        .def("rand", &Tensor::rand, "Create a tensor initialised with random values")
        .def("diag", &Tensor::diag, "Create a tensor with specified values along the diagonal")
        .def("ones", &Tensor::ones, "Create a tensor initialised with ones")
        .def("zeros", &Tensor::zeros, "Create a tensor initialised with zeros")

        // property setters
        .def("dtype", &Tensor::dType, py::return_value_policy::reference, "Set the data type of the tensor")
        .def("device", &Tensor::device, py::return_value_policy::reference, "Set the device that the tensor lives on")
        .def("requires_grad", &Tensor::requiresGrad, py::return_value_policy::reference,
             "Set Whether or not this tensor requires gradient to be calculated")
        .def("has_batch_dim", &Tensor::hasBatchDim, py::return_value_policy::reference,
             "Set Whether or not the first dimension should be interpreted as a batch dim for this tensor")

        // utilities
        .def("to_string", &Tensor::toString, "print out a summary of this tensor to a string")
        .def("add_batch_dim", &Tensor::addBatchDim, py::return_value_policy::reference,
             "Add a batch dimension to the start of this tensor if it doesn't have one already")

        // getters
        .def("get_shape", &Tensor::getShape, "Get the shape of this tensor")
        .def("get_values", &Tensor::getValues, "Get the subset of values in this tensor at a specified location")

        // maffs
        .def("matmul", &Tensor::getShape, "Matrix multiplication")
        .def("outer", &Tensor::outer, "Tensor outer product")
        .def("mul", &Tensor::mul, "Element-wise multiplication")
        .def("div", &Tensor::div, "Element-wise division")
        .def("pow", py::overload_cast<const Tensor &, float>(&Tensor::pow), "Raise to scalar power")
        .def("pow", py::overload_cast<const Tensor &, std::complex<float>>(&Tensor::pow), "Raise to scalar power")
        .def("exp", &Tensor::exp, "Take exponential")
        .def("transpose", &Tensor::transpose, "Get the matrix transpose")
        .def("scale", py::overload_cast<const Tensor &, float>(&Tensor::scale), "Scalar multiplication")
        .def("scale", py::overload_cast<const Tensor &, std::complex<float>>(&Tensor::scale), "Scalar multiplication")
        .def("sin", &Tensor::sin, "Element-wise trigonometric sine function")
        .def("cos", &Tensor::cos, "Element-wise trigonometric cosine function")
        .def("sum", py::overload_cast<const Tensor &>(&Tensor::sum), "Get the sum of all values in a tensor")
        .def("sum", py::overload_cast<const Tensor &, const std::vector<long int> &>(&Tensor::sum),
             "Get the sum of all values in a tensor")
        .def("cumsum", py::overload_cast<const Tensor &, int>(&Tensor::cumsum),
             "Get the cumulative sum over some dimension")
        // .def("eig", &Tensor::eig. "calculate eigenvalues") <- Will need to define some additional fn to return tuple
        // of values

        // complex number stuff
        .def("real", &Tensor::real, "Get real part of a complex tensor")
        .def("imag", &Tensor::imag, "Get imaginary part of a complex tensor")
        .def("conj", &Tensor::conj, "Get complex conjugate of a complex tensor")
        .def("angle", &Tensor::angle, "Get element-wise phases of a complex tensor")
        .def("abs", &Tensor::abs, "Get element-wise magnitudes of a complex tensor")

        // gradient stuff
        .def("backward", &Tensor::backward, py::call_guard<py::gil_scoped_release>(),
             "Do the backward propagation from this tensor")
        .def("grad", &Tensor::grad, "Get the accumulated gradient stored in this tensor after calling backward()")

        ;
}

void initPropagator(py::module &m)
{
    auto m_propagator = m.def_submodule("propagator");
}

void initDtypes(py::module &m)
{
    auto m_dtypes = m.def_submodule("dtype");

    py::enum_<NTdtypes::scalarType>(m_dtypes, "scalar_type")
        .value("int", NTdtypes::scalarType::kInt)
        .value("float", NTdtypes::scalarType::kFloat)
        .value("double", NTdtypes::scalarType::kDouble)
        .value("complex_float", NTdtypes::scalarType::kComplexFloat)
        .value("complex_double", NTdtypes::scalarType::kComplexDouble)

        ;

    py::enum_<NTdtypes::deviceType>(m_dtypes, "device_type")
        .value("cpu", NTdtypes::deviceType::kCPU)
        .value("gpu", NTdtypes::deviceType::kGPU)

        ;
}