// pybind11 stuff
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

// nuTens stuff
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>

namespace py = pybind11;

void initTensor(py::module & /*m*/);
void initPropagator(py::module & /*m*/);
void initDtypes(py::module & /*m*/);

// initialise the top level module "_pyNuTens"
// NOLINTNEXTLINE
PYBIND11_MODULE(_pyNuTens, m)
{
    m.doc() = "Library to calculate neutrino oscillations";
    initTensor(m);
    initPropagator(m);
    initDtypes(m);

#ifdef VERSION_INFO
    m.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

void initTensor(py::module &m)
{
    auto m_tensor = m.def_submodule("tensor");

    py::class_<Tensor>(m_tensor, "Tensor")
        .def(py::init()) // <- default constructor
        .def(py::init<std::vector<float>, NTdtypes::scalarType, NTdtypes::deviceType, bool>())

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

        // setters
        .def("set_value", py::overload_cast<const Tensor &, const Tensor &>(&Tensor::setValue),
             "Set a value at a specific index of this tensor")
        .def("set_value",
             py::overload_cast<const std::vector<std::variant<int, std::string>> &, const Tensor &>(&Tensor::setValue),
             "Set a value at a specific index of this tensor")
        .def("set_value", py::overload_cast<const std::vector<int> &, float>(&Tensor::setValue),
             "Set a value at a specific index of this tensor")
        .def("set_value", py::overload_cast<const std::vector<int> &, std::complex<float>>(&Tensor::setValue),
             "Set a value at a specific index of this tensor")

        // getters
        .def("get_shape", &Tensor::getShape, "Get the shape of this tensor")
        .def("get_values", &Tensor::getValues, "Get the subset of values in this tensor at a specified location")
        .def("get_value", &Tensor::getVariantValue, "Get the data stored at a particular index of the tensor")

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

        // operator overloads
        .def(-py::self);

    ; // end of Tensor non-static functions

    // Tensor creation functions
    m_tensor.def("eye", &Tensor::eye, "Create a tensor initialised with an identity matrix");
    m_tensor.def("rand", &Tensor::rand, "Create a tensor initialised with random values");
    m_tensor.def("diag", &Tensor::diag, "Create a tensor with specified values along the diagonal");
    m_tensor.def("ones", &Tensor::ones, "Create a tensor initialised with ones");
    m_tensor.def("zeros", &Tensor::zeros, "Create a tensor initialised with zeros");

    // maffs
    m_tensor.def("matmul", &Tensor::matmul, "Matrix multiplication");
    m_tensor.def("outer", &Tensor::outer, "Tensor outer product");
    m_tensor.def("mul", &Tensor::mul, "Element-wise multiplication");
    m_tensor.def("div", &Tensor::div, "Element-wise division");
    m_tensor.def("pow", py::overload_cast<const Tensor &, float>(&Tensor::pow), "Raise to scalar power");
    m_tensor.def("pow", py::overload_cast<const Tensor &, std::complex<float>>(&Tensor::pow), "Raise to scalar power");
    m_tensor.def("exp", &Tensor::exp, "Take exponential");
    m_tensor.def("transpose", &Tensor::transpose, "Get the matrix transpose");
    m_tensor.def("scale", py::overload_cast<const Tensor &, float>(&Tensor::scale), "Scalar multiplication");
    m_tensor.def("scale", py::overload_cast<const Tensor &, std::complex<float>>(&Tensor::scale),
                 "Scalar multiplication");
    m_tensor.def("sin", &Tensor::sin, "Element-wise trigonometric sine function");
    m_tensor.def("cos", &Tensor::cos, "Element-wise trigonometric cosine function");
    m_tensor.def("sum", py::overload_cast<const Tensor &>(&Tensor::sum), "Get the sum of all values in a tensor");
    m_tensor.def("sum", py::overload_cast<const Tensor &, const std::vector<long int> &>(&Tensor::sum),
                 "Get the sum of all values in a tensor");
    m_tensor.def("cumsum", py::overload_cast<const Tensor &, int>(&Tensor::cumsum),
                 "Get the cumulative sum over some dimension");
    // m_tensor.def("eig", &Tensor::eig. "calculate eigenvalues") <- Will need to define some additional fn to return
    // tuple of values
}

void initPropagator(py::module &m)
{
    auto m_propagator = m.def_submodule("propagator");

    py::class_<Propagator>(m_propagator, "Propagator")
        .def(py::init<int, float>())
        .def("calculate_probabilities", &Propagator::calculateProbs,
             "Calculate the oscillation probabilities for neutrinos of specified energies")
        .def("set_matter_solver", &Propagator::setMatterSolver,
             "Set the matter effect solver that the propagator should use")
        .def("set_masses", &Propagator::setMasses, "Set the neutrino mass state eigenvalues")
        .def("set_PMNS", py::overload_cast<Tensor &>(&Propagator::setPMNS),
             "Set the PMNS matrix that the propagator should use")
        .def("set_PMNS", py::overload_cast<const std::vector<int> &, float>(&Propagator::setPMNS),
             "Set the PMNS matrix that the propagator should use")
        .def("set_PMNS", py::overload_cast<const std::vector<int> &, std::complex<float>>(&Propagator::setPMNS),
             "Set the PMNS matrix that the propagator should use");

    py::class_<BaseMatterSolver, std::shared_ptr<BaseMatterSolver>>(m_propagator, "BaseSolver");

    py::class_<ConstDensityMatterSolver, std::shared_ptr<ConstDensityMatterSolver>, BaseMatterSolver>(
        m_propagator, "ConstDensitySolver")
        .def(py::init<int, float>());
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