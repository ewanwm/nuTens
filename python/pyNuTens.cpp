// pybind11 stuff
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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