// pybind11 stuff
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>

// nuTens stuff
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/propagator/units.hpp>
#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <tests/barger-propagator.hpp>

#if USE_PYTORCH
#include <torch/torch.h>
#include <torch/extension.h>
#endif

namespace py = pybind11;

void initTensor(py::module & /*m*/);
void initPropagator(py::module & /*m*/);
void initDtypes(py::module & /*m*/);
void initUnits(py::module & /*m*/);
void initTesting(py::module & /*m*/);

// initialise the top level module "_pyNuTens"
// NOLINTNEXTLINE
PYBIND11_MODULE(_pyNuTens, m)
{
    m.doc() = "Library to calculate neutrino oscillations";
    initDtypes(m);
    initUnits(m);
    initTensor(m);
    initPropagator(m);
    initTesting(m);

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
        .def("dtype", &Tensor::dType, py::return_value_policy::reference, 
            "Set the data type of the tensor",
            py::arg("new_dtype")
        )
        .def("device", &Tensor::device, py::return_value_policy::reference, 
            "Set the device that the tensor lives on",
            py::arg("new_device")
        )
        .def("requires_grad", &Tensor::requiresGrad, py::return_value_policy::reference,
            "Set Whether or not this tensor requires gradient to be calculated",
            py::arg("new_value")
        )
        .def("has_batch_dim", &Tensor::getHasBatchDim,
            "Check Whether or not the first dimension should be interpreted as a batch dim for this tensor"
        )
        .def("has_batch_dim", &Tensor::hasBatchDim, py::return_value_policy::reference,
            "Set Whether or not the first dimension should be interpreted as a batch dim for this tensor",
            py::arg("new_value")
        )

        // utilities
        .def("to_string", &Tensor::toString, 
            "get a summary of this tensor as a string"
        )
        .def("add_batch_dim", &Tensor::addBatchDim, py::return_value_policy::reference,
            "Add a batch dimension to the start of this tensor if it doesn't have one already"
        )

        // setters
        .def("set_value", py::overload_cast<const Tensor &, const Tensor &>(&Tensor::setValue),
            "Set a value at a specific index of this tensor",
            py::arg("indices"), py::arg("value")
        )
        .def("set_value",
            py::overload_cast<const std::vector<std::variant<int, std::string>> &, const Tensor &>(&Tensor::setValue),
            "Set a value at a specific index of this tensor",
            py::arg("indices"), py::arg("value")
        )
        .def("set_value", py::overload_cast<const std::vector<int> &, float>(&Tensor::setValue),
            "Set a value at a specific index of this tensor",
            py::arg("indices"), py::arg("value")
        )
        .def("set_value", py::overload_cast<const std::vector<int> &, std::complex<float>>(&Tensor::setValue),
            "Set a value at a specific index of this tensor",
            py::arg("indices"), py::arg("value")
        )

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
        .def(-py::self)


#if USE_PYTORCH
        .def("torch_tensor", &Tensor::getTensor, py::return_value_policy::reference,
            "Get the pytorch tensor that lives inside this tensor. Only available if using the pytorch backend..."
        )

        .def_static("from_torch_tensor", Tensor::fromTorchTensor,
            "construct a nuTens Tensor from a pytorch tensor"
        )
#endif
        
        // end of Tensor non-static functions
        
        // Tensor creation functions
        .def_static("eye", &Tensor::eye, 
            "Create a tensor initialised with an identity matrix",
            py::arg("n"), py::arg("dtype") = NTdtypes::kFloat, py::arg("device") = NTdtypes::kCPU, py::arg("requires_grad") = true)
        .def_static("rand", &Tensor::rand, 
            "Create a tensor initialised with random values",
            py::arg("shape"), py::arg("dtype") = NTdtypes::kFloat, py::arg("device") = NTdtypes::kCPU, py::arg("requires_grad") = true)
        .def_static("diag", &Tensor::diag, 
            "Create a tensor with specified values along the diagonal",
            py::arg("diagonal"))
        .def_static("ones", &Tensor::ones, 
            "Create a tensor initialised with ones",
            py::arg("shape"), py::arg("dtype") = NTdtypes::kFloat, py::arg("device") = NTdtypes::kCPU, py::arg("requires_grad") = true)
        .def_static("zeros", &Tensor::zeros, 
            "Create a tensor initialised with zeros",
            py::arg("shape"), py::arg("dtype") = NTdtypes::kFloat, py::arg("device") = NTdtypes::kCPU, py::arg("requires_grad") = true)

        .doc() = 
            "Tensor defines a basic interface for creating and manipulating tensors."
            "To create tensors you should use the static constructor methods.\n"
            "Alternatively you can chain together multiple property setters.\n"
            "\n"
            "For example\n"
            "\n"
            ".. code-block::\n"
            "\n"    
            "    from nuTens.tensor import Tensor, dtype\n"  
            "    tensor = Tensor.ones([3,3], dtype.scalar_type.float, dtype.device_type.cpu)\n"
            "\n"
            "will get you a 3x3 tensor of floats that lives on the CPU.\n"
            "\n"
            "This is equivalent to\n"
            "\n"
            ".. code-block::"
            "\n"
            "    tensor = Tensor.ones([3,3]).dtype(dtype.scalar_type.float).device(dtype.device_type.cpu);\n"
            "\n"
    ;

    // maffs
    m_tensor.def("matmul", &Tensor::matmul, 
        "Matrix multiplication",
        py::arg("t1"), py::arg("t2")
    );
    m_tensor.def("outer", &Tensor::outer, 
        "Tensor outer product",
        py::arg("t1"), py::arg("t2")
    );
    m_tensor.def("mul", &Tensor::mul, 
        "Element-wise multiplication",
        py::arg("t1"), py::arg("t2")
    );
    m_tensor.def("div", &Tensor::div, 
        "Element-wise division",
        py::arg("t1"), py::arg("t2")
    );
    m_tensor.def("pow", py::overload_cast<const Tensor &, float>(&Tensor::pow), 
        "Raise to scalar power",
        py::arg("t1"), py::arg("power")
    );
    m_tensor.def("pow", py::overload_cast<const Tensor &, std::complex<float>>(&Tensor::pow), 
        "Raise to scalar power",
        py::arg("t1"), py::arg("power")
    );
    m_tensor.def("exp", &Tensor::exp, 
        "Take element-wise exponential of a tensor",
        py::arg("t1")
    );
    m_tensor.def("transpose", &Tensor::transpose, 
        "Get the matrix transpose",
        py::arg("t1"), py::arg("index_1"), py::arg("index_2")
    );
    m_tensor.def("scale", py::overload_cast<const Tensor &, float>(&Tensor::scale), 
        "Scalar multiplication",
        py::arg("t1"), py::arg("scalar")
    );
    m_tensor.def("scale", py::overload_cast<const Tensor &, std::complex<float>>(&Tensor::scale),
        "Scalar multiplication",
        py::arg("t1"), py::arg("scalar")
    );
    m_tensor.def("sin", &Tensor::sin, 
        "Element-wise trigonometric sine function",
        py::arg("t1")
    );
    m_tensor.def("cos", &Tensor::cos, 
        "Element-wise trigonometric cosine function",
        py::arg("t1")
    );
    m_tensor.def("sum", py::overload_cast<const Tensor &>(&Tensor::sum), 
        "Get the sum of all values in a tensor",
        py::arg("t1")
    );
    m_tensor.def("sum", py::overload_cast<const Tensor &, const std::vector<long int> &>(&Tensor::sum),
        "Get the sum over particular dimensions",
        py::arg("t1"), py::arg("dimensions")
    );
    m_tensor.def("cumsum", py::overload_cast<const Tensor &, int>(&Tensor::cumsum),
        "Get the cumulative sum over particular dimensions",
        py::arg("t1"), py::arg("dimensions")
    );
    // m_tensor.def("eig", &Tensor::eig. "calculate eigenvalues") <- Will need to define some additional fn to return
    // tuple of values
}

void initPropagator(py::module &m)
{
     auto m_propagator = m.def_submodule("propagator");

    py::class_<Propagator>(m_propagator, "Propagator")
        .def(py::init<int, float>())
        .def("calculate_probabilities", &Propagator::calculateProbs,
            "Calculate the oscillation probabilities for neutrinos of specified energies"
        )
        .def("set_matter_solver", &Propagator::setMatterSolver,
            "Set the matter effect solver that the propagator should use",
            py::arg("new_matter_solver")
        )
        .def("set_masses", &Propagator::setMasses, 
            "Set the neutrino mass state eigenvalues",
            py::arg("new_masses")
        )
        .def("set_energies", py::overload_cast<Tensor &>(&Propagator::setEnergies),
            "Set the neutrino energies that the propagator should use",
            py::arg("new_energies")
        )
        .def("set_PMNS", py::overload_cast<Tensor &>(&Propagator::setPMNS),
            "Set the PMNS matrix that the propagator should use",
            py::arg("new_matrix")
        )
        .def("set_PMNS", py::overload_cast<const std::vector<int> &, float>(&Propagator::setPMNS),
            "Set a particular value within the PMNS matrix used by the propagator",
            py::arg("indices"), py::arg("value")
        )
        .def("set_PMNS", py::overload_cast<const std::vector<int> &, std::complex<float>>(&Propagator::setPMNS),
            "Set the PMNS matrix that the propagator should use",
            py::arg("indices"), py::arg("value")
        );

    py::class_<BaseMatterSolver, std::shared_ptr<BaseMatterSolver>>(m_propagator, "BaseMatterSolver")
        .def("set_PMNS", &BaseMatterSolver::setPMNS,
            "Set the PMNS matrix that the solver should use",
            py::arg("new_matrix")
        )
        .def("set_energies", &BaseMatterSolver::setEnergies,
            "Set the neutrino energies",
            py::arg("new_energies")
        )
        .def("set_masses", &BaseMatterSolver::setMasses,
            "Set the neutrino masses the solver should use",
            py::arg("new_masses")
        )
        .def("calculate_eigenvalues", &BaseMatterSolver::calculateEigenvalues,
            "calculate the eigenvalues of the Hamiltonian",
            py::arg("eigenvector_out"), py::arg("eigenvalue_out")
        )
        ;

     py::class_<ConstDensityMatterSolver, std::shared_ptr<ConstDensityMatterSolver>, BaseMatterSolver>(
          m_propagator, "ConstDensitySolver")
          .def(py::init<int, float>());
}

void initDtypes(py::module &m)
{
    auto m_dtypes = m.def_submodule("dtype",
        "This module defines various data types used in nuTens");

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

void initUnits(py::module &m)
{
    auto m_units = m.def_submodule("units",
        "Defines some helpful units, which are really just conversion factors to eV");

    m_units.attr("eV")  = py::float_(Units::eV);
    m_units.attr("MeV") = py::float_(Units::MeV);
    m_units.attr("GeV") = py::float_(Units::GeV);

    m_units.attr("cm") = py::float_(Units::cm);
    m_units.attr("m")  = py::float_(Units::m);
    m_units.attr("km") = py::float_(Units::km);
    
}

void initTesting(py::module &m)
{
    auto m_testing = m.def_submodule("testing",
        "Some helpful utilities to use when writing python tests for your code");

    py::class_<Testing::TwoFlavourBarger>(m_testing, "TwoFlavourBarger")
        .def(py::init<>())
        .def("set_params", &Testing::TwoFlavourBarger::setParams, 
            py::arg("m1"), py::arg("m2"), py::arg("theta"), py::arg("baseline"), py::arg("density") = (float)-999.9
        )
        .def("lv", &Testing::TwoFlavourBarger::lv,
            "Calculates the vacuum oscillation length",
            py::arg("energy")
        )
        .def("lm", &Testing::TwoFlavourBarger::lm,
            "Calculates the matter oscillation length"
        )
        .def("calculate_effective_angle", &Testing::TwoFlavourBarger::calculateEffectiveAngle,
            "Calculates the effective mixing angle, alpha, in matter",
            py::arg("energy")
        )
        .def("calculate_effective_dm2", &Testing::TwoFlavourBarger::calculateEffectiveDm2,
            "Calculates the effective delta m^2 in matter",
            py::arg("energy")
        )
        .def("get_PMNS_element", &Testing::TwoFlavourBarger::getPMNSelement,
            "Calculates the effective i,j-th element of the mizing matrix for a given energy",
            py::arg("energy"), py::arg("i"), py::arg("j")
        )
        .def("calculate_prob", &Testing::TwoFlavourBarger::calculateProb,
            "Calculate probability of transitioning from state i to state j for a given energy",
            py::arg("energy"), py::arg("i"), py::arg("j")
        )
    ;
}