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
#include <tests/nuFast.hpp>
#include <nuTens/propagator/base-mixing-matrix.hpp>
#include <nuTens/propagator/DP-propagator.hpp>
#include <nuTens/propagator/pmns-matrix.hpp>

#if USE_PYTORCH
#include <torch/torch.h>
#include <torch/extension.h>
#endif

namespace py = pybind11;

using namespace nuTens;

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
        .def(py::init<std::vector<float>, dtypes::scalarType, dtypes::deviceType, bool>())

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
        .def("unsqueeze", &Tensor::unsqueeze, py::return_value_policy::reference,
            "add an extra dimension to this tensor at the specified location",
            py::arg("dim")
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
        .def("set_value", py::overload_cast<const std::vector<int> &, double>(&Tensor::setValue),
            "Set a value at a specific index of this tensor",
            py::arg("indices"), py::arg("value")
        )
        .def("set_value", py::overload_cast<const std::vector<int> &, std::complex<float>>(&Tensor::setValue),
            "Set a value at a specific index of this tensor",
            py::arg("indices"), py::arg("value")
        )
        .def("set_value", py::overload_cast<const std::vector<int> &, std::complex<double>>(&Tensor::setValue),
            "Set a value at a specific index of this tensor",
            py::arg("indices"), py::arg("value"))

        // getters
        .def("get_shape", &Tensor::getShape, "Get the shape of this tensor")
        .def("get_values", &Tensor::getValues, py::arg("indices"), "Get the subset of values in this tensor at a specified location")
        .def("get_value", &Tensor::getVariantValue, py::arg("indices"), "Get the data stored at a particular index of the tensor")

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
            py::arg("n"), py::arg("dtype") = dtypes::kFloat, py::arg("device") = dtypes::kCPU, py::arg("requires_grad") = true)
        .def_static("rand", &Tensor::rand, 
            "Create a tensor initialised with random values",
            py::arg("shape"), py::arg("dtype") = dtypes::kFloat, py::arg("device") = dtypes::kCPU, py::arg("requires_grad") = true)
        .def_static("diag", &Tensor::diag, 
            "Create a tensor with specified values along the diagonal",
            py::arg("diagonal"))
        .def_static("ones", &Tensor::ones, 
            "Create a tensor initialised with ones",
            py::arg("shape"), py::arg("dtype") = dtypes::kFloat, py::arg("device") = dtypes::kCPU, py::arg("requires_grad") = true)
        .def_static("zeros", &Tensor::zeros, 
            "Create a tensor initialised with zeros",
            py::arg("shape"), py::arg("dtype") = dtypes::kFloat, py::arg("device") = dtypes::kCPU, py::arg("requires_grad") = true)

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
        .def(py::init<int, float, bool>(), 
            py::arg("n_generations"), py::arg("baseline"), py::arg("anti_neutrino")=false)
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
        .def("set_mixing_matrix", py::overload_cast<Tensor &>(&Propagator::setMixingMatrix),
            "Set the mixing matrix that the propagator should use",
            py::arg("new_matrix")
        )
        .def("set_mixing_matrix", py::overload_cast<const std::vector<int> &, float>(&Propagator::setMixingMatrix),
            "Set a particular value within the mixing matrix used by the propagator",
            py::arg("indices"), py::arg("value")
        )
        .def("set_mixing_matrix", py::overload_cast<const std::vector<int> &, std::complex<float>>(&Propagator::setMixingMatrix),
            "Set the mixing matrix that the propagator should use",
            py::arg("indices"), py::arg("value")
        )
        .def("set_baseline", (&Propagator::setBaseline),
            "Set the baseline that the propagator should use",
            py::arg("new_value")
        )
        .def("get_baseline", (&Propagator::getBaseline),
            "Get the baseline used by the propagator"
        )
        .def("set_antineutrino", (&Propagator::setAntiNeutrino),
            "Set whether the propagator should calculate oscillations for anti-neutrinos",
            py::arg("new_value")
        )
        ;


    py::class_<DPpropagator, Propagator>(m_propagator, "DPpropagator")
        .def(py::init<float, bool, float, int>(), 
            py::arg("baseline"), py::arg("anti_neutrino")=false, py::arg("density"), py::arg("NR_iterations"))
        .def("set_parameters", &DPpropagator::setParameters,
            "set the parameters for the oscillation calculations",
            py::arg("new_theta12"), py::arg("new_theta23"), py::arg("new_theta13"), py::arg("new_deltaCP"), py::arg("new_deltamsq21"), py::arg("new_deltamsq31")
        )
        .def("set_energies", &DPpropagator::setEnergies,
            "set the neutrino energies",
            py::arg("new_energies")
        )
        .def("calculate_probs", &DPpropagator::calculateProbs
        )
        .def("get_theta12", &DPpropagator::getTheta12)
        .def("get_theta23", &DPpropagator::getTheta23)
        .def("get_theta13", &DPpropagator::getTheta13)
        .def("get_deltacp", &DPpropagator::getDeltaCP)
        .def("get_deltamsq21", &DPpropagator::getDmsp21)
        .def("get_deltamsq31", &DPpropagator::getDmsq31)
        .def("get_energies", &DPpropagator::getEnergies)
        ;

    py::class_<BaseMatterSolver, std::shared_ptr<BaseMatterSolver>>(m_propagator, "BaseMatterSolver")
        .def("set_mixing_matrix", &BaseMatterSolver::setMixingMatrix,
            "Set the mixing matrix that the solver should use",
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
        .def("set_antineutrino", (&BaseMatterSolver::setAntiNeutrino),
            "Set whether the solver should calculate values for anti-neutrinos",
            py::arg("new_value")
        )
        ;

     py::class_<ConstDensityMatterSolver, std::shared_ptr<ConstDensityMatterSolver>, BaseMatterSolver>(
        m_propagator, "ConstDensitySolver")
        .def(py::init<int, float, bool>(), 
            py::arg("n_generations"), py::arg("density"), py::arg("anti_neutrino")=false)
        .def("set_density", (&ConstDensityMatterSolver::setDensity),
            "Set the density that the solver should use",
            py::arg("new_value")
        )
        .def("get_density", (&ConstDensityMatterSolver::getDensity),
            "Get the density used by the solver"
        )
        ;


    py::class_<BaseMixingMatrix, std::shared_ptr<BaseMixingMatrix>>(m_propagator, "BaseMixingMatrix")
        .def("build", (&BaseMixingMatrix::build))
        ;

     py::class_<PMNSmatrix, std::shared_ptr<PMNSmatrix>, BaseMixingMatrix>(
        m_propagator, "PMNSmatrix")
        .def(py::init<>())
        .def("set_parameter_values", (&PMNSmatrix::setParameterValues),
            py::arg("theta_12"), py::arg("theta_13"), py::arg("theta_23"), py::arg("delta_cp"))
        .def("get_theta_12_tensor", (&PMNSmatrix::getTheta12Tensor), py::return_value_policy::reference)
        .def("get_theta_13_tensor", (&PMNSmatrix::getTheta13Tensor), py::return_value_policy::reference)
        .def("get_theta_23_tensor", (&PMNSmatrix::getTheta23Tensor), py::return_value_policy::reference)
        .def("get_delta_cp_tensor", (&PMNSmatrix::getDeltaCPTensor), py::return_value_policy::reference)
        ;

}

void initDtypes(py::module &m)
{
    auto m_dtypes = m.def_submodule("dtype",
        "This module defines various data types used in nuTens");

    py::enum_<dtypes::scalarType>(m_dtypes, "scalar_type")
        .value("int", dtypes::scalarType::kInt)
        .value("float", dtypes::scalarType::kFloat)
        .value("double", dtypes::scalarType::kDouble)
        .value("complex_float", dtypes::scalarType::kComplexFloat)
        .value("complex_double", dtypes::scalarType::kComplexDouble)
    ;

    py::enum_<dtypes::deviceType>(m_dtypes, "device_type")
        .value("cpu", dtypes::deviceType::kCPU)
        .value("gpu", dtypes::deviceType::kGPU)
    ;
}

void initUnits(py::module &m)
{
    auto m_units = m.def_submodule("units",
        "Defines some helpful units, which are really just conversion factors to eV");

    m_units.attr("eV")  = py::float_(units::eV);
    m_units.attr("MeV") = py::float_(units::MeV);
    m_units.attr("GeV") = py::float_(units::GeV);

    m_units.attr("cm") = py::float_(units::cm);
    m_units.attr("m")  = py::float_(units::m);
    m_units.attr("km") = py::float_(units::km);
    
}

void initTesting(py::module &m)
{
    auto m_testing = m.def_submodule("testing",
        "Some helpful utilities to use when writing python tests for your code"
    )
    .def("nufast_probability_matter", [](double s12sq, double s13sq, double s23sq, double delta, double dm21, double dm31, double L, double E, double rho, double Ye, double Nnewton) 
        {
            // the probabilities as a raw c array
            double probs_returned[3][3];

            // get the probabilities
            Probability_Matter_LBL(s12sq, s13sq, s23sq, delta, dm21, dm31, L, E, rho, Ye, Nnewton, &probs_returned);

            // turn them into a vector so they can be returned as a numpy array
            std::vector<std::vector<double>> ret = {
                {probs_returned[0][0], probs_returned[0][1], probs_returned[0][2]},
                {probs_returned[1][0], probs_returned[1][1], probs_returned[1][2]},
                {probs_returned[2][0], probs_returned[2][1], probs_returned[2][2]}
            };

            return ret;
        },
        "Calculates the oscillation probabilities using nufast",
        py::arg("sin_squared_theta12"), py::arg("sin_squared_theta13"), py::arg("sin_squared_theta23"), 
        py::arg("delta_cp"), py::arg("delta_m_squared_21"), py::arg("delta_m_squared_31"), 
        py::arg("baseline"), py::arg("energy"), py::arg("rho"), py::arg("Ye"), py::arg("N_Newton")
    )
    ;

    py::class_<testing::TwoFlavourBarger>(m_testing, "TwoFlavourBarger")
        .def(py::init<>())
        .def("set_params", &testing::TwoFlavourBarger::setParams, 
            py::arg("m1"), py::arg("m2"), py::arg("theta"), py::arg("baseline"), py::arg("density") = (float)-999.9, py::arg("anti_neutrino") = false
        )
        .def("lv", &testing::TwoFlavourBarger::lv,
            "Calculates the vacuum oscillation length",
            py::arg("energy")
        )
        .def("lm", &testing::TwoFlavourBarger::lm,
            "Calculates the matter oscillation length"
        )
        .def("calculate_effective_angle", &testing::TwoFlavourBarger::calculateEffectiveAngle,
            "Calculates the effective mixing angle, alpha, in matter",
            py::arg("energy")
        )
        .def("calculate_effective_dm2", &testing::TwoFlavourBarger::calculateEffectiveDm2,
            "Calculates the effective delta m^2 in matter",
            py::arg("energy")
        )
        .def("get_PMNS_element", &testing::TwoFlavourBarger::getPMNSelement,
            "Calculates the effective i,j-th element of the mizing matrix for a given energy",
            py::arg("energy"), py::arg("i"), py::arg("j")
        )
        .def("calculate_prob", &testing::TwoFlavourBarger::calculateProb,
            "Calculate probability of transitioning from state i to state j for a given energy",
            py::arg("energy"), py::arg("i"), py::arg("j")
        )
    ;

    py::class_<testing::ThreeFlavourBarger>(m_testing, "ThreeFlavourBarger")
        .def(py::init<>())
        .def("set_params", &testing::ThreeFlavourBarger::setParams, 
            py::arg("m1"), py::arg("m2"), py::arg("m3"), py::arg("theta12"), py::arg("theta13"), py::arg("theta23"), py::arg("deltaCP"), py::arg("baseline"), py::arg("density") = (float)-999.9, py::arg("anti_neutrino") = false
        )
        .def("alpha", &testing::ThreeFlavourBarger::alpha,
            "Calculates alpha term used in calculating the mass eigenvalues",
            py::arg("energy")
        )
        .def("beta", &testing::ThreeFlavourBarger::beta,
            "Calculates beta term used in calculating the mass eigenvalues",
            py::arg("energy")
        )
        .def("gamma", &testing::ThreeFlavourBarger::gamma,
            "Calculates gamma term used in calculating the mass eigenvalues",
            py::arg("energy")
        )
        .def("calculate_effective_m2", &testing::ThreeFlavourBarger::calculateEffectiveM2,
            "Calculates the effective hamiltonian eigenvalues (the M^2) in matter",
            py::arg("energy"), py::arg("index")
        )
        .def("get_hamiltonian_element", &testing::ThreeFlavourBarger::getHamiltonianElement,
            "Calculates an element of the Hamiltonian",
            py::arg("energy"), py::arg("a"), py::arg("b")
        )
        .def("get_transition_matrix_element", &testing::ThreeFlavourBarger::getTransitionMatrixElement,
            "Calculates an element of the transition matrix from one mass eigenstate to another due to the presense of matter",
            py::arg("energy"), py::arg("a"), py::arg("b")
        )
        .def("calculate_prob", &testing::ThreeFlavourBarger::calculateProb,
            "Calculate probability of transitioning from state i to state j for a given energy",
            py::arg("energy"), py::arg("i"), py::arg("j")
        )
    ;
}