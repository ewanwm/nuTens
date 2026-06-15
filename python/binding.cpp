// pybind11 stuff
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/native_enum.h>

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

void initDtypes(py::module & /*m_nuTens*/);
void initTensor(py::module & /*m_nuTens*/);
void initPropagator(py::module & /*m_nuTens*/);
void initUnits(py::module & /*m_nuTens*/);
void initTesting(py::module & /*m_nuTens*/);

// initialise the top level module "_pyNuTens"
// NOLINTNEXTLINE
PYBIND11_MODULE(_pyNuTens, m_nuTens)
{
    m_nuTens.doc() = "Library to calculate neutrino oscillations";
    initDtypes(m_nuTens);
    initUnits(m_nuTens);
    initTensor(m_nuTens);
    initPropagator(m_nuTens);
    initTesting(m_nuTens);

#ifdef VERSION_INFO
     m_nuTens.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
     m_nuTens.attr("__version__") = "dev";
#endif
}

// helper function to convert a nuTens tensor to a numpy array
py::buffer_info tensorToNumpy(const Tensor &tensor){

     size_t size = 0;
     std::string format = "";

     switch (tensor.getDType())
     {
     case dtypes::kFloat:
         size = sizeof(float);
         format = pybind11::format_descriptor<float>::format();
         break;

     case dtypes::kComplexFloat:
         size = sizeof(std::complex<float>);
         format = pybind11::format_descriptor<std::complex<float>>::format();
         break;

     default:
         NT_ERROR("Invalid dtype has been set for this tensor: {}", tensor.getDType());
         NT_ERROR("{}:{}", __FILE__, __LINE__);
         throw;
    }

// backend specific stuff for extracting data and layout
#if USE_PYTORCH
    at::Tensor torchTensor = tensor.getTensor().contiguous();
    void *dataPtr = torchTensor.data_ptr();

    // linter seems to struggle with recogising this type and thinks it is an int
    // and always thinks it is uninitialised
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<long int> strides = torchTensor.strides().vec();

#else

#   error Only pytorch supported right now :(

#endif

    // linter seems to struggle with recogising this type and thinks it is an int
    // and always thinks it is uninitialised
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<int> stridesBytes = {};

    // convert strides into bytes
    for(const long int &stride : strides) {
        stridesBytes.push_back(stride * size);
    }

    return py::buffer_info(
        dataPtr,     /* Pointer to buffer */
        size,                       /* Size of one scalar */
        format.c_str(),             /* Python struct-style format descriptor */
        torchTensor.dim(),          /* Number of dimensions */
        torchTensor.sizes().vec(),  /* Buffer dimensions */
        stridesBytes                /* Strides (in bytes) for each index */
    );
}

void initTensor(py::module &m_nuTens)
{
    auto m_tensor = m_nuTens.def_submodule("tensor");

    py::class_<Tensor>(m_tensor, "Tensor", py::buffer_protocol())
        .def(py::init()) // <- default constructor
        .def(py::init<std::vector<float>, dtypes::scalarType, dtypes::deviceType, bool>(),
            py::arg("values"), py::arg("dtype") = dtypes::scalarType::kFloat, py::arg("device") = dtypes::kCPU, py::arg("requires_grad") = true)

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
        .def("get_dtype", &Tensor::getDType, "Get the type of the data contained within this tensor")
        .def("get_device", &Tensor::getDevice, "Get the device that this tensor lives on")
        .def("get_requires_grad", &Tensor::getRequiresGrad, "Get whether or not this tensor will collect gradients")

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
        .def(py::self + py::self)
        .def(py::self + float())
        .def(float() + py::self)
        .def(float() * py::self)
        .def(py::self * float())
        .def(py::self / float())
        .def(-py::self)
        .def("__repr__", &Tensor::toString)


#if USE_PYTORCH
        .def("torch_tensor", &Tensor::getTensor, py::return_value_policy::reference,
            "Get the pytorch tensor that lives inside this tensor. Only available if using the pytorch backend..."
        )

        .def_static("from_torch_tensor", Tensor::fromTorchTensor,
            "construct a nuTens Tensor from a pytorch tensor"
        )

        // construct from a numpy array
        .def(
            py::init(
                [](py::array_t<float> buffer, bool requiresGrad){

                    /* Request a buffer descriptor from Python */
                    py::buffer_info info = buffer.request();

                    return Tensor::fromTorchTensor(torch::from_blob(info.ptr, info.shape)).requiresGrad(requiresGrad);
                }
            ),
            "Construct a tensor from a numpy array",
            py::arg("array"), py::arg("requires_grad") = true
        ) 
        .def(
            py::init(
                [](py::array_t<std::complex<float>> buffer, bool requiresGrad){

                    /* Request a buffer descriptor from Python */
                    py::buffer_info info = buffer.request();

                    auto options = torch::TensorOptions()
                        .dtype(torch::kComplexFloat)
                        .requires_grad(requiresGrad);

                    return Tensor::fromTorchTensor(torch::from_blob(info.ptr, info.shape, options));
                }
            ),
            "Construct a tensor from an \"array like\" object",
            py::arg("array_like"), py::arg("requires_grad") = true

        ) 
#endif
        // get a numpy array of tensor contents
        .def("numpy",
            [](Tensor &tensor) -> py::array {
                return py::array(tensorToNumpy(tensor));
            },
            "Get a numpy array with the contents of the tensor"
        )
        
        // return a python buffer interface object
        .def_buffer(
            [](Tensor &tensor) -> py::buffer_info {
                return tensorToNumpy(tensor);
            }
        )

        // end of Tensor non-static functions
        
        // Tensor creation functions
        .def_static("eye", &Tensor::eye, 
            "Create a tensor initialised with an identity matrix",
            py::arg("n"), py::arg("dtype") = dtypes::scalarType::kFloat, py::arg("device") = dtypes::deviceType::kCPU, py::arg("requires_grad") = true)
        .def_static("rand", &Tensor::rand, 
            "Create a tensor initialised with random values",
            py::arg("shape"), py::arg("dtype") = dtypes::scalarType::kFloat, py::arg("device") = dtypes::kCPU, py::arg("requires_grad") = true)
        .def_static("diag", &Tensor::diag, 
            "Create a tensor with specified values along the diagonal",
            py::arg("diagonal"))
        .def_static("ones", &Tensor::ones, 
            "Create a tensor initialised with ones",
            py::arg("shape"), py::arg("dtype") = dtypes::scalarType::kFloat, py::arg("device") = dtypes::kCPU, py::arg("requires_grad") = true)
        .def_static("zeros", &Tensor::zeros, 
            "Create a tensor initialised with zeros",
            py::arg("shape"), py::arg("dtype") = dtypes::scalarType::kFloat, py::arg("device") = dtypes::kCPU, py::arg("requires_grad") = true)

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
        py::arg("tensor_1"), py::arg("tensor_2")
    );
    m_tensor.def("outer", &Tensor::outer, 
        "Tensor outer product",
        py::arg("tensor_1"), py::arg("tensor_2")
    );
    m_tensor.def("mul", &Tensor::mul, 
        "Element-wise multiplication",
        py::arg("tensor_1"), py::arg("tensor_2")
    );
    m_tensor.def("add", &Tensor::add, 
        "Element-wise addition",
        py::arg("tensor_1"), py::arg("tensor_2")
    );
    m_tensor.def("div", &Tensor::div, 
        "Element-wise division",
        py::arg("tensor_1"), py::arg("tensor_2")
    );
    m_tensor.def("pow", py::overload_cast<const Tensor &, float>(&Tensor::pow), 
        "Raise to scalar power",
        py::arg("tensor_1"), py::arg("power")
    );
    m_tensor.def("pow", py::overload_cast<const Tensor &, std::complex<float>>(&Tensor::pow), 
        "Raise to scalar power",
        py::arg("tensor_1"), py::arg("power")
    );
    m_tensor.def("exp", &Tensor::exp, 
        "Take element-wise exponential of a tensor",
        py::arg("tensor_1")
    );
    m_tensor.def("transpose", &Tensor::transpose, 
        "Get the matrix transpose",
        py::arg("tensor_1"), py::arg("index_1"), py::arg("index_2")
    );
    m_tensor.def("scale", py::overload_cast<const Tensor &, float>(&Tensor::scale), 
        "Scalar multiplication",
        py::arg("tensor_1"), py::arg("scalar")
    );
    m_tensor.def("scale", py::overload_cast<const Tensor &, std::complex<float>>(&Tensor::scale),
        "Scalar multiplication",
        py::arg("tensor_1"), py::arg("scalar")
    );
    m_tensor.def("sin", &Tensor::sin, 
        "Element-wise trigonometric sine function",
        py::arg("tensor_1")
    );
    m_tensor.def("cos", &Tensor::cos, 
        "Element-wise trigonometric cosine function",
        py::arg("tensor_1")
    );
    m_tensor.def("sum", py::overload_cast<const Tensor &>(&Tensor::sum), 
        "Get the sum of all values in a tensor",
        py::arg("tensor_1")
    );
    m_tensor.def("sum", py::overload_cast<const Tensor &, const std::vector<long int> &>(&Tensor::sum),
        "Get the sum over particular dimensions",
        py::arg("tensor_1"), py::arg("dimensions")
    );
    m_tensor.def("cumsum", py::overload_cast<const Tensor &, int>(&Tensor::cumsum),
        "Get the cumulative sum over particular dimensions",
        py::arg("tensor_1"), py::arg("dimensions")
    );
    // m_tensor.def("eig", &Tensor::eig. "calculate eigenvalues") <- Will need to define some additional fn to return
    // tuple of values
}

void initPropagator(py::module &m_nuTens)
{
    auto m_propagator = m_nuTens.def_submodule("propagator");

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
        .def(py::init<int>(), 
            py::arg("NR_iterations"))
        .def("set_parameters", &DPpropagator::setParameters,
            "set the parameters for the oscillation calculations",
            py::arg("new_theta12"), py::arg("new_theta23"), py::arg("new_theta13"), py::arg("new_deltaCP"), py::arg("new_deltamsq21"), py::arg("new_deltamsq31"), py::arg("sin_squared_thetas") = false
        )
        .def("set_antineutrino", &DPpropagator::setAntiNeutrino,
            "set whether to calculate anti-neutrino probabilities",
            py::arg("new_value")
        )
        .def("set_baseline", &DPpropagator::setBaseline,
            "set the baseline",
            py::arg("new_baseline")
        )
        .def("set_density", &DPpropagator::setDensity,
            "set the density",
            py::arg("new_density")
        )
        .def("set_energies", &DPpropagator::setEnergies,
            "set the neutrino energies",
            py::arg("new_energies")
        )
        .def("set_sin_squared_thetas", &DPpropagator::setSinSquaredThetas,
            "If `True`, the provided theta_ij values will be interpreted as sin^2(theta_ij) meaning that some of the computation can be shortcut and the probability calculation will be sped up. Note however that this will force the thetas to be in the lower octant (which is probably fine for most applications)",
            py::arg("new_value")
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

     py::class_<ConstDensityMatterSolver, std::shared_ptr<ConstDensityMatterSolver>, BaseMatterSolver>(
        m_propagator, "ConstDensitySolver")
        .def(py::init<int>(), 
            py::arg("n_generations"))
        .def("set_density", (&ConstDensityMatterSolver::setDensity),
            "Set the density that the solver should use",
            py::arg("new_value")
        )
        .def("set_antineutrino", (&ConstDensityMatterSolver::setAntiNeutrino),
            "Set the density that the solver should use",
            py::arg("new_value")
        )
        .def("set_mixing_matrix", (&ConstDensityMatterSolver::setMixingMatrix),
            "Set the mixing that the solver should use",
            py::arg("new_value")
        )
        .def("set_masses", (&ConstDensityMatterSolver::setMasses),
            "Set the neutrino masses that the solver should use",
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

void initDtypes(py::module &m_nuTens)
{
    auto m_dtypes = m_nuTens.def_submodule("dtype",
        "This module defines various data types used in nuTens");

    py::native_enum<dtypes::scalarType>(m_dtypes, "scalar_type", "enum.Enum")
        .value("float", dtypes::scalarType::kFloat)
        .value("double", dtypes::scalarType::kDouble)
        .value("complex_float", dtypes::scalarType::kComplexFloat)
        .value("complex_double", dtypes::scalarType::kComplexDouble)
        .finalize()
    ;

    py::native_enum<dtypes::deviceType>(m_dtypes, "device_type", "enum.Enum")
        .value("cpu", dtypes::deviceType::kCPU)
        .value("gpu", dtypes::deviceType::kGPU)
        .finalize()
    ;
}

void initUnits(py::module &m_nuTens)
{
    auto m_units = m_nuTens.def_submodule("units",
        "Defines some helpful units, which are really just conversion factors to eV");

    m_units.attr("eV")  = py::float_(units::eV);
    m_units.attr("MeV") = py::float_(units::MeV);
    m_units.attr("GeV") = py::float_(units::GeV);

    m_units.attr("cm") = py::float_(units::cm);
    m_units.attr("m")  = py::float_(units::m);
    m_units.attr("km") = py::float_(units::km);
    
}

void initTesting(py::module &m_nuTens)
{
    auto m_testing = m_nuTens.def_submodule("testing",
        "Some helpful utilities to use when writing python tests for your code"
    )
    .def("nufast_probability_matter", [](double s12sq, double s13sq, double s23sq, double delta, double dm21, double dm31, double baseline, double energy, double rho, double electronDensity, int Nnewton) 
        {
            // the probabilities as a raw c array
            double probs_returned[3][3]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

            // get the probabilities
            Probability_Matter_LBL(s12sq, s13sq, s23sq, delta, dm21, dm31, baseline, energy, rho, electronDensity, Nnewton, &probs_returned);

            // turn them into a vector so they can be returned as a numpy 
            
            // linter seems to struggle with recogising this type and thinks it is an int
            // and always thinks it is uninitialised
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            std::vector<std::vector<double>> ret {
                {
                    {probs_returned[0][0], probs_returned[0][1], probs_returned[0][2]},
                    {probs_returned[1][0], probs_returned[1][1], probs_returned[1][2]},
                    {probs_returned[2][0], probs_returned[2][1], probs_returned[2][2]}
                }
            };

            return ret;
        },
        "Calculates the oscillation probabilities using nufast",
        py::arg("sin_squared_theta12"), py::arg("sin_squared_theta13"), py::arg("sin_squared_theta23"), 
        py::arg("delta_cp"), py::arg("delta_m_squared_21"), py::arg("delta_m_squared_31"), 
        py::arg("baseline"), py::arg("energy"), py::arg("rho"), py::arg("Ye"), py::arg("N_Newton")
    )
    ;

    py::class_<testing::TwoFlavourBarger<>>(m_testing, "TwoFlavourBarger")
        .def(py::init<>())
        .def("set_m1", &testing::TwoFlavourBarger<>::setMass1, 
            py::arg("m1")
        )
        .def("set_m2", &testing::TwoFlavourBarger<>::setMass2, 
            py::arg("m2")
        )
        .def("set_theta", &testing::TwoFlavourBarger<>::setTheta, 
            py::arg("theta")
        )
        .def("set_baseline", &testing::TwoFlavourBarger<>::setBaseline, 
            py::arg("baseline")
        )
        .def("set_density", &testing::TwoFlavourBarger<>::setDensity, 
            py::arg("density")
        )
        .def("set_antineutrino", &testing::TwoFlavourBarger<>::setAntiNeutrino, 
            py::arg("antineutrino")
        )
        .def("l_vac", &testing::TwoFlavourBarger<>::lVac,
            "Calculates the vacuum oscillation length",
            py::arg("energy")
        )
        .def("l_matter", &testing::TwoFlavourBarger<>::lMatter,
            "Calculates the matter oscillation length"
        )
        .def("calculate_effective_angle", &testing::TwoFlavourBarger<>::calculateEffectiveAngle,
            "Calculates the effective mixing angle, alpha, in matter",
            py::arg("energy")
        )
        .def("calculate_effective_dm2", &testing::TwoFlavourBarger<>::calculateEffectiveDm2,
            "Calculates the effective delta m_nuTens^2 in matter",
            py::arg("energy")
        )
        .def("get_PMNS_element", &testing::TwoFlavourBarger<>::getPMNSelement,
            "Calculates the effective i,j-th element of the mizing matrix for a given energy",
            py::arg("energy"), py::arg("i"), py::arg("j")
        )
        .def("calculate_prob", &testing::TwoFlavourBarger<>::calculateProb,
            "Calculate probability of transitioning from state i to state j for a given energy",
            py::arg("energy"), py::arg("i"), py::arg("j")
        )
    ;

    py::class_<testing::ThreeFlavourBarger<>>(m_testing, "ThreeFlavourBarger")
        .def(py::init<>())
        .def("set_m1", &testing::ThreeFlavourBarger<>::setMass1, 
            py::arg("m1")
        )
        .def("set_m2", &testing::ThreeFlavourBarger<>::setMass2, 
            py::arg("m2")
        )
        .def("set_m3", &testing::ThreeFlavourBarger<>::setMass3, 
            py::arg("m3")
        )
        .def("set_theta12", &testing::ThreeFlavourBarger<>::setTheta12, 
            py::arg("theta12")
        )
        .def("set_theta13", &testing::ThreeFlavourBarger<>::setTheta13, 
            py::arg("theta13")
        )
        .def("set_theta23", &testing::ThreeFlavourBarger<>::setTheta23, 
            py::arg("theta23")
        )
        .def("set_deltacp", &testing::ThreeFlavourBarger<>::setDeltaCP, 
            py::arg("deltacp")
        )
        .def("set_baseline", &testing::ThreeFlavourBarger<>::setBaseline, 
            py::arg("baseline")
        )
        .def("set_density", &testing::ThreeFlavourBarger<>::setDensity, 
            py::arg("density")
        )
        .def("set_antineutrino", &testing::ThreeFlavourBarger<>::setAntiNeutrino, 
            py::arg("antineutrino")
        )

        .def("alpha", &testing::ThreeFlavourBarger<>::calculateAlpha,
            "Calculates alpha term used in calculating the mass eigenvalues",
            py::arg("energy")
        )
        .def("beta", &testing::ThreeFlavourBarger<>::calculateBeta,
            "Calculates beta term used in calculating the mass eigenvalues",
            py::arg("energy")
        )
        .def("gamma", &testing::ThreeFlavourBarger<>::calculateGamma,
            "Calculates gamma term used in calculating the mass eigenvalues",
            py::arg("energy")
        )
        .def("calculate_effective_m2", &testing::ThreeFlavourBarger<>::calculateEffectiveM2,
            "Calculates the effective hamiltonian eigenvalues (the m_nuTens^2) in matter",
            py::arg("energy"), py::arg("index")
        )
        .def("get_hamiltonian_element", &testing::ThreeFlavourBarger<>::getHamiltonianElement,
            "Calculates an element of the Hamiltonian",
            py::arg("energy"), py::arg("a"), py::arg("b")
        )
        .def("get_transition_matrix_element", &testing::ThreeFlavourBarger<>::getTransitionMatrixElement,
            "Calculates an element of the transition matrix from one mass eigenstate to another due to the presense of matter",
            py::arg("energy"), py::arg("a"), py::arg("b")
        )
        .def("calculate_prob", &testing::ThreeFlavourBarger<>::calculateProb,
            "Calculate probability of transitioning from state i to state j for a given energy",
            py::arg("energy"), py::arg("i"), py::arg("j")
        )
    ;
}