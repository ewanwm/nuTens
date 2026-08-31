from __future__ import annotations
import nuTens.dtype
import nuTens.pyNuTens.dtype
import nuTens.pyNuTens.tensor
import typing
__all__: list[str] = ['BaseMatterSolver', 'BaseMixingMatrix', 'ConstDensitySolver', 'DPpropagator', 'PMNSmatrix', 'Propagator']
class BaseMatterSolver:
    def calculate_eigenvalues(self) -> list[nuTens.pyNuTens.tensor.Tensor]:
        """
        calculate the eigenvalues of the Hamiltonian. Returns tuple containing <eigenvectors, eigenvalues>
        """
    def set_antineutrino(self, new_value: bool) -> BaseMatterSolver:
        """
        Set whether the solver should calculate values for anti-neutrinos
        """
    def set_energies(self, new_energies: nuTens.pyNuTens.tensor.Tensor) -> BaseMatterSolver:
        """
        Set the neutrino energies
        """
    def set_masses(self, new_masses: nuTens.pyNuTens.tensor.Tensor) -> BaseMatterSolver:
        """
        Set the neutrino masses the solver should use
        """
    def set_mixing_matrix(self, new_matrix: nuTens.pyNuTens.tensor.Tensor) -> BaseMatterSolver:
        """
        Set the mixing matrix that the solver should use
        """
class BaseMixingMatrix:
    def build(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
class ConstDensitySolver(BaseMatterSolver):
    def __init__(self, n_generations: typing.SupportsInt | typing.SupportsIndex, device: nuTens.pyNuTens.dtype.device_type = nuTens.dtype.device_type.cpu) -> None:
        ...
    def get_density(self) -> float:
        """
        Get the density used by the solver
        """
    def set_antineutrino(self, new_value: bool) -> ConstDensitySolver:
        """
        Set the density that the solver should use
        """
    def set_density(self, new_value: typing.SupportsFloat | typing.SupportsIndex) -> ConstDensitySolver:
        """
        Set the density that the solver should use
        """
    def set_masses(self, new_value: nuTens.pyNuTens.tensor.Tensor) -> ConstDensitySolver:
        """
        Set the neutrino masses that the solver should use
        """
    def set_mixing_matrix(self, new_value: nuTens.pyNuTens.tensor.Tensor) -> ConstDensitySolver:
        """
        Set the mixing that the solver should use
        """
class DPpropagator(Propagator):
    def __init__(self, NR_iterations: typing.SupportsInt | typing.SupportsIndex, device: nuTens.pyNuTens.dtype.device_type = nuTens.dtype.device_type.cpu) -> None:
        ...
    def calculate_probs(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_deltacp(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_deltamsq21(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_deltamsq31(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_energies(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_theta12(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_theta13(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_theta23(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def set_antineutrino(self, new_value: bool) -> DPpropagator:
        """
        set whether to calculate anti-neutrino probabilities
        """
    def set_baseline(self, new_baseline: typing.SupportsFloat | typing.SupportsIndex) -> DPpropagator:
        """
        set the baseline
        """
    def set_deltacp(self, delta_cp: nuTens.pyNuTens.tensor.Tensor) -> DPpropagator:
        ...
    def set_density(self, new_density: typing.SupportsFloat | typing.SupportsIndex) -> DPpropagator:
        """
        set the density
        """
    def set_dmsq21(self, dmsq_21: nuTens.pyNuTens.tensor.Tensor) -> DPpropagator:
        ...
    def set_dmsq31(self, dmsq_31: nuTens.pyNuTens.tensor.Tensor) -> DPpropagator:
        ...
    def set_energies(self, new_energies: nuTens.pyNuTens.tensor.Tensor) -> DPpropagator:
        """
        set the neutrino energies
        """
    def set_sin_squared_thetas(self, new_value: bool) -> DPpropagator:
        """
        If `True`, the provided theta_ij values will be interpreted as sin^2(theta_ij) meaning that some of the computation can be shortcut and the probability calculation will be sped up. Note however that this will force the thetas to be in the lower octant (which is probably fine for most applications)
        """
    def set_theta12(self, theta_12: nuTens.pyNuTens.tensor.Tensor) -> DPpropagator:
        ...
    def set_theta13(self, theta_13: nuTens.pyNuTens.tensor.Tensor) -> DPpropagator:
        ...
    def set_theta23(self, theta_23: nuTens.pyNuTens.tensor.Tensor) -> DPpropagator:
        ...
class PMNSmatrix(BaseMixingMatrix):
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, device: nuTens.pyNuTens.dtype.device_type) -> None:
        ...
    def get_deltacp_tensor(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_theta12_tensor(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_theta13_tensor(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def get_theta23_tensor(self) -> nuTens.pyNuTens.tensor.Tensor:
        ...
    def set_deltacp(self, delta_cp: typing.SupportsFloat | typing.SupportsIndex) -> PMNSmatrix:
        ...
    def set_theta12(self, theta_12: typing.SupportsFloat | typing.SupportsIndex) -> PMNSmatrix:
        ...
    def set_theta13(self, theta_13: typing.SupportsFloat | typing.SupportsIndex) -> PMNSmatrix:
        ...
    def set_theta23(self, theta_23: typing.SupportsFloat | typing.SupportsIndex) -> PMNSmatrix:
        ...
class Propagator:
    def __init__(self, n_generations: typing.SupportsInt | typing.SupportsIndex, device: nuTens.pyNuTens.dtype.device_type = nuTens.dtype.device_type.cpu) -> None:
        ...
    def calculate_probabilities(self) -> nuTens.pyNuTens.tensor.Tensor:
        """
        Calculate the oscillation probabilities for neutrinos of specified energies
        """
    def get_baseline(self) -> float:
        """
        Get the baseline used by the propagator
        """
    def set_antineutrino(self, new_value: bool) -> Propagator:
        """
        Set whether the propagator should calculate oscillations for anti-neutrinos
        """
    def set_baseline(self, new_value: typing.SupportsFloat | typing.SupportsIndex) -> Propagator:
        """
        Set the baseline that the propagator should use
        """
    def set_energies(self, new_energies: nuTens.pyNuTens.tensor.Tensor) -> Propagator:
        """
        Set the neutrino energies that the propagator should use
        """
    def set_masses(self, new_masses: nuTens.pyNuTens.tensor.Tensor) -> Propagator:
        """
        Set the neutrino mass state eigenvalues
        """
    def set_matter_solver(self, new_matter_solver: BaseMatterSolver) -> Propagator:
        """
        Set the matter effect solver that the propagator should use
        """
    def set_mixing_matrix(self, new_matrix: nuTens.pyNuTens.tensor.Tensor) -> Propagator:
        """
        Set the mixing matrix that the propagator should use
        """
