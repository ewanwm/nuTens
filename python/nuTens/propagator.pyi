from __future__ import annotations
import collections.abc
import nuTens._pyNuTens.tensor
import typing
__all__: list[str] = ['BaseMatterSolver', 'BaseMixingMatrix', 'ConstDensitySolver', 'DPpropagator', 'PMNSmatrix', 'Propagator']
class BaseMatterSolver:
    def calculate_eigenvalues(self, eigenvector_out: nuTens._pyNuTens.tensor.Tensor, eigenvalue_out: nuTens._pyNuTens.tensor.Tensor) -> None:
        """
        calculate the eigenvalues of the Hamiltonian
        """
    def set_antineutrino(self, new_value: bool) -> None:
        """
        Set whether the solver should calculate values for anti-neutrinos
        """
    def set_energies(self, new_energies: nuTens._pyNuTens.tensor.Tensor) -> None:
        """
        Set the neutrino energies
        """
    def set_masses(self, new_masses: nuTens._pyNuTens.tensor.Tensor) -> None:
        """
        Set the neutrino masses the solver should use
        """
    def set_mixing_matrix(self, new_matrix: nuTens._pyNuTens.tensor.Tensor) -> None:
        """
        Set the mixing matrix that the solver should use
        """
class BaseMixingMatrix:
    def build(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
class ConstDensitySolver(BaseMatterSolver):
    def __init__(self, n_generations: typing.SupportsInt | typing.SupportsIndex, density: typing.SupportsFloat | typing.SupportsIndex, anti_neutrino: bool = False) -> None:
        ...
    def get_density(self) -> float:
        """
        Get the density used by the solver
        """
    def set_density(self, new_value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the density that the solver should use
        """
class DPpropagator(Propagator):
    def __init__(self, baseline: typing.SupportsFloat | typing.SupportsIndex, anti_neutrino: bool = False, density: typing.SupportsFloat | typing.SupportsIndex, NR_iterations: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def calculate_probs(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_deltacp(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_deltamsq21(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_deltamsq31(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_energies(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_theta12(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_theta13(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_theta23(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def set_baseline(self, new_baseline: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        set the baseline
        """
    def set_density(self, new_density: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        set the density
        """
    def set_energies(self, new_energies: nuTens._pyNuTens.tensor.Tensor) -> None:
        """
        set the neutrino energies
        """
    def set_parameters(self, new_theta12: nuTens._pyNuTens.tensor.Tensor, new_theta23: nuTens._pyNuTens.tensor.Tensor, new_theta13: nuTens._pyNuTens.tensor.Tensor, new_deltaCP: nuTens._pyNuTens.tensor.Tensor, new_deltamsq21: nuTens._pyNuTens.tensor.Tensor, new_deltamsq31: nuTens._pyNuTens.tensor.Tensor, sin_squared_thetas: bool = False) -> None:
        """
        set the parameters for the oscillation calculations
        """
    def set_sin_squared_thetas(self, new_value: bool) -> None:
        """
        If `True`, the provided theta_ij values will be interpreted as sin^2(theta_ij) meaning that some of the computation can be shortcut and the probability calculation will be sped up. Note however that this will force the thetas to be in the lower octant (which is probably fine for most applications)
        """
class PMNSmatrix(BaseMixingMatrix):
    def __init__(self) -> None:
        ...
    def get_delta_cp_tensor(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_theta_12_tensor(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_theta_13_tensor(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def get_theta_23_tensor(self) -> nuTens._pyNuTens.tensor.Tensor:
        ...
    def set_parameter_values(self, theta_12: typing.SupportsFloat | typing.SupportsIndex, theta_13: typing.SupportsFloat | typing.SupportsIndex, theta_23: typing.SupportsFloat | typing.SupportsIndex, delta_cp: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Propagator:
    def __init__(self, n_generations: typing.SupportsInt | typing.SupportsIndex, baseline: typing.SupportsFloat | typing.SupportsIndex, anti_neutrino: bool = False) -> None:
        ...
    def calculate_probabilities(self) -> nuTens._pyNuTens.tensor.Tensor:
        """
        Calculate the oscillation probabilities for neutrinos of specified energies
        """
    def get_baseline(self) -> float:
        """
        Get the baseline used by the propagator
        """
    def set_antineutrino(self, new_value: bool) -> None:
        """
        Set whether the propagator should calculate oscillations for anti-neutrinos
        """
    def set_baseline(self, new_value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the baseline that the propagator should use
        """
    def set_energies(self, new_energies: nuTens._pyNuTens.tensor.Tensor) -> None:
        """
        Set the neutrino energies that the propagator should use
        """
    def set_masses(self, new_masses: nuTens._pyNuTens.tensor.Tensor) -> None:
        """
        Set the neutrino mass state eigenvalues
        """
    def set_matter_solver(self, new_matter_solver: BaseMatterSolver) -> None:
        """
        Set the matter effect solver that the propagator should use
        """
    @typing.overload
    def set_mixing_matrix(self, new_matrix: nuTens._pyNuTens.tensor.Tensor) -> None:
        """
        Set the mixing matrix that the propagator should use
        """
    @typing.overload
    def set_mixing_matrix(self, indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a particular value within the mixing matrix used by the propagator
        """
    @typing.overload
    def set_mixing_matrix(self, indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], value: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the mixing matrix that the propagator should use
        """
