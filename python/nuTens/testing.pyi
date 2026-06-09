"""
Some helpful utilities to use when writing python tests for your code
"""
from __future__ import annotations
import typing
__all__: list[str] = ['ThreeFlavourBarger', 'TwoFlavourBarger', 'nufast_probability_matter']
class ThreeFlavourBarger:
    def __init__(self) -> None:
        ...
    def alpha(self, energy: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Calculates alpha term used in calculating the mass eigenvalues
        """
    def beta(self, energy: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Calculates beta term used in calculating the mass eigenvalues
        """
    def calculate_effective_m2(self, energy: typing.SupportsFloat | typing.SupportsIndex, index: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Calculates the effective hamiltonian eigenvalues (the m_nuTens^2) in matter
        """
    def calculate_prob(self, energy: typing.SupportsFloat | typing.SupportsIndex, i: typing.SupportsInt | typing.SupportsIndex, j: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Calculate probability of transitioning from state i to state j for a given energy
        """
    def gamma(self, energy: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Calculates gamma term used in calculating the mass eigenvalues
        """
    def get_hamiltonian_element(self, energy: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> complex:
        """
        Calculates an element of the Hamiltonian
        """
    def get_transition_matrix_element(self, energy: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> complex:
        """
        Calculates an element of the transition matrix from one mass eigenstate to another due to the presense of matter
        """
    def set_params(self, m1: typing.SupportsFloat | typing.SupportsIndex, m2: typing.SupportsFloat | typing.SupportsIndex, m3: typing.SupportsFloat | typing.SupportsIndex, theta12: typing.SupportsFloat | typing.SupportsIndex, theta13: typing.SupportsFloat | typing.SupportsIndex, theta23: typing.SupportsFloat | typing.SupportsIndex, deltaCP: typing.SupportsFloat | typing.SupportsIndex, baseline: typing.SupportsFloat | typing.SupportsIndex, density: typing.SupportsFloat | typing.SupportsIndex = -999.9000244140625, anti_neutrino: bool = False) -> None:
        ...
class TwoFlavourBarger:
    def __init__(self) -> None:
        ...
    def calculate_effective_angle(self, energy: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Calculates the effective mixing angle, alpha, in matter
        """
    def calculate_effective_dm2(self, energy: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Calculates the effective delta m_nuTens^2 in matter
        """
    def calculate_prob(self, energy: typing.SupportsFloat | typing.SupportsIndex, i: typing.SupportsInt | typing.SupportsIndex, j: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Calculate probability of transitioning from state i to state j for a given energy
        """
    def get_PMNS_element(self, energy: typing.SupportsFloat | typing.SupportsIndex, i: typing.SupportsInt | typing.SupportsIndex, j: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Calculates the effective i,j-th element of the mizing matrix for a given energy
        """
    def lm(self) -> float:
        """
        Calculates the matter oscillation length
        """
    def lv(self, energy: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Calculates the vacuum oscillation length
        """
    def set_params(self, m1: typing.SupportsFloat | typing.SupportsIndex, m2: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex, baseline: typing.SupportsFloat | typing.SupportsIndex, density: typing.SupportsFloat | typing.SupportsIndex = -999.9000244140625, anti_neutrino: bool = False) -> None:
        ...
def nufast_probability_matter(sin_squared_theta12: typing.SupportsFloat | typing.SupportsIndex, sin_squared_theta13: typing.SupportsFloat | typing.SupportsIndex, sin_squared_theta23: typing.SupportsFloat | typing.SupportsIndex, delta_cp: typing.SupportsFloat | typing.SupportsIndex, delta_m_squared_21: typing.SupportsFloat | typing.SupportsIndex, delta_m_squared_31: typing.SupportsFloat | typing.SupportsIndex, baseline: typing.SupportsFloat | typing.SupportsIndex, energy: typing.SupportsFloat | typing.SupportsIndex, rho: typing.SupportsFloat | typing.SupportsIndex, Ye: typing.SupportsFloat | typing.SupportsIndex, N_Newton: typing.SupportsFloat | typing.SupportsIndex) -> list[list[float]]:
    """
    Calculates the oscillation probabilities using nufast
    """
