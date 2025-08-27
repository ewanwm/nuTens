import unittest
import math as m
import nuTens as nt
from nuTens.tensor import Tensor, matmul
from nuTens.testing import TwoFlavourBarger
from nuTens.propagator import ConstDensitySolver

import numpy as np
import typing

import pytest


@pytest.mark.parametrize("mass_diff", np.linspace(0.001, 0.01, 10, True))
@pytest.mark.parametrize("theta",     np.linspace(0.0, 2.0 * m.pi, 30, True))
class TestTwoFlavourConstMatter: 

    baseline=295.0 * nt.units.km
    density=2.5    
    energy = 1.0 * nt.units.GeV
    energy_tensor = Tensor.ones([1, 1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
    energy_tensor.set_value([0, 0], energy)

    def setup_tensor_inputs(self, mass_diff:float, theta:float) -> typing.Tuple[Tensor]:
        
        pmns = Tensor.zeros([1, 2, 2], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
        pmns.set_value([0, 0, 0], m.cos(theta))
        pmns.set_value([0, 0, 1], m.sin(theta))
        pmns.set_value([0, 1, 0], -m.sin(theta))
        pmns.set_value([0, 1, 1], m.cos(theta))

        masses = Tensor.zeros([1,2], nt.dtype.scalar_type.float, nt.dtype.device_type.cpu, False)
        masses.set_value([0,0], 0.0)
        masses.set_value([0,1], mass_diff)

        return pmns, masses

    def test_compare_effective_PMNS(self, mass_diff:float, theta:float):

        # set up barger
        barger = TwoFlavourBarger()
        barger.set_params(m1=0.0, m2=mass_diff, theta=theta, baseline=self.baseline, density=self.density)
        
        pmns, masses = self.setup_tensor_inputs(mass_diff, theta)

        # set up tensor solver
        tensor_solver = ConstDensitySolver(2, self.density)
        tensor_solver.set_mixing_matrix(pmns)
        tensor_solver.set_masses(masses)
        tensor_solver.set_energies(self.energy_tensor)

        # calculate the evals + evecs to print for help when debugging
        evecs = nt.tensor.Tensor()
        evals = nt.tensor.Tensor()
        tensor_solver.calculate_eigenvalues(evecs, evals)

        tensor_effective_pmns = matmul(pmns, evecs)

        print(f"Tensor solver evals: {evals.to_string()}")

        print(f"Tensor solver evecs: {evecs.to_string()}")

        print(f"Tensor PMNS: \n{tensor_effective_pmns.to_string()}")

        print(f"Barger PMNS:\n"
            f"{barger.get_PMNS_element(self.energy, i=0, j=0)}, "
            f"{barger.get_PMNS_element(self.energy, i=0, j=1)}, \n"
            f"{barger.get_PMNS_element(self.energy, i=1, j=0)}, "
            f"{barger.get_PMNS_element(self.energy, i=1, j=1)}, "
        )

        assert (
            pytest.approx(abs(tensor_effective_pmns.get_value([0, 0, 0])), abs = 1e-6) == abs(barger.get_PMNS_element(self.energy, i=0, j=0))
            ), f"ConstMatterSolver effectivePMNS[0,0] != barger PMNS"
        assert (
            pytest.approx(abs(tensor_effective_pmns.get_value([0, 0, 1])), abs = 1e-6) == abs(barger.get_PMNS_element(self.energy, i=0, j=1))
            ), f"ConstMatterSolver effectivePMNS[0,0] != barger PMNS"
        assert (
            pytest.approx(abs(tensor_effective_pmns.get_value([0, 1, 0])), abs = 1e-6) == abs(barger.get_PMNS_element(self.energy, i=1, j=0))
            ), f"ConstMatterSolver effectivePMNS[0,0] != barger PMNS"
        assert (
            pytest.approx(abs(tensor_effective_pmns.get_value([0, 1, 1])), abs = 1e-6) == abs(barger.get_PMNS_element(self.energy, i=1, j=1))
            ), f"ConstMatterSolver effectivePMNS[0,0] != barger PMNS"


    def test_compare_osc_probs(self, mass_diff:float, theta:float):

        # set up barger
        barger = TwoFlavourBarger()
        barger.set_params(m1=0.0, m2=mass_diff, theta=theta, baseline=self.baseline, density=self.density)
        
        pmns, masses = self.setup_tensor_inputs(mass_diff, theta)

        # set up tensor solver
        propagator = nt.propagator.Propagator(2, self.baseline)
        matter_solver = ConstDensitySolver(2, self.density)
        
        propagator.set_matter_solver(matter_solver)
        propagator.set_mixing_matrix(pmns)
        propagator.set_masses(masses)
        propagator.set_energies(self.energy_tensor)

        # calculate the evals + evecs + effective PMNS to print 
        # for help when debugging
        evecs = nt.tensor.Tensor()
        evals = nt.tensor.Tensor()
        matter_solver.calculate_eigenvalues(evecs, evals)
        tensor_effective_pmns = matmul(pmns, evecs)

        tensor_osc_probs = propagator.calculate_probabilities()

        print(f"Tensor solver evals: {evals.to_string()}")

        print(f"Tensor solver evecs: {evecs.to_string()}")

        print(f"Tensor PMNS: \n{tensor_effective_pmns.to_string()}")

        print(f"Barger PMNS:\n"
            f"{barger.get_PMNS_element(self.energy, i=0, j=0)}, "
            f"{barger.get_PMNS_element(self.energy, i=0, j=1)}, \n"
            f"{barger.get_PMNS_element(self.energy, i=1, j=0)}, "
            f"{barger.get_PMNS_element(self.energy, i=1, j=1)}, "
        )

        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 0, 0]), abs = 1e-6) == barger.calculate_prob(self.energy, i=0, j=0)
            ), f"Const matter osc prob[0,0] != barger osc prob"
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 0, 1]), abs = 1e-6) == barger.calculate_prob(self.energy, i=0, j=1)
            ), f"Const matter osc prob[0,1] != barger osc prob"
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 1, 0]), abs = 1e-6) == barger.calculate_prob(self.energy, i=1, j=0)
            ), f"Const matter osc prob[1,0] != barger osc prob"
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 1, 1]), abs = 1e-6) == barger.calculate_prob(self.energy, i=1, j=1)
            ), f"Const matter osc prob[1,1] != barger osc prob"


if __name__ == '__main__':
    unittest.main()