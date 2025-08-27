import unittest
import math as m
import nuTens as nt
from nuTens.tensor import Tensor, matmul, scale
from nuTens.testing import ThreeFlavourBarger
from nuTens.propagator import ConstDensitySolver, PMNSmatrix

import numpy as np
import typing

import pytest

@pytest.mark.parametrize("theta12",     np.linspace(0.0, 2.0 * m.pi, 5, True))
@pytest.mark.parametrize("theta13",     np.linspace(0.0, 2.0 * m.pi, 5, True))
@pytest.mark.parametrize("theta23",     np.linspace(0.0, 2.0 * m.pi, 5, True))
class TestTwoFlavourConstMatter: 

    m1 = 0.0
    m2 = 0.008 * nt.units.eV
    m3 = 0.01 * nt.units.eV

    deltaCP = 0.25 * m.pi

    baseline=295.0 * nt.units.km
    density=2.5    
    
    energy = 1.0 * nt.units.GeV
    
    energy_tensor = Tensor.ones([1, 1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
    energy_tensor.set_value([0, 0], energy)

    def setup_tensor_inputs(self, theta12:float, theta13:float, theta23:float) -> typing.Tuple[Tensor]:
        
        pmns = PMNSmatrix()
        pmns.set_parameter_values(theta12, theta13, theta23, self.deltaCP)

        masses = Tensor([self.m1, self.m2, self.m3], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False).add_batch_dim()

        return pmns.build(), masses

    def test_compare_osc_probs(self, theta12:float, theta13:float, theta23:float):

        print(f"------ const density oscillation probabilities ------\n")
        print(f"  theta12 = {theta12}")
        print(f"  theta13 = {theta13}")
        print(f"  theta23 = {theta23}\n")

        # set up barger
        barger = ThreeFlavourBarger()
        barger.set_params(self.m1, self.m2, self.m3, theta12, theta13, theta23, self.deltaCP, self.baseline, self.density)
        
        pmns, masses = self.setup_tensor_inputs(theta12, theta13, theta23)

        # set up tensor solver
        propagator = nt.propagator.Propagator(3, self.baseline)
        matter_solver = ConstDensitySolver(3, self.density)
        
        propagator.set_matter_solver(matter_solver)
        propagator.set_mixing_matrix(pmns)
        propagator.set_masses(masses)
        propagator.set_energies(self.energy_tensor)

        # calculate the evals + evecs + effective PMNS to print 
        # for help when debugging
        evecs = nt.tensor.Tensor()
        evals = nt.tensor.Tensor()
        matter_solver.calculate_eigenvalues(evecs, evals)

        tensor_osc_probs = propagator.calculate_probabilities()

        print(f"Tensor solver evals: {evals.to_string()}\n")

        print(f"Tensor solver effective Mi^2: {scale(evals, 2.0 * self.energy).to_string()}\n")

        print(f"Barger M1^2: {barger.calculate_effective_m2(self.energy, 0)}")
        print(f"Barger M2^2: {barger.calculate_effective_m2(self.energy, 1)}")
        print(f"Barger M3^2: {barger.calculate_effective_m2(self.energy, 2)}\n")

        print(f"Oscillation probabilities:")
        print(f"[0,0] tensor solver {tensor_osc_probs.get_value([0, 0, 0]):.5f} :: barger propagator {barger.calculate_prob(self.energy, i=0, j=0):.5f}")
        print(f"[0,1] tensor solver {tensor_osc_probs.get_value([0, 0, 1]):.5f} :: barger propagator {barger.calculate_prob(self.energy, i=0, j=1):.5f}")
        print(f"[0,2] tensor solver {tensor_osc_probs.get_value([0, 0, 2]):.5f} :: barger propagator {barger.calculate_prob(self.energy, i=0, j=2):.5f}")

        print(f"[1,0] tensor solver {tensor_osc_probs.get_value([0, 1, 0]):.5f} :: barger propagator {barger.calculate_prob(self.energy, i=1, j=0):.5f}")
        print(f"[1,1] tensor solver {tensor_osc_probs.get_value([0, 1, 1]):.5f} :: barger propagator {barger.calculate_prob(self.energy, i=1, j=1):.5f}")
        print(f"[1,2] tensor solver {tensor_osc_probs.get_value([0, 1, 2]):.5f} :: barger propagator {barger.calculate_prob(self.energy, i=1, j=2):.5f}")

        print(f"[2,0] tensor solver {tensor_osc_probs.get_value([0, 2, 0]):.5f} :: barger propagator {barger.calculate_prob(self.energy, i=2, j=0):.5f}")
        print(f"[2,1] tensor solver {tensor_osc_probs.get_value([0, 2, 1]):.5f} :: barger propagator {barger.calculate_prob(self.energy, i=2, j=1):.5f}")
        print(f"[2,2] tensor solver {tensor_osc_probs.get_value([0, 2, 2]):.5f} :: barger propagator {barger.calculate_prob(self.energy, i=2, j=2):.5f}")

        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 0, 0]), abs = 1e-5) == barger.calculate_prob(self.energy, i=0, j=0)
            ), f"Const matter osc prob[0,0] != barger osc prob"
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 0, 1]), abs = 1e-5) == barger.calculate_prob(self.energy, i=0, j=1)
            ), f"Const matter osc prob[0,1] != barger osc prob"
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 0, 2]), abs = 1e-5) == barger.calculate_prob(self.energy, i=0, j=2)
            ), f"Const matter osc prob[0,2] != barger osc prob"
        
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 1, 0]), abs = 1e-5) == barger.calculate_prob(self.energy, i=1, j=0)
            ), f"Const matter osc prob[1,0] != barger osc prob"
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 1, 1]), abs = 1e-5) == barger.calculate_prob(self.energy, i=1, j=1)
            ), f"Const matter osc prob[1,1] != barger osc prob"
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 1, 2]), abs = 1e-5) == barger.calculate_prob(self.energy, i=1, j=2)
            ), f"Const matter osc prob[1,2] != barger osc prob"
        
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 2, 0]), abs = 1e-5) == barger.calculate_prob(self.energy, i=2, j=0)
            ), f"Const matter osc prob[2,0] != barger osc prob"
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 2, 1]), abs = 1e-5) == barger.calculate_prob(self.energy, i=2, j=1)
            ), f"Const matter osc prob[2,1] != barger osc prob"
        assert (
            pytest.approx(tensor_osc_probs.get_value([0, 2, 2]), abs = 1e-5) == barger.calculate_prob(self.energy, i=2, j=2)
            ), f"Const matter osc prob[2,2] != barger osc prob"


if __name__ == '__main__':
    unittest.main()