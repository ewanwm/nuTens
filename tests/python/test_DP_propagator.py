import unittest
import math as m
import nuTens as nt
from nuTens.tensor import Tensor
from nuTens.testing import nufast_probability_matter
from nuTens.propagator import DPpropagator

import numpy as np
import typing

import pytest

@pytest.mark.parametrize("theta23", np.linspace(0.0, 0.5 * m.pi, 10, True))
class TestDPpropagator: 

    baseline = 295.0 * nt.units.km
    density = 2.6

    m1 = 0.0   * nt.units.eV
    m2 = 0.008 * nt.units.eV
    m3 = 0.02  * nt.units.eV

    dmsq21 = Tensor([m2 * m2 - m1 * m1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
    dmsq31 = Tensor([m3 * m3 - m1 * m1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

    dcp = Tensor([m.pi / 4.0], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

    theta13 = Tensor([0.3 * m.pi], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
    theta12 = Tensor([0.2 * m.pi], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
    
    energy = Tensor([0.5 * nt.units.GeV], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False).unsqueeze(0)

    def test_compare_nufast(self, theta23:float):

        theta23_tensor = Tensor([theta23], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

        # set up DPpropagator
        dp_propagator = DPpropagator(self.baseline, False, self.density, 10)
        dp_propagator.set_energies(self.energy)
        dp_propagator.set_parameters(
            self.theta12, theta23_tensor, self.theta13, 
            self.dcp, self.dmsq21, self.dmsq31)
        
        print("theta12: ", self.theta12.to_string())
        print("theta13: ", self.theta13.to_string())
        print("theta23: ", theta23_tensor.to_string())
        print("deltacp: ", self.dcp.to_string())
        print("dmsq21:  ", self.dmsq21.to_string())
        print("dmsq31:  ", self.dmsq31.to_string())
        print()

        dp_probabilities = dp_propagator.calculate_probs()

        print("DPpropagator probabilities:")
        print(dp_probabilities.to_string())
        print()

        nufast_probabilities = np.array(
            nufast_probability_matter(
                m.sin(self.theta12.get_value([0]).real) ** 2,
                m.sin(self.theta13.get_value([0]).real) ** 2,
                m.sin(theta23) ** 2,
                self.dcp.get_value([0]).real,
                -self.dmsq21.get_value([0]).real,
                -self.dmsq31.get_value([0]).real,
                self.baseline / nt.units.km,
                self.energy.get_value([0,0]).real / nt.units.GeV,
                1.0,
                self.density,
                10 
            )
        )

        print("nufast probabilities: ")
        print(nufast_probabilities)

        assert (
            pytest.approx(dp_probabilities.get_value([0, 0, 0]), abs = 1e-6) == nufast_probabilities[0][0]
            ), f"DPpropagator osc prob [0,0] != nufast osc prob"
        assert (
            pytest.approx(dp_probabilities.get_value([0, 0, 1]), abs = 1e-6) == nufast_probabilities[0][1]
            ), f"DPpropagator osc prob [0,1] != nufast osc prob"
        assert (
            pytest.approx(dp_probabilities.get_value([0, 0, 2]), abs = 1e-6) == nufast_probabilities[0][2]
            ), f"DPpropagator osc prob [0,2] != nufast osc prob"
        

        assert (
            pytest.approx(dp_probabilities.get_value([0, 1, 0]), abs = 1e-6) == nufast_probabilities[1][0]
            ), f"DPpropagator osc prob [1,0] != nufast osc prob"
        assert (
            pytest.approx(dp_probabilities.get_value([0, 1, 1]), abs = 1e-6) == nufast_probabilities[1][1]
            ), f"DPpropagator osc prob [1,1] != nufast osc prob"
        assert (
            pytest.approx(dp_probabilities.get_value([0, 1, 2]), abs = 1e-6) == nufast_probabilities[1][2]
            ), f"DPpropagator osc prob [1,2] != nufast osc prob"
        

        assert (
            pytest.approx(dp_probabilities.get_value([0, 2, 0]), abs = 1e-6) == nufast_probabilities[2][0]
            ), f"DPpropagator osc prob [2,0] != nufast osc prob"
        assert (
            pytest.approx(dp_probabilities.get_value([0, 2, 1]), abs = 1e-6) == nufast_probabilities[2][1]
            ), f"DPpropagator osc prob [2,1] != nufast osc prob"
        assert (
            pytest.approx(dp_probabilities.get_value([0, 2, 2]), abs = 1e-6) == nufast_probabilities[2][2]
            ), f"DPpropagator osc prob [2,2] != nufast osc prob"


if __name__ == '__main__':
    unittest.main()