import unittest
import math as m
import nuTens as nt
from nuTens import tensor
from nuTens.testing import TwoFlavourBarger
from nuTens.propagator import ConstDensitySolver

class TestTwoFlavourConstMatter(unittest.TestCase):

    def test_compare_barger(self):
        energy = 1.0 * nt.units.GeV
        m1=0.0
        m2=0.008 * nt.units.eV
        theta=0.88853
        baseline=295.0 * nt.units.km
        density=2.5

        barger = TwoFlavourBarger()
        barger.set_params(m1, m2, theta, baseline, density)

        print(f"Barger:"
              f"alpha={barger.calculate_effective_angle(energy)}, "
              f"DM^2 ={barger.calculate_effective_dm2(energy)}, \n"
              f"Probs: \n"
              f"p_00 ={barger.calculate_prob(energy, i=0, j=0)}, "
              f"p_01 ={barger.calculate_prob(energy, i=0, j=1)}, "
              f"p_10 ={barger.calculate_prob(energy, i=1, j=0)}, "
              f"p_11 ={barger.calculate_prob(energy, i=1, j=1)}, "
        )

        energy_tensor = tensor.ones([1, 1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
        energy_tensor.set_value([0, 0], energy)
        
        PMNS = nt.tensor.zeros([1, 2, 2], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)
        PMNS.set_value([0, 0, 0], m.cos(theta))
        PMNS.set_value([0, 0, 1], m.sin(theta))
        PMNS.set_value([0, 1, 0], -m.sin(theta))
        PMNS.set_value([0, 1, 1], m.cos(theta))

        masses = tensor.zeros([1,2], nt.dtype.scalar_type.float, nt.dtype.device_type.cpu, True)
        masses.set_value([0,0], m1)
        masses.set_value([0,1], m2)
        
        tensor_solver = ConstDensitySolver(2, density)
        tensor_solver.set_PMNS(PMNS)
        tensor_solver.set_masses(masses)
        tensor_solver.set_energies(energy_tensor)

        evecs = nt.tensor.Tensor()
        evals = nt.tensor.Tensor()

        tensor_solver.calculate_eigenvalues(evecs, evals)

        PMNSeff = tensor.matmul(PMNS, evecs)

        print(f"Tensor PMNS: \n{PMNSeff.to_string()}")

        print(f"Barger PMNS:\n"
            f"{barger.get_PMNS_element(energy, i=0, j=0)}, "
            f"{barger.get_PMNS_element(energy, i=0, j=1)}, \n"
            f"{barger.get_PMNS_element(energy, i=1, j=0)}, "
            f"{barger.get_PMNS_element(energy, i=1, j=1)}, "
        )

        self.assertTrue(abs(PMNSeff.get_value([0, 0, 0]) - barger.get_PMNS_element(energy, i=0, j=0)) < 0.0001, 
                        f"ConstMatterSolver effectivePMNS[0,0] != barger PMNS")
        self.assertTrue(abs(PMNSeff.get_value([0, 1, 1]) - barger.get_PMNS_element(energy, i=1, j=1)) < 0.0001, 
                        f"ConstMatterSolver effectivePMNS[1,1] != barger PMNS")
        self.assertTrue(abs(PMNSeff.get_value([0, 0, 1]) - barger.get_PMNS_element(energy, i=0, j=1)) < 0.0001, 
                        f"ConstMatterSolver effectivePMNS[0,1] != barger PMNS")
        self.assertTrue(abs(PMNSeff.get_value([0, 1, 0]) - barger.get_PMNS_element(energy, i=1, j=0)) < 0.0001, 
                        f"ConstMatterSolver effectivePMNS[1,0] != barger PMNS")


if __name__ == '__main__':
    unittest.main()