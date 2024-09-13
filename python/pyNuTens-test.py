import torch
import pyNuTens as nt
import matplotlib.pyplot as plt

energies = nt.tensor.ones([100, 1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

for i in range(100):
    energies.set_value([i,0], 100.0 + i*1000.0)

energies.requires_grad(True)

PMNS = nt.tensor.zeros([3,3], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)
masses = nt.tensor.zeros([1,3], nt.dtype.scalar_type.float, nt.dtype.device_type.cpu, True)

masses.set_value([0,0], 0.1)
masses.set_value([0,1], 0.2)
masses.set_value([0,2], 0.3)

PMNS.set_value([0,0], 0.8)
PMNS.set_value([1,1], 0.6)
PMNS.set_value([2,2], 0.55)

PMNS.set_value([0,1], 0.1j + 0.45)
PMNS.set_value([0,2], 0.12j + 0.35)

PMNS.set_value([1,0], 0.23j + 0.12)
PMNS.set_value([1,2], 0.11j + 0.47)

PMNS.set_value([1,0], 0.2j + 0.12)
PMNS.set_value([1,0], 0.2j + 0.12)

print("PMNS: ")
print(PMNS.to_string())

print("\nMasses: ")
print(masses.to_string())
print()

propagator = nt.propagator.Propagator(3, 100.0)
matter_solver = nt.propagator.ConstDensitySolver(3, 2.79)

propagator.set_PMNS(PMNS)
propagator.set_masses(masses)
#propagator.set_matter_solver(matter_solver)

probabilities = propagator.calculate_probabilities(energies)

print("probabilities: ")
print(probabilities.to_string())