import torch
import pyNuTens as nt
from pyNuTens import tensor
from pyNuTens.tensor import Tensor
import matplotlib.pyplot as plt
import typing


def build_PMNS(theta12: Tensor, theta13: Tensor, theta23: Tensor, deltaCP: Tensor):
    # set up the three matrices to build the PMNS matrix
    M1 = nt.tensor.zeros([1, 3, 3], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)
    M2 = nt.tensor.zeros([1, 3, 3], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)
    M3 = nt.tensor.zeros([1, 3, 3], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)

    M1.set_value([0, 0, 0], 1.0)
    M1.set_value([0, 1, 1], tensor.cos(theta23))
    M1.set_value([0, 1, 2], tensor.sin(theta23))
    M1.set_value([0, 2, 1], -tensor.sin(theta23))
    M1.set_value([0, 2, 2], tensor.cos(theta23))
    M1.requires_grad(True)

    M2.set_value([0, 1, 1], 1.0)
    M2.set_value([0, 0, 0], tensor.cos(theta13))
    M2.set_value([0, 0, 2], tensor.mul(tensor.sin(theta13), tensor.exp(tensor.scale(deltaCP, -1.0J))))
    M2.set_value([0, 2, 0], -tensor.mul(tensor.sin(theta13), tensor.exp(tensor.scale(deltaCP, 1.0J))))
    M2.set_value([0, 2, 2], tensor.cos(theta13))
    M2.requires_grad(True)

    M3.set_value([0, 2, 2], 1.0)
    M3.set_value([0, 0, 0], tensor.cos(theta12))
    M3.set_value([0, 0, 1], tensor.sin(theta12))
    M3.set_value([0, 1, 0], -tensor.sin(theta12))
    M3.set_value([0, 1, 1], tensor.cos(theta12))
    M3.requires_grad(True)

    # Build PMNS
    PMNS = tensor.matmul(M1, tensor.matmul(M2, M3))
    PMNS.requires_grad(True)

    return PMNS


energies = nt.tensor.ones([100, 1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

for i in range(100):
    energies.set_value([i,0], 100.0 + i*1000.0)

energies.requires_grad(True)

theta23 = Tensor([0.23], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
theta13 = Tensor([0.13], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
theta12 = Tensor([0.12], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
deltaCP = Tensor([0.5], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

PMNS = build_PMNS(theta12, theta13, theta23, deltaCP)


masses = nt.tensor.zeros([1,3], nt.dtype.scalar_type.float, nt.dtype.device_type.cpu, True)

masses.set_value([0,0], 0.1)
masses.set_value([0,1], 0.2)
masses.set_value([0,2], 0.3)


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

prob_sum = tensor.sum(probabilities, [0])
print("energy integrated probabilities: ")
print(prob_sum.to_string())