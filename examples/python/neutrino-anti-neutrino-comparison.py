import nuTens as nt
import numpy as np
from nuTens import tensor
from nuTens.tensor import Tensor
import matplotlib.pyplot as plt
import math as m
import typing

N_ENERGIES = 10000

def build_PMNS(theta12: Tensor, theta13: Tensor, theta23: Tensor, deltaCP: Tensor):
    """ Construct a mixing matrix in the usual parameterisation """
    # set up the three matrices to build the mixing matrix
    M1 = Tensor.zeros([1, 3, 3], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
    M2 = Tensor.zeros([1, 3, 3], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)
    M3 = Tensor.zeros([1, 3, 3], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

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


## First we build up a tensor to contain the test energies
energies = Tensor.ones([N_ENERGIES, 1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

for i, e in enumerate(np.logspace(-2, 1, N_ENERGIES, True)):
    energies.set_value([i,0], e * nt.units.GeV)
energies.requires_grad(True)

## define tensors with oscillation parameters
theta23 = Tensor([0.82], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)
theta13 = Tensor([0.15], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)
theta12 = Tensor([0.58], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)
deltaCP = Tensor([m.pi / 2.0], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)

## make the matrix
PMNS = build_PMNS(theta12, theta13, theta23, deltaCP)

## set the mass tensor
masses = Tensor.zeros([1,3], nt.dtype.scalar_type.float, nt.dtype.device_type.cpu, False)

masses.set_value([0,0], 0.0)
masses.set_value([0,1], 0.00868 * nt.units.eV)
masses.set_value([0,2], 0.0501 * nt.units.eV)

masses.requires_grad(True)

baseline = 295.0 * nt.units.km

## set up the propagator object
propagator = nt.propagator.Propagator(3, baseline)
matter_solver = nt.propagator.ConstDensitySolver(3, 2.79)

propagator.set_matter_solver(matter_solver)
propagator.set_mixing_matrix(PMNS)
propagator.set_masses(masses)

## run!
propagator.set_energies(energies)
probabilities = propagator.calculate_probabilities()
propagator.set_antineutrino(True)
antinu_probabilities = propagator.calculate_probabilities()

energy_list = []
for i in range(N_ENERGIES):
    energy_list.append(energies.get_value([i, 0]))

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot([ e / nt.units.GeV for e in energy_list], [probabilities.get_value([i, 1, 1]) for i in range(N_ENERGIES)], linewidth=0.7, label = "mu -> mu", c="C0")
axs[0].plot([ e / nt.units.GeV for e in energy_list], [probabilities.get_value([i, 1, 2]) for i in range(N_ENERGIES)], linewidth=0.7, label = "mu -> tau", c="C1")
axs[1].plot([ e / nt.units.GeV for e in energy_list], [probabilities.get_value([i, 1, 0]) for i in range(N_ENERGIES)], linewidth=0.7, label = "mu -> e", c="C2")

axs[0].plot([ e / nt.units.GeV for e in energy_list], [antinu_probabilities.get_value([i, 1, 1]) for i in range(N_ENERGIES)], linestyle="dotted", label = "anti_mu -> anti_mu", c="C0")
axs[0].plot([ e / nt.units.GeV for e in energy_list], [antinu_probabilities.get_value([i, 1, 2]) for i in range(N_ENERGIES)], linestyle="dotted", label = "anti_mu -> anti_tau", c="C1")
axs[1].plot([ e / nt.units.GeV for e in energy_list], [antinu_probabilities.get_value([i, 1, 0]) for i in range(N_ENERGIES)], linestyle="dotted", label = "anti_mu -> anti_e", c="C2")

fig.supxlabel("Energy [GeV]")
fig.supylabel("Oscillation probability")
axs[0].legend()
axs[1].legend()

axs[0].set_xscale("log")
axs[1].set_xscale("log")

fig.suptitle("Osc probs with deltaCP = pi / 2")

plt.show()
plt.savefig("nu-vs-antinu-oscillation-probabilities-dcp-0.5-pi.png")
plt.clf()



## Now lets do dcp = 0 to check that there are no oscillations

deltaCP = Tensor([0.0], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)

## make the matrix
PMNS = build_PMNS(theta12, theta13, theta23, deltaCP)

## run!
probabilities = propagator.calculate_probabilities()
propagator.set_antineutrino(True)
antinu_probabilities = propagator.calculate_probabilities()

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot([ e / nt.units.GeV for e in energy_list], [probabilities.get_value([i, 1, 1]) for i in range(N_ENERGIES)], linewidth=0.7, label = "mu -> mu", c="C0")
axs[0].plot([ e / nt.units.GeV for e in energy_list], [probabilities.get_value([i, 1, 2]) for i in range(N_ENERGIES)], linewidth=0.7, label = "mu -> tau", c="C1")
axs[1].plot([ e / nt.units.GeV for e in energy_list], [probabilities.get_value([i, 1, 0]) for i in range(N_ENERGIES)], linewidth=0.7, label = "mu -> e", c="C2")

axs[0].plot([ e / nt.units.GeV for e in energy_list], [antinu_probabilities.get_value([i, 1, 1]) for i in range(N_ENERGIES)], linestyle="dotted", label = "anti_mu -> anti_mu", c="C0")
axs[0].plot([ e / nt.units.GeV for e in energy_list], [antinu_probabilities.get_value([i, 1, 2]) for i in range(N_ENERGIES)], linestyle="dotted", label = "anti_mu -> anti_tau", c="C1")
axs[1].plot([ e / nt.units.GeV for e in energy_list], [antinu_probabilities.get_value([i, 1, 0]) for i in range(N_ENERGIES)], linestyle="dotted", label = "anti_mu -> anti_e", c="C2")

fig.supxlabel("Energy [GeV]")
fig.supylabel("Oscillation probability")
axs[0].legend()
axs[1].legend()

fig.suptitle("Osc probs with deltaCP = 0.0")

axs[0].set_xscale("log")
axs[1].set_xscale("log")

plt.show()
plt.savefig("nu-vs-antinu-oscillation-probabilities-dcp-0.png")