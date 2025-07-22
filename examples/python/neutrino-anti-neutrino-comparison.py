import nuTens as nt
import numpy as np
from nuTens import tensor
from nuTens.tensor import Tensor
import matplotlib.pyplot as plt
import math as m
import typing

N_ENERGIES = 10000

## First we build up a tensor to contain the test energies
energies = Tensor.ones([N_ENERGIES, 1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

for i, e in enumerate(np.logspace(-2, 1, N_ENERGIES, True)):
    energies.set_value([i,0], e * nt.units.GeV)
energies.requires_grad(True)

## make the matrix
PMNS = nt.propagator.PMNSmatrix()
PMNS.set_parameter_values(theta_12 = 0.58, theta_13 = 0.15, theta_23 = 0.82, delta_cp = m.pi / 2.0)

## set the mass tensor
masses = Tensor.zeros([1,3], nt.dtype.scalar_type.float, nt.dtype.device_type.cpu, False)

masses.set_value([0,0], 0.0)
masses.set_value([0,1], 0.00868 * nt.units.eV)
masses.set_value([0,2], 0.0501 * nt.units.eV)

baseline = 295.0 * nt.units.km

## set up the propagator object
propagator = nt.propagator.Propagator(3, baseline)
matter_solver = nt.propagator.ConstDensitySolver(3, 2.79)

propagator.set_matter_solver(matter_solver)
propagator.set_mixing_matrix(PMNS.build())
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



## Now lets do dcp = 0 to check that there is no difference between nu and anti-nu

## make the matrix
PMNS.set_parameter_values(theta_12 = 0.58, theta_13 = 0.15, theta_23 = 0.82, delta_cp = 0.0)

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