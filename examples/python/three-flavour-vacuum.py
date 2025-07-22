import nuTens as nt
from nuTens import tensor
from nuTens.tensor import Tensor
import matplotlib.pyplot as plt
import typing

N_ENERGIES = 10000

## First we build up a tensor to contain the test energies
energies = Tensor.ones([N_ENERGIES, 1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

for i in range(N_ENERGIES):
    energies.set_value([i,0], (1.0e-6 + i*0.2e-3) * nt.units.GeV)

energies.requires_grad(True)

## make the matrix
PMNS = nt.propagator.PMNSmatrix()
PMNS.set_parameter_values(theta_12 = 0.58, theta_13 = 0.15, theta_23 = 0.82, delta_cp = 1.5)

## set the mass tensor
masses = Tensor.zeros([1,3], nt.dtype.scalar_type.float, nt.dtype.device_type.cpu, False)

masses.set_value([0,0], 0.0)
masses.set_value([0,1], 0.00868 * nt.units.eV)
masses.set_value([0,2], 0.0501 * nt.units.eV)

## print info about the parameters
print("PMNS: ")
print(PMNS.build().to_string())

print("\nMasses: ")
print(masses.to_string())
print()

## set up the propagator object
propagator = nt.propagator.Propagator(3, 295.0 * nt.units.km)
matter_solver = nt.propagator.ConstDensitySolver(3, 2.79)

## uncomment for matter oscillations
#propagator.set_matter_solver(matter_solver)

propagator.set_mixing_matrix(PMNS.build())
propagator.set_masses(masses)

## run!
propagator.set_energies(energies)
probabilities = propagator.calculate_probabilities()

## print out some test values
prob_sum = tensor.scale(tensor.sum(probabilities, [0]), 1.0 / float(N_ENERGIES))
print("energy integrated probabilities: ")
print(prob_sum.to_string())

## check that the autograd functionality works
mu_survival_prob = prob_sum.get_values([1,1])
print("mu survival prob:")
print(mu_survival_prob.to_string())

mu_survival_prob.backward()
print("theta_13 grad: ")
print(PMNS.get_theta_13_tensor().grad().to_string())


## make plots of the oscillation probabilities
energy_list = []
e_survival_prob_list = []
mu_survival_prob_list = []
tau_survival_prob_list = []

mu_to_e_prob_list = []
mu_to_tau_prob_list = []
mu_total_prob_list = []

for i in range(N_ENERGIES):
    energy_list.append(energies.get_value([i, 0]))
    e_survival_prob_list.append(probabilities.get_value([i, 0, 0]))
    mu_survival_prob_list.append(probabilities.get_value([i, 1, 1]))
    tau_survival_prob_list.append(probabilities.get_value([i, 2, 2]))

    mu_to_e_prob_list.append(probabilities.get_value([i, 1, 0]))
    mu_to_tau_prob_list.append(probabilities.get_value([i, 1, 2]))

    mu_total_prob_list.append(
        probabilities.get_value([i, 1, 0]) +
        probabilities.get_value([i, 1, 1]) + 
        probabilities.get_value([i, 1, 2])
    )

plt.plot([ e / nt.units.GeV for e in energy_list], e_survival_prob_list, label = "electron")
plt.plot([ e / nt.units.GeV for e in energy_list], mu_survival_prob_list, label = "muon")
plt.plot([ e / nt.units.GeV for e in energy_list], tau_survival_prob_list, label = "tau")
plt.xlabel("Energy [GeV]")
plt.ylabel("Survival probability")
plt.legend()
plt.show()
plt.savefig("survival_probs.png")

plt.clf()
fig, axs = plt.subplots(2, 1, sharex=True)
axs[1].plot([ e / nt.units.GeV for e in energy_list], mu_to_e_prob_list, label = "numu -> nue")
axs[1].set_ylim((0.0, 0.1))
axs[0].plot([ e / nt.units.GeV for e in energy_list], mu_to_tau_prob_list, label = "numu -> nutau")
axs[0].plot([ e / nt.units.GeV for e in energy_list], mu_survival_prob_list, label = "numu -> numu")
axs[0].plot([ e / nt.units.GeV for e in energy_list], mu_total_prob_list, label = "Total")
axs[1].set_xlabel("Energy [GeV]")
axs[0].legend()
axs[1].legend()
fig.suptitle("Three flavour oscillation probabilities")
fig.supylabel("Oscillation probability")
fig.savefig("oscillation_probs.png")
