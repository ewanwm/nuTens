import nuTens as nt
from nuTens import tensor
from nuTens.tensor import Tensor
import matplotlib.pyplot as plt
import typing

N_ENERGIES = 10000

def build_PMNS(theta12: Tensor):
    """ Construct a mixing matrix in the usual parameterisation """
    # set up the three matrices to build the mixing matrix
    PMNS = Tensor.zeros([1, 2, 2], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

    PMNS.set_value([0, 0, 0], tensor.cos(theta12))
    PMNS.set_value([0, 0, 1], tensor.sin(theta12))
    PMNS.set_value([0, 1, 0], -tensor.sin(theta12))
    PMNS.set_value([0, 1, 1], tensor.cos(theta12))

    return PMNS.requires_grad(True)

## First we build up a tensor to contain the test energies
energies = Tensor.ones([N_ENERGIES, 1], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, False)

for i in range(N_ENERGIES):
    energies.set_value([i,0], ( 1.0e-6 + i*0.2e-3 ) * nt.units.GeV)

energies.requires_grad(True)

## define tensors with oscillation parameters
theta12 = Tensor([0.15], nt.dtype.scalar_type.complex_float, nt.dtype.device_type.cpu, True)

## make the matrix
PMNS = build_PMNS(theta12)

## set the mass tensor
masses = Tensor.zeros([1,2], nt.dtype.scalar_type.float, nt.dtype.device_type.cpu, False)

masses.set_value([0,0], 0.00868 * nt.units.eV)
masses.set_value([0,1], 0.0501 * nt.units.eV)

masses.requires_grad(True)

## print info about the parameters
print("PMNS: ")
print(PMNS.to_string())

print("\nMasses: ")
print(masses.to_string())
print()

## set up the propagator object
propagator = nt.propagator.Propagator(2, 295 * nt.units.km)
matter_solver = nt.propagator.ConstDensitySolver(2, 2.79)

propagator.set_matter_solver(matter_solver)
propagator.set_mixing_matrix(PMNS)
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
print("theta_12 grad: ")
print(theta12.grad().to_string())

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
    
    mu_to_e_prob_list.append(probabilities.get_value([i, 1, 0]))
    
    mu_total_prob_list.append(
        probabilities.get_value([i, 1, 0]) +
        probabilities.get_value([i, 1, 1]) 
    )

plt.plot([ e / nt.units.GeV for e in energy_list ], e_survival_prob_list, label = "a -> a")
plt.plot([ e / nt.units.GeV for e in energy_list ], mu_survival_prob_list, label = "b -> b")
plt.xlabel("Energy [GeV]")
plt.ylabel("Survival probability")
plt.legend()
plt.show()
plt.savefig("two-flavour-matter-survival-probs.png")

plt.clf()
plt.plot([ e / nt.units.GeV for e in energy_list ], mu_to_e_prob_list, label = "b -> a")
plt.xlabel("Energy [GeV]")
plt.ylabel("Oscillation probability")
plt.legend()
plt.show()
plt.savefig("two-flavour-matter-oscillation-probs.png")