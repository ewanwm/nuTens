"""Example script using simple Barger propagator for two flavour neutrino oscillations

This will give you an example of how to use the  the Barger propagator 
implemented in the testing library of nuTens, which you should use to 
implement your own tests of any fancy new propagators you implement.

Will produce four plots:
  
    - two-flavour-vacuum-oscillation-barger-probs.png
    - two-flavour-matter-oscillation-barger-probs.png

    These can be compared against figure 2 in 
    [Vernon D. Barger, K. Whisnant, S. Pakvasa, and R. J. N. Phillips. Matter Effects on Three-Neutrino Oscillations. Phys. Rev. D, 22:2718, 1980.]
    to check that the propagator is working as expected, 
    and also as a useful reference to compare against.

    - two-flavour-vacuum-oscillation-barger-probs-LoverE.png
    - two-flavour-matter-oscillation-barger-probs-LoverE.png

    These are more standart L/E plots.

"""

import nuTens as nt
from nuTens.testing import TwoFlavourBarger
import matplotlib.pyplot as plt
import numpy as np
import typing


## oscillation parameters, taken from Barger paper
baseline = 5e6 * nt.units.m 
density  = 0.5 # g/N_a/cm^3 * N_a/N_e
mass_diff = 1.0 * nt.units.eV 
alpha = 0.3927 # 22.5 degrees

energies = np.logspace(start=4.0, stop=8.0, num=1000, endpoint=True, base=10.0) * nt.units.MeV
energy_list = [e for e in energies]

## "classical" Barger propagator
vacuum_barger = TwoFlavourBarger()
barger = TwoFlavourBarger()
vacuum_barger.set_params(0.0, mass_diff, alpha, baseline, 0)
barger.set_params(0.0, mass_diff, alpha, baseline, density)


#########################################################################
### Make plots as fn of E / dm^2 to compare with fig2 in Barger et al ###
#########################################################################

plt.clf()
plt.plot((energies / nt.units.MeV) / (mass_diff * mass_diff), [vacuum_barger.calculate_prob(e, i=0, j=0) for e in energy_list], label = "nu_a -> nu_a")
plt.plot((energies / nt.units.MeV) / (mass_diff * mass_diff), [vacuum_barger.calculate_prob(e, i=0, j=1) for e in energy_list], label = "nu_a -> nu_b")
plt.plot((energies / nt.units.MeV) / (mass_diff * mass_diff), [vacuum_barger.calculate_prob(e, i=0, j=0) + vacuum_barger.calculate_prob(e, i=0, j=1) for e in energy_list], label = "Total")
plt.xlabel("E / dm^2 [MeV / eV^2]")
plt.ylabel("Oscillation probability")
plt.title("Vacuum Oscillations")
plt.legend()
plt.semilogx()
plt.show()
plt.savefig("two-flavour-vacuum-oscillation-barger-probs.png")

plt.clf()
plt.plot((energies / nt.units.MeV) / (mass_diff * mass_diff), [barger.calculate_prob(e, i=0, j=0) for e in energy_list], label = "nu_a -> nu_a")
plt.plot((energies / nt.units.MeV) / (mass_diff * mass_diff), [barger.calculate_prob(e, i=0, j=1) for e in energy_list], label = "nu_a -> nu_b")
plt.plot((energies / nt.units.MeV) / (mass_diff * mass_diff), [barger.calculate_prob(e, i=0, j=0) + barger.calculate_prob(e, i=0, j=1) for e in energy_list], label = "Total")
plt.xlabel("E / dm^2 [MeV / eV^2]")
plt.ylabel("Oscillation probability")
plt.title("Matter Oscillations")
plt.legend()
plt.semilogx()
plt.show()
plt.savefig("two-flavour-matter-oscillation-barger-probs.png")




###############################
### Make plots as fn of L/E ###
###############################

# change up the parameters
baseline = 10.0 * nt.units.km # value not super important since we are plotting L/E
density  = 0.5 # g/N_a/cm^3 * N_a/N_e
mass_diff = 0.04816637831 * nt.units.eV 
alpha = 0.59437

energies = np.logspace(start=-4.0, stop=2.0, num=1000, endpoint=True, base=10.0) * nt.units.GeV
energy_list = [e for e in energies]

vacuum_barger.set_params(m1=0.0, m2=mass_diff, theta=alpha, baseline=baseline, density=0.0)
barger.set_params(0.0, mass_diff, alpha, baseline, density)

plt.clf()
plt.plot( ( baseline / nt.units.km) / (energies / nt.units.GeV), [vacuum_barger.calculate_prob(e, i=0, j=0) for e in energy_list], label = "nu_a -> nu_a")
plt.plot( ( baseline / nt.units.km) / (energies / nt.units.GeV), [vacuum_barger.calculate_prob(e, i=0, j=1) for e in energy_list], label = "nu_b -> nu_b")
plt.plot( ( baseline / nt.units.km) / (energies / nt.units.GeV), [vacuum_barger.calculate_prob(e, i=0, j=0) + vacuum_barger.calculate_prob(e, i=0, j=1) for e in energy_list], label = "Total")
plt.xlabel("L / E [km / GeV]")
plt.ylabel("Oscillation probability")
plt.legend()
plt.xlim((0,4000))
plt.show()
plt.savefig("two-flavour-vacuum-oscillation-barger-probs-LoverE.png")

plt.clf()
plt.plot( ( baseline / nt.units.km) / (energies / nt.units.GeV), [barger.calculate_prob(e, i=0, j=0) for e in energy_list], label = "nu_a -> nu_a")
plt.plot( ( baseline / nt.units.km) / (energies / nt.units.GeV), [barger.calculate_prob(e, i=0, j=1) for e in energy_list], label = "nu_a -> nu_b")
plt.plot( ( baseline / nt.units.km) / (energies / nt.units.GeV), [barger.calculate_prob(e, i=0, j=0) + barger.calculate_prob(e, i=0, j=1) for e in energy_list], label = "Total")
plt.xlabel("L / E [km / GeV]")
plt.ylabel("Oscillation probability")
plt.legend()
plt.xlim((0,4000))
plt.show()
plt.savefig("two-flavour-matter-oscillation-barger-probs-LoverE.png")