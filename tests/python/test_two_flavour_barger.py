import unittest
import math as m
import numpy as np
import nuTens as nt
from nuTens.testing import TwoFlavourBarger

class TestTwoFlavourConstMatter(unittest.TestCase):

    def test_vacuum_zero_theta_barger(self):
        """ Test that we get no oscillations when theta == 0 for a range of energies
        """

        baseline = 500.0 * nt.units.km
        barger = TwoFlavourBarger()

        barger.set_params(m1=0.0, m2=0.005, theta=0.0, baseline=baseline)
        energies = np.logspace(0.0, 2.0, 100 ) * nt.units.GeV

        for energy in energies:
        
            self.assertEqual(barger.calculate_prob(energy, 0, 0), 1.0,
                            f"a->a prob with theta == 0 is not 1 with E = {energy}")
            self.assertEqual(barger.calculate_prob(energy, 0, 1), 0.0,
                            f"a->b prob with theta == 0 is not 0 with E = {energy}")
            self.assertEqual(barger.calculate_prob(energy, 1, 1), 1.0,
                            f"b->b prob with theta == 0 is not 1 with E = {energy}")
            self.assertEqual(barger.calculate_prob(energy, 1, 0), 0.0,
                            f"b->a prob with theta == 0 is not 0 with E = {energy}")
            
    def test_vacuum_zero_dm2_barger(self):
        """ Test vacuum propagations for some fixed param values
        """

        baseline = 500.0 * nt.units.km
        barger = TwoFlavourBarger()

        # check that we get no vacuum oscillations when theta == 0 for a range of
        # energies
        barger.set_params(m1=0.01, m2=0.01, theta=m.pi/2.0, baseline=baseline)
        energies = np.logspace(0.0, 2.0, 100 ) * nt.units.GeV

        for energy in energies:
        
            self.assertEqual(barger.calculate_prob(energy, 0, 0), 1.0,
                            f"a->a prob with dm^2 == 0 is not 1 with E = {energy}")
            self.assertEqual(barger.calculate_prob(energy, 0, 1), 0.0,
                            f"a->b prob with dm^2 == 0 is not 0 with E = {energy}")
            self.assertEqual(barger.calculate_prob(energy, 1, 1), 1.0,
                            f"b->b prob with dm^2 == 0 is not 1 with E = {energy}")
            self.assertEqual(barger.calculate_prob(energy, 1, 0), 0.0,
                            f"b->a prob with dm^2 == 0 is not 0 with E = {energy}")
        
class TestVacuumOscProbs(unittest.TestCase):
    """ check vacuum oscillation probs for some fixed parameters values against externally calculated probs
    """

    # theta = pi/8, m1 = 1, m2 = 2, E = 3, L = 4
    # => prob_(alpha != beta) = sin^2(Pi/4) * sin^2(1) = 0.35403670913
    #    prob_(alpha == beta) =      1 - 0.35403670913 = 0.64596329086

    barger = TwoFlavourBarger()
    energy = 3.0

    barger.set_params(m1=1.0, m2=2.0, theta=m.pi / 8.0, baseline=4.0)

    def test_survuval(self):
        self.assertAlmostEqual(self.barger.calculate_prob(self.energy, 0, 0), 0.64596329086, 6,
                        f"a->a vacuum prob not as expected")
        self.assertAlmostEqual(self.barger.calculate_prob(self.energy, 1, 1), 0.64596329086, 6,
                        f"b->b vacuum prob not as expected")
        
    def test_oscillation(self):
        self.assertAlmostEqual(self.barger.calculate_prob(self.energy, 0, 1), 0.35403670913, 6,
                        f"a->b vacuum prob not as expected")
        self.assertAlmostEqual(self.barger.calculate_prob(self.energy, 1, 0), 0.35403670913, 6,
                        f"b->a vacuum prob not as expected")
        
 
class TestVacuumOscProbs(unittest.TestCase):
    """test matter propagations for some fixed param values 
    """

    ## theta = 0.24, m1 = 0.04eV, m2 = 0.001eV, E = 1GeV, L = 250 km, density = 2
    ## lv = 4pi * E / dm^2 = 7.8588934e+12 
    ## lm = 2pi / ( sqrt(2) * G * density ) = 4.1177454e+13 
    ## gamma = atan( sin( 2theta ) / (cos( 2theta ) - lv / lm) ) / 2.0
    ##       = atan(0.663342 ) / 2 = 0.292848614 rad
    ## dM2 = dm^2 * sqrt( 1 - 2 * (lv / lm) * cos(2theta) + (lv / lm)^2)
    ##     = 0.001599 * sqrt ( 1 - 2 * 0.19085428 * 0.8869949 +  0.19085428 ^2)
    ##     = 0.001335765
    ##
    ## => prob_(alpha != beta) = sin^2(2*gamma) * sin^2( 1.27 (L / E) * dM2 )
    ##                         = sin^2( 2 * 0.292848614 ) * sin^2( 1.27 * 250 / 1 * 0.001599)
    ##                         = 0.0517436
    ##    prob_(alpha == beta) =      1 - 0.0517436  = 0.9482564

    energy = 1.0 * nt.units.GeV
    barger = TwoFlavourBarger()
    barger.set_params(m1=0.04 * nt.units.eV, m2=0.001 * nt.units.eV, theta=0.24,
                        baseline=250.0 * nt.units.km, density=2.0)
    
    def test_vacuum_osc_length(self):
        self.assertAlmostEqual(self.barger.lv(self.energy), 7.8588934e+12, -6,
                               f"bad vacuum osc length")
    
    def test_matter_osc_length(self):
        self.assertAlmostEqual(self.barger.lm(), 4.1177454e+13, -6,
                               f"bad matter osc length")
    
    def test_effective_mixing_angle(self):
        self.assertAlmostEqual(self.barger.calculate_effective_angle(self.energy), 0.292848614, 6,
                               f"bad effective mixing angle")
    
    def test_effective_dm2(self):
        self.assertAlmostEqual(self.barger.calculate_effective_dm2(self.energy), 0.001335765, 6,
                        f"bad effective dm^2")

    def test_survuval(self):
        self.assertAlmostEqual(self.barger.calculate_prob(self.energy, 1, 1), 0.9482564, 3,
                        f"b->b vacuum prob not as expected")
        self.assertAlmostEqual(self.barger.calculate_prob(self.energy, 0, 0), 0.9482564, 3,
                        f"a->a vacuum prob not as expected")
    
    def test_oscillation(self):
        self.assertAlmostEqual(self.barger.calculate_prob(self.energy, 0, 1), 0.0517436, 3,
                        f"a->b vacuum prob not as expected")
        self.assertAlmostEqual(self.barger.calculate_prob(self.energy, 1, 0), 0.0517436, 3,
                        f"b->a vacuum prob not as expected")
