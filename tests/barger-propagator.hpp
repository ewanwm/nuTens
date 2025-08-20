#pragma once

#include <cmath>
#include <complex>

#include <iostream>
#include <nuTens/propagator/constants.hpp>

// make a simple propagator based on Barger:
// [Vernon D. Barger, K. Whisnant, S. Pakvasa, and R. J. N. Phillips. Matter
// Effects on Three-Neutrino Oscillations. Phys. Rev. D, 22:2718, 1980.] for
// the purposes of testing so we can compare our propagators against something
// solid.

// doesn't need to be very efficient or very fancy.
// goal is more to be clear and simple so we can be confident in the results.
// also don't want to use any fancy classes like tensors,
// just want to use vv simple c++ standard objects so is independent of the
// rest of the nuTens libraries.

namespace nuTens
{
namespace testing
{

class TwoFlavourBarger
{
  public:
    // set the parameters of this propagator
    // negative density values will be interpreted as propagating in vacuum
    inline void setParams(float m1, float m2, float theta, float baseline, float density = -999.9, bool antiNeutrino = false)
    {
        _m1 = m1;
        _m2 = m2;
        _theta = theta;
        _baseline = baseline;
        _density = density;
        _antiNeutrino = antiNeutrino;
    };

    // characteristic length in vacuum
    [[nodiscard]] inline float lv(float energy) const
    {
        return 4.0 * M_PI * energy / (_m1 * _m1 - _m2 * _m2);
    }

    // characteristic length in matter
    [[nodiscard]] inline float lm() const
    {
        float lm = 2.0 * M_PI / (nuTens::constants::Groot2 * _density);
    
        // for anti-neutrinos, sign of lm is reversed
        if (_antiNeutrino)
        {
            lm *= -1;
        }

        return lm;
    }

    // calculate the modified rotation angle
    [[nodiscard]] inline float calculateEffectiveAngle(float energy) const
    {
        float ret = NAN;

        if (_density > 0.0)
        {
            ret = std::atan2(std::sin(2.0 * _theta), (std::cos(2.0 * _theta) - lv(energy) / lm())) / 2.0;
        }
        else
        {
            ret = _theta;
        }

        return ret;
    }

    // calculate the modified delta M^2
    [[nodiscard]] inline float calculateEffectiveDm2(float energy) const
    {
        float ret = NAN;

        if (_density > 0.0)
        {
            ret = (_m1 * _m1 - _m2 * _m2) * std::sqrt(1.0 - 2.0 * (lv(energy) / lm()) * std::cos(2.0 * _theta) +
                                                      (lv(energy) / lm()) * (lv(energy) / lm()));
        }
        else
        {
            ret = (_m1 * _m1 - _m2 * _m2);
        }

        return ret;
    }

    // get the good old 2 flavour mixing matrix entries
    [[nodiscard]] inline float getPMNSelement(float energy, int alpha, int beta) const
    {
        if ((alpha > 1 || alpha < 0) || (beta > 1 || beta < 0))
        {
            std::cerr << "ERROR: TwoFlavourBarger class only supports flavour "
                         "indices of 0 or 1"
                      << std::endl;
            std::cerr << "       you supplied alpha = " << alpha << ", "
                      << "beta = " << beta << std::endl;
            std::cerr << "       " << __FILE__ << ": " << __LINE__ << std::endl;

            throw;
        }

        float ret = NAN;

        float gamma = calculateEffectiveAngle(energy);

        // on diagonal elements
        if (alpha == 0 && beta == 0 || alpha == 1 && beta == 1)
        {
            ret = std::cos(gamma);
        }
        // off diagonal elements
        else if (alpha == 0 && beta == 1)
        {
            ret = std::sin(gamma);
        }
        else if (alpha == 1 && beta == 0)
        {
            ret = -std::sin(gamma);
        }

        // should be caught at start of function but just in case...
        else
        {
            std::cerr << "ERROR: how did you get here????" << std::endl;
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
            throw;
        }

        return ret;
    }

    // get the good old 2 flavour vacuum oscillation probability
    [[nodiscard]] inline float calculateProb(float energy, int alpha, int beta) const
    {
        if ((alpha > 1 || alpha < 0) || (beta > 1 || beta < 0))
        {
            std::cerr << "ERROR: TwoFlavourBarger class only supports flavour "
                         "indices of 0 or 1"
                      << std::endl;
            std::cerr << "       you supplied alpha = " << alpha << ", "
                      << "beta = " << beta << std::endl;
            std::cerr << "       " << __FILE__ << ": " << __LINE__ << std::endl;
            throw;
        }

        float ret = NAN;

        // get the effective oscillation parameters
        // if in vacuum (_density <= 0.0) these should just return the "raw" values
        float gamma = calculateEffectiveAngle(energy);
        float dM2 = calculateEffectiveDm2(energy);

        // now get the actual probabilities
        float sin2Gamma = std::sin(2.0 * gamma);
        float sinPhi = std::sin(dM2 * 2.0 * M_PI * _baseline / (4.0 * energy));

        float offAxis = sin2Gamma * sin2Gamma * sinPhi * sinPhi;
        float onAxis = 1.0 - offAxis;

        if (alpha == beta)
        {
            ret = onAxis;
        }
        else
        {
            ret = offAxis;
        }

        return ret;
    }

  private:
    // oscillation parameters
    float _m1;
    float _m2;
    float _theta;

    // characteristic lengths in vacuum and matter
    float _lv;
    float _lm;

    // other parameters
    float _baseline;
    float _density;

    // anti-neutrino flag
    bool _antiNeutrino;
};

class ThreeFlavourBarger 
{
  public:

    // set the parameters of this propagator
    // negative density values will be interpreted as propagating in vacuum
    inline void setParams(
        float m1, float m2, float m3, 
        float theta12, float theta13, float theta23, 
        float deltaCP,
        float baseline, float density = -999.9f, bool antiNeutrino = false)
    {
        _m1 = m1;
        _m2 = m2;
        _m3 = m3;
        _theta12 = theta12;
        _theta13 = theta13;
        _theta23 = theta23;
        _deltaCP = deltaCP;
        _baseline = baseline;
        _density = density;
        _antiNeutrino = antiNeutrino;
    };

    /// calculate the alpha factor used in the eigenvalue computation
    [[nodiscard]] inline float alpha(float energy) const 
    {
        float dmsq12 = _m1 * _m1 - _m2 * _m2;
        float dmsq13 = _m1 * _m1 - _m3 * _m3;

        return 2.0 * constants::Groot2 * energy * _density + dmsq12 + dmsq13;
    }

    /// calculate the beta factor used in the eigenvalue computation
    [[nodiscard]] inline float beta(float energy) const 
    {
        float               Ue2 = std::sin(_theta12) * std::cos(_theta13);
        std::complex<float> Ue3 = std::sin(_theta13) * std::exp(-std::complex<float>(0.0, 1.0) * _deltaCP);
        float dmsq12 = _m1 * _m1 - _m2 * _m2;
        float dmsq13 = _m1 * _m1 - _m3 * _m3;

        return (
            dmsq12 * dmsq13 + 
            2.0 * constants::Groot2 * energy * _density * (
                dmsq12 * (1.0 - Ue2 * Ue2) +
                dmsq13 * (1.0 - std::abs(Ue3) * std::abs(Ue3))
            )
        );
    }

    /// calculate the gamma factor used in the eigenvalue computation
    [[nodiscard]] inline float gamma(float energy) const 
    {
        float Ue1 = std::cos(_theta12) * std::cos(_theta13);
        float dmsq12 = _m1 * _m1 - _m2 * _m2;
        float dmsq13 = _m1 * _m1 - _m3 * _m3;

        return 2 * constants::Groot2 * energy * _density * dmsq12 * dmsq13 * Ue1 * Ue1;
    }

    /// calculate effective M^2 values (eigenvalues of the hamiltonian) due to matter effects
    /// @param energy The neutrino energy
    /// @param index The index of the eigenvalue. should be [0-2]
    [[nodiscard]] inline float calculateEffectiveM2(float energy, int index) const 
    {
        float a = alpha(energy);
        float b = beta(energy);
        float c = gamma(energy);

        // calculate argument of arccos
        float arg = (2.0 * a*a*a - 9.0 * a*b + 27.0 * c) / ( 2.0 * std::pow( a*a - 3.0 * b, 3.0 / 2.0) ); 

        // calculate the coefficient of the cos term
        float coeff = - (2.0 / 3.0) * std::sqrt( a*a - 3.0 * b );

        return coeff * std::cos( ( 1.0 / 3.0 ) * ( std::acos(arg) + index * 2.0 * M_PI ) ) + _m1 * _m1 - a / 3.0;
    }

  private:
    // oscillation parameters
    float _m1;
    float _m2;
    float _m3;
    float _theta12;
    float _theta13;
    float _theta23;
    float _deltaCP;

    // other parameters
    float _baseline;
    float _density;

    // anti-neutrino flag
    bool _antiNeutrino;
};

} // namespace testing

} // namespace nuTens