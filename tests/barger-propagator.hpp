#pragma once

#include <array>

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
        double m1, double m2, double m3, 
        double theta12, double theta13, double theta23, 
        double deltaCP,
        double baseline, double density = -999.9f, bool antiNeutrino = false)
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

        // fill the mass array
        masses[0] = _m1;
        masses[1] = _m2;
        masses[2] = _m3;

        // fill the PMNS matrix elements
        pmnsMatrix[0][0] = std::complex<double>(std::cos(theta12) * std::cos(theta13), 0.0);
        pmnsMatrix[0][1] = std::complex<double>(std::sin(theta12) * std::cos(theta13), 0.0);
        pmnsMatrix[0][2] = std::sin(theta13) * std::exp(std::complex<double>(0.0, -1.0) * deltaCP);

        pmnsMatrix[1][0] = -std::sin(theta12) * std::cos(theta23) - std::cos(theta12) * std::sin(theta23) * std::sin(theta13) * std::exp(std::complex<double>(0.0, 1.0) * deltaCP);
        pmnsMatrix[1][1] = std::cos(theta12) * std::cos(theta23)  - std::sin(theta12) * std::sin(theta23) * std::sin(theta13) * std::exp(std::complex<double>(0.0, 1.0) * deltaCP);
        pmnsMatrix[1][2] = std::complex<double>(std::sin(theta23) * std::cos(theta13), 0.0);

        pmnsMatrix[2][0] = std::sin(theta12) * std::sin(theta23)  - std::cos(theta12) * std::cos(theta23) * std::sin(theta13) * std::exp(std::complex<double>(0.0, 1.0) * deltaCP);
        pmnsMatrix[2][1] = -std::cos(theta12) * std::sin(theta23) - std::sin(theta12) * std::cos(theta23) * std::sin(theta13) * std::exp(std::complex<double>(0.0, 1.0) * deltaCP);
        pmnsMatrix[2][2] = std::complex<double>(std::cos(theta23) * std::cos(theta13), 0.0);

    };

    /// calculate the alpha factor used in the eigenvalue computation
    [[nodiscard]] inline double alpha(float energy) const 
    {
        float dmsq12 = _m1 * _m1 - _m2 * _m2;
        float dmsq13 = _m1 * _m1 - _m3 * _m3;

        double ret;
        
        if (_antiNeutrino) {
            ret = - 2.0 * constants::Groot2 * energy * _density + dmsq12 + dmsq13;
        }
        else {
            ret = 2.0 * constants::Groot2 * energy * _density + dmsq12 + dmsq13;
        }

        return ret;
    }

    /// calculate the beta factor used in the eigenvalue computation
    [[nodiscard]] inline double beta(float energy) const 
    {
        double dmsq12 = _m1 * _m1 - _m2 * _m2;
        double dmsq13 = _m1 * _m1 - _m3 * _m3;

        double ret;

        if (_antiNeutrino) {
            ret = (
                dmsq12 * dmsq13 + 
                - 2.0 * constants::Groot2 * energy * _density * (
                    dmsq12 * (1.0 - std::abs(pmnsMatrix[0][1]) * std::abs(pmnsMatrix[0][1])) +
                    dmsq13 * (1.0 - std::abs(pmnsMatrix[0][2]) * std::abs(pmnsMatrix[0][2]))
                )
            );
        }
        else {
            ret = (
                dmsq12 * dmsq13 + 
                2.0 * constants::Groot2 * energy * _density * (
                    dmsq12 * (1.0 - std::abs(pmnsMatrix[0][1]) * std::abs(pmnsMatrix[0][1])) +
                    dmsq13 * (1.0 - std::abs(pmnsMatrix[0][2]) * std::abs(pmnsMatrix[0][2]))
                )
            );
        }

        return ret;
    }

    /// calculate the gamma factor used in the eigenvalue computation
    [[nodiscard]] inline double gamma(float energy) const 
    {
        float dmsq12 = _m1 * _m1 - _m2 * _m2;
        float dmsq13 = _m1 * _m1 - _m3 * _m3;

        double ret;
        if (_antiNeutrino) {
            ret = -2 * constants::Groot2 * energy * _density * dmsq12 * dmsq13 * std::abs(pmnsMatrix[0][0]) * std::abs(pmnsMatrix[0][0]);
        }
        else {
            ret = 2 * constants::Groot2 * energy * _density * dmsq12 * dmsq13 * std::abs(pmnsMatrix[0][0]) * std::abs(pmnsMatrix[0][0]);
        }

        return ret;
    }

    /// calculate effective M^2 values (eigenvalues of the hamiltonian) due to matter effects
    /// @param energy The neutrino energy
    /// @param index The index of the eigenvalue. should be [0-2]
    [[nodiscard]] inline double calculateEffectiveM2(float energy, int index) const 
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

    /// @brief Calculate an element of the hamiltonian
    /// @param energy The neutrino energy
    /// @param k Row
    /// @param j Column
    /// @return Matrix element
    [[nodiscard]] inline std::complex<double> getHamiltonianElement(float energy, int a, int b) const {
        
        std::complex<double> ret = 0.0;

        if ( a == b ) {
            ret += masses[a] * masses[a] / (2.0 * energy);
        }

        if (_antiNeutrino) {
            ret += constants::Groot2 * _density * pmnsMatrix[0][b] * std::conj(pmnsMatrix[0][a]);
        }
        else {
            ret -= constants::Groot2 * _density * pmnsMatrix[0][b] * std::conj(pmnsMatrix[0][a]);
        }

        return ret;
    }

    /// @brief Calculate an element of the "X" transition matrix matrix (equation 11 in Barger et al)
    /// @param energy The neutrino energy
    /// @param a Row
    /// @param b Column
    /// @return Matrix element
    [[nodiscard]] inline std::complex<double> getTransitionMatrixElement(double energy, int a, int b) const {

        std::complex<double> ret = 0.0;

        // interpret density <= 0.0 as vacuum, then transition matrix
        // is just matrix with exponential terms along diagonal
        if (_density <= 0.0 ) {
            
            if ( a == b )
                ret = std::exp(- 0.5 * std::complex<double>(0.0, 1.0) * masses[a] * masses[a] * _baseline * 2.0 * M_PI / energy);
        
            else 
                ret = 0.0;
        }

        else {
            for (int k = 0; k < 3; k++) {

                std::complex<double> numerator = 4.0 * energy * energy * (
                    getHamiltonianElement(energy, a, 0) * getHamiltonianElement(energy, 0, b) +
                    getHamiltonianElement(energy, a, 1) * getHamiltonianElement(energy, 1, b) +
                    getHamiltonianElement(energy, a, 2) * getHamiltonianElement(energy, 2, b) 
                );

                std::complex<double> constant = 1.0;
                std::complex<double> denominator = 1.0;

                for (int j = 0; j < 3; j++) {

                    if (j == k) continue;

                    numerator -= 2.0 * energy * getHamiltonianElement(energy, a, b) * calculateEffectiveM2(energy, j);
                    denominator *= calculateEffectiveM2(energy, k) - calculateEffectiveM2(energy, j);
                    constant *= calculateEffectiveM2(energy, j);

                }

                std::complex<double> prod = numerator;

                if (a == b) 
                    prod += constant;
            
                prod /= denominator;

                std::complex<double> exponential = std::exp(-std::complex<double>(0.0, 1.0) * calculateEffectiveM2(energy, k) * _baseline * 2.0 * M_PI / (2.0 * energy));

                ret += prod * exponential;
            }
        }

        return ret;
    }

    /// @brief calculate oscillation probability from flavour alpha to flavour beta
    /// @param energy neutrino energy
    /// @param alpha initial flavour index
    /// @param beta final flavour index
    /// @return oscillation probability
    [[nodiscard]] inline double calculateProb(float energy, int alpha, int beta) const {

        std::complex<double> ret = 0.0;

        for (int i = 0; i < 3; i++) {

            for (int j = 0; j < 3; j++) {
            
                if (_antiNeutrino) {
                    ret += std::conj(pmnsMatrix[beta][i] * getTransitionMatrixElement(energy, i, j)) * pmnsMatrix[alpha][j];
                }
                else {
                    ret += pmnsMatrix[alpha][i] * getTransitionMatrixElement(energy, i, j) * std::conj(pmnsMatrix[beta][j]);
                }

            }
        
        }

        return std::abs(ret) * std::abs(ret);
    }
    

  private:
    // oscillation parameters
    double _m1;
    double _m2;
    double _m3;
    double _theta12;
    double _theta13;
    double _theta23;
    double _deltaCP;

    // other parameters
    double _baseline;
    double _density;

    // anti-neutrino flag
    bool _antiNeutrino;

    std::array<std::array<std::complex<double>, 3>, 3> pmnsMatrix;
    std::array<double, 3> masses;
};

} // namespace testing

} // namespace nuTens