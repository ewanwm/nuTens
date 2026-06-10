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

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
namespace nuTens::testing
{

template <typename T = float> class TwoFlavourBarger
{
  public:
    /// @{ Set parameters of the propagator
    inline TwoFlavourBarger &setMass1(T mass1)
    {
        _mass1 = mass1;
        return *this;
    }
    inline TwoFlavourBarger &setMass2(T mass2)
    {
        _mass2 = mass2;
        return *this;
    }
    inline TwoFlavourBarger &setTheta(T theta)
    {
        _theta = theta;
        return *this;
    }
    inline TwoFlavourBarger &setBaseline(T baseline)
    {
        _baseline = baseline;
        return *this;
    }
    inline TwoFlavourBarger &setDensity(T density)
    {
        _density = density;
        return *this;
    }
    inline TwoFlavourBarger &setAntiNeutrino(bool antiNeutrino)
    {
        _antiNeutrino = antiNeutrino;
        return *this;
    }
    /// @}

    // characteristic length in vacuum
    [[nodiscard]] inline T lVac(T energy) const
    {
        return 4.0 * M_PI * energy / (_mass1 * _mass1 - _mass2 * _mass2);
    }

    // characteristic length in matter
    [[nodiscard]] inline T lMatter() const
    {
        T lMatter = constants::twoPi / (nuTens::constants::Groot2 * _density);

        // for anti-neutrinos, sign of lMatter is reversed
        if (_antiNeutrino)
        {
            lMatter *= -1;
        }

        return lMatter;
    }

    // calculate the modified rotation angle
    [[nodiscard]] inline T calculateEffectiveAngle(T energy) const
    {
        T ret = NAN;

        if (_density > 0.0)
        {
            ret = std::atan2(std::sin(2.0 * _theta), (std::cos(2.0 * _theta) - lVac(energy) / lMatter())) / 2.0;
        }
        else
        {
            ret = _theta;
        }

        return ret;
    }

    // calculate the modified delta M^2
    [[nodiscard]] inline T calculateEffectiveDm2(T energy) const
    {
        T ret = NAN;

        if (_density > 0.0)
        {
            ret = (_mass1 * _mass1 - _mass2 * _mass2) *
                  std::sqrt(1.0 - 2.0 * (lVac(energy) / lMatter()) * std::cos(2.0 * _theta) +
                            (lVac(energy) / lMatter()) * (lVac(energy) / lMatter()));
        }
        else
        {
            ret = (_mass1 * _mass1 - _mass2 * _mass2);
        }

        return ret;
    }

    // get the good old 2 flavour mixing matrix entries
    [[nodiscard]] inline T getPMNSelement(T energy, int alpha, int beta) const
    {
        // LCOV_EXCL_START
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
        // LCOV_EXCL_STOP

        T ret = NAN;

        T gamma = calculateEffectiveAngle(energy);

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
        // LCOV_EXCL_START
        else
        {
            std::cerr << "ERROR: how did you get here????" << std::endl;
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
            throw;
        }
        // LCOV_EXCL_STOP

        return ret;
    }

    // get the good old 2 flavour vacuum oscillation probability
    [[nodiscard]] inline T calculateProb(T energy, int alpha, int beta) const
    {
        // LCOV_EXCL_START
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
        // LCOV_EXCL_STOP

        T ret = NAN;

        // get the effective oscillation parameters
        // if in vacuum (_density <= 0.0) these should just return the "raw" values
        T gamma = calculateEffectiveAngle(energy);
        T dM2 = calculateEffectiveDm2(energy);

        // now get the actual probabilities
        T sin2Gamma = std::sin(2.0 * gamma);
        T sinPhi = std::sin(dM2 * constants::twoPi * _baseline / (4.0 * energy));

        T offAxis = sin2Gamma * sin2Gamma * sinPhi * sinPhi;
        T onAxis = 1.0 - offAxis;

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
    T _mass1 = NAN;
    T _mass2 = NAN;
    T _theta = NAN;

    // characteristic lengths in vacuum and matter
    T _lv = NAN;
    T _lm = NAN;

    // other parameters
    T _baseline = NAN;
    T _density = NAN;

    // anti-neutrino flag
    bool _antiNeutrino = false;
};

template <typename T = float> class ThreeFlavourBarger
{
  public:
    // set the parameters of this propagator
    // negative density values will be interpreted as propagating in vacuum
    inline void setParams(T mass1, T mass2, T mass3, T theta12, T theta13, T theta23, T deltaCP, T baseline,
                          T density = -999.9, bool antiNeutrino = false)
    {
        _mass1 = mass1;
        _mass2 = mass2;
        _mass3 = mass3;
        _theta12 = theta12;
        _theta13 = theta13;
        _theta23 = theta23;
        _deltaCP = deltaCP;
        _baseline = baseline;
        _density = density;
        _antiNeutrino = antiNeutrino;

        // fill the mass array
        masses[0] = _mass1;
        masses[1] = _mass2;
        masses[2] = _mass3;

        // fill the PMNS matrix elements
        pmnsMatrix[0][0] = std::complex<T>(std::cos(theta12) * std::cos(theta13), 0.0);
        pmnsMatrix[0][1] = std::complex<T>(std::sin(theta12) * std::cos(theta13), 0.0);
        pmnsMatrix[0][2] = std::sin(theta13) * std::exp(std::complex<T>(0.0, -1.0) * deltaCP);

        pmnsMatrix[1][0] = -std::sin(theta12) * std::cos(theta23) - std::cos(theta12) * std::sin(theta23) *
                                                                        std::sin(theta13) *
                                                                        std::exp(std::complex<T>(0.0, 1.0) * deltaCP);
        pmnsMatrix[1][1] = std::cos(theta12) * std::cos(theta23) - std::sin(theta12) * std::sin(theta23) *
                                                                       std::sin(theta13) *
                                                                       std::exp(std::complex<T>(0.0, 1.0) * deltaCP);
        pmnsMatrix[1][2] = std::complex<T>(std::sin(theta23) * std::cos(theta13), 0.0);

        pmnsMatrix[2][0] = std::sin(theta12) * std::sin(theta23) - std::cos(theta12) * std::cos(theta23) *
                                                                       std::sin(theta13) *
                                                                       std::exp(std::complex<T>(0.0, 1.0) * deltaCP);
        pmnsMatrix[2][1] = -std::cos(theta12) * std::sin(theta23) - std::sin(theta12) * std::cos(theta23) *
                                                                        std::sin(theta13) *
                                                                        std::exp(std::complex<T>(0.0, 1.0) * deltaCP);
        pmnsMatrix[2][2] = std::complex<T>(std::cos(theta23) * std::cos(theta13), 0.0);
    };

    /// calculate the alpha factor used in the eigenvalue computation
    [[nodiscard]] inline T calculateAlpha(T energy) const
    {
        T dmsq12 = _mass1 * _mass1 - _mass2 * _mass2;
        T dmsq13 = _mass1 * _mass1 - _mass3 * _mass3;

        T ret = NAN;

        if (_antiNeutrino)
        {
            ret = -2.0 * constants::Groot2 * energy * _density + dmsq12 + dmsq13;
        }
        else
        {
            ret = 2.0 * constants::Groot2 * energy * _density + dmsq12 + dmsq13;
        }

        return ret;
    }

    /// calculate the beta factor used in the eigenvalue computation
    [[nodiscard]] inline T calculateBeta(T energy) const
    {
        T dmsq12 = _mass1 * _mass1 - _mass2 * _mass2;
        T dmsq13 = _mass1 * _mass1 - _mass3 * _mass3;

        T ret = NAN;

        if (_antiNeutrino)
        {
            ret = (dmsq12 * dmsq13 + -2.0 * constants::Groot2 * energy * _density *
                                         (dmsq12 * (1.0 - std::abs(pmnsMatrix[0][1]) * std::abs(pmnsMatrix[0][1])) +
                                          dmsq13 * (1.0 - std::abs(pmnsMatrix[0][2]) * std::abs(pmnsMatrix[0][2]))));
        }
        else
        {
            ret = (dmsq12 * dmsq13 + 2.0 * constants::Groot2 * energy * _density *
                                         (dmsq12 * (1.0 - std::abs(pmnsMatrix[0][1]) * std::abs(pmnsMatrix[0][1])) +
                                          dmsq13 * (1.0 - std::abs(pmnsMatrix[0][2]) * std::abs(pmnsMatrix[0][2]))));
        }

        return ret;
    }

    /// calculate the gamma factor used in the eigenvalue computation
    [[nodiscard]] inline T calculateGamma(T energy) const
    {
        T dmsq12 = _mass1 * _mass1 - _mass2 * _mass2;
        T dmsq13 = _mass1 * _mass1 - _mass3 * _mass3;

        T ret = NAN;
        if (_antiNeutrino)
        {
            ret = -2 * constants::Groot2 * energy * _density * dmsq12 * dmsq13 * std::abs(pmnsMatrix[0][0]) *
                  std::abs(pmnsMatrix[0][0]);
        }
        else
        {
            ret = 2 * constants::Groot2 * energy * _density * dmsq12 * dmsq13 * std::abs(pmnsMatrix[0][0]) *
                  std::abs(pmnsMatrix[0][0]);
        }

        return ret;
    }

    /// calculate effective M^2 values (eigenvalues of the hamiltonian) due to matter effects
    /// @param energy The neutrino energy
    /// @param index The index of the eigenvalue. should be [0-2]
    [[nodiscard]] inline T calculateEffectiveM2(T energy, int index) const
    {
        T alpha = calculateAlpha(energy);
        T beta = calculateBeta(energy);
        T gamma = calculateGamma(energy);

        // calculate argument of arccos
        T arg = (2.0 * std::pow(alpha, 3) - 9.0 * alpha * beta + 27.0 * gamma) /
                (2.0 * std::pow(alpha * alpha - 3.0 * beta, 3.0 / 2.0));

        // calculate the coefficient of the cos term
        T coeff = -(2.0 / 3.0) * std::sqrt(alpha * alpha - 3.0 * beta);

        return coeff * std::cos((1.0 / 3.0) * (std::acos(arg) + index * constants::twoPi)) + _mass1 * _mass1 -
               alpha / 3.0;
    }

    /// @brief Calculate an element of the hamiltonian
    /// @param energy The neutrino energy
    /// @param idx1 Row
    /// @param idx2 Column
    /// @return Matrix element
    [[nodiscard]] inline std::complex<T> getHamiltonianElement(T energy, int idx1, int idx2) const
    {

        std::complex<T> ret = 0.0;

        if (idx1 == idx2)
        {
            ret += masses.at(idx1) * masses.at(idx1) / (2.0 * energy);
        }

        if (_antiNeutrino)
        {
            ret += static_cast<T>(constants::Groot2) * _density * pmnsMatrix[0].at(idx2) *
                   std::conj(pmnsMatrix[0].at(idx1));
        }
        else
        {
            ret -= static_cast<T>(constants::Groot2) * _density * pmnsMatrix[0].at(idx2) *
                   std::conj(pmnsMatrix[0].at(idx1));
        }

        return ret;
    }

    /// @brief Calculate an element of the "X" transition matrix matrix (equation 11 in Barger et al)
    /// @param energy The neutrino energy
    /// @param idx1 Row
    /// @param idx2 Column
    /// @return Matrix element
    [[nodiscard]] inline std::complex<T> getTransitionMatrixElement(T energy, int idx1, int idx2) const
    {

        std::complex<T> ret = 0.0;

        // interpret density <= 0.0 as vacuum, then transition matrix
        // is just matrix with exponential terms along diagonal
        if (_density <= 0.0)
        {

            if (idx1 == idx2)
            {
                ret = std::exp(-(T)0.5 * std::complex<T>(0.0, 1.0) * masses.at(idx1) * masses.at(idx1) * _baseline *
                               (T)constants::twoPi / energy);
            }
            else
            {
                ret = 0.0;
            }
        }

        else
        {
            for (int k = 0; k < 3; k++)
            {

                std::complex<T> numerator =
                    (T)4.0 * energy * energy *
                    (getHamiltonianElement(energy, idx1, 0) * getHamiltonianElement(energy, 0, idx2) +
                     getHamiltonianElement(energy, idx1, 1) * getHamiltonianElement(energy, 1, idx2) +
                     getHamiltonianElement(energy, idx1, 2) * getHamiltonianElement(energy, 2, idx2));

                std::complex<T> constant = 1.0;
                std::complex<T> denominator = 1.0;

                for (int j = 0; j < 3; j++)
                {

                    if (j == k)
                    {
                        continue;
                    }

                    numerator -=
                        (T)2.0 * energy * getHamiltonianElement(energy, idx1, idx2) * calculateEffectiveM2(energy, j);
                    denominator *= calculateEffectiveM2(energy, k) - calculateEffectiveM2(energy, j);
                    constant *= calculateEffectiveM2(energy, j);
                }

                std::complex<T> prod = numerator;

                if (idx1 == idx2)
                {
                    prod += constant;
                }

                prod /= denominator;

                std::complex<T> exponential = std::exp(-std::complex<T>(0.0, 1.0) * calculateEffectiveM2(energy, k) *
                                                       _baseline * (T)2.0 * (T)M_PI / ((T)2.0 * energy));

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
    [[nodiscard]] inline T calculateProb(T energy, int alpha, int beta) const
    {

        std::complex<T> ret = 0.0;

        for (int i = 0; i < 3; i++)
        {

            for (int j = 0; j < 3; j++)
            {

                if (_antiNeutrino)
                {
                    ret += std::conj(pmnsMatrix.at(beta)[i] * getTransitionMatrixElement(energy, i, j)) *
                           pmnsMatrix.at(alpha)[j];
                }
                else
                {
                    ret += pmnsMatrix.at(alpha)[i] * getTransitionMatrixElement(energy, i, j) *
                           std::conj(pmnsMatrix.at(beta)[j]);
                }
            }
        }

        return std::abs(ret) * std::abs(ret);
    }

  private:
    // oscillation parameters
    T _mass1 = NAN;
    T _mass2 = NAN;
    T _mass3 = NAN;
    T _theta12 = NAN;
    T _theta13 = NAN;
    T _theta23 = NAN;
    T _deltaCP = NAN;

    // other parameters
    T _baseline = NAN;
    T _density = NAN;

    // anti-neutrino flag
    bool _antiNeutrino = false;

    std::array<std::array<std::complex<T>, 3>, 3> pmnsMatrix{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
    std::array<T, 3> masses{{0.0, 0.0, 0.0}};
};

} // namespace nuTens::testing

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)