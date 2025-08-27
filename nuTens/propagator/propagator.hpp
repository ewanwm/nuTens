#pragma once

#include <memory>
#include <nuTens/propagator/base-matter-solver.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <vector>

/// @file propagator.hpp

namespace nuTens
{

class Propagator
{
    /*!
     * @class Propagator
     * @brief Neutrino oscillation probability calculator
     *
     * This class is used to propagate neutrinos over some baseline and calculate
     * the probability that they will oscillate to another flavour. A Propagator
     * can be configured using the Setters by assigning parameters (neutrino
     * masses and mixing matrix elements). You can assign a matter solver (a
     * derivative of BaseMatterSolver) to deal with matter effects using
     * setMatterSolver(). calculateProbs() can then be used to calculate energy
     * dependent oscillation probabilities.
     *
     * (The specifics of this interface may change in the future)
     */

  public:
    /// @brief Constructor
    /// @param nGenerations The number of generations the propagator should
    /// expect
    /// @param baseline The baseline to propagate over
    Propagator(int nGenerations, float baseline, bool antiNeutrino = false)
        : _baseline(baseline), _nGenerations(nGenerations), _antiNeutrino(antiNeutrino) {};

    /// @brief Calculate the oscillation probabilities
    /// @param energies The energies of the neutrinos
    [[nodiscard]] virtual Tensor calculateProbs();

    /// @name Setters
    /// @{

    /// @brief Set whether we are dealing with anti-neutrinos
    /// @param newValue
    virtual inline void setAntiNeutrino(bool newValue)
    {
        NT_PROFILE();

        _antiNeutrino = newValue;

        if (_matterSolver)
        {
            _matterSolver->setAntiNeutrino(newValue);
        }
    }

    /// @brief Set a matter solver to use to deal with matter effects
    /// @param newSolver A derivative of BaseMatterSolver
    /// @warning Should be called *before* setMixingMatrix and setMasses
    virtual inline void setMatterSolver(const std::shared_ptr<BaseMatterSolver> &newSolver)
    {
        NT_PROFILE();
        _matterSolver = newSolver;
    }

    /// \todo Should add a check to tensors supplied to the setters to see how
    /// many dimensions they have, and if missing a batch dimension, add one.

    /// @brief Set the neutrino energies
    /// @param newEnergies The neutrino energies
    virtual void setEnergies(Tensor &newEnergies)
    {
        NT_PROFILE();

        _energies = newEnergies;
        _weightMatrix = Tensor::ones({_energies.getBatchDim(), _nGenerations, _nGenerations}, dtypes::kComplexFloat)
                            .requiresGrad(false);
        _weightArgDenom =
            Tensor::scale(Tensor::scale(_energies, 2.0),
                          std::complex<float>(1.0) / (std::complex<float>(-1.0J) * _baseline * 2.0f * (float)M_PI));

        if (_matterSolver)
        {
            _matterSolver->setEnergies(newEnergies);
        }
    }

    /// @brief Set the masses corresponding to the vacuum hamiltonian eigenstates
    /// @param newMasses The new masses to use. This tensor is expected to have a
    /// batch dimension + 1 more dimensions of size nGenerations. The batch
    /// dimension can (and probably should) be 1 and it will be broadcast to
    /// match the batch dimension of the energies supplied to calculateProbs().
    /// So dimension should be {1, nGenerations}.
    virtual void setMasses(Tensor &newMasses)
    {
        NT_PROFILE();

        _masses = newMasses;
        if (_matterSolver != nullptr)
        {
            _matterSolver->setMasses(newMasses);
        }
    }

    /// @brief Set a whole new mixing matrix
    /// @param newMatrix The new matrix to use
    virtual inline void setMixingMatrix(Tensor &newMatrix)
    {
        NT_PROFILE();
        _mixingMatrix = newMatrix;
        if (_matterSolver != nullptr)
        {
            _matterSolver->setMixingMatrix(newMatrix);
        }
    }

    /// \todo add setMixingMatrix(const std::vector<int> &indices, float value) methods
    /// to BaseMatterSolver? maybe have these setters in a base class of both
    /// Propagator and BaseMatterSolver ??

    /// @brief Set a single element of the mixing matrix
    /// @param indices The index of the value to set
    /// @param value The new value
    inline void setMixingMatrix(const std::vector<int> &indices, float value)
    {
        NT_PROFILE();
        _mixingMatrix.setValue(indices, value);
    }

    /// @brief Set a single element of the mixing matrix
    /// @param indices The index of the value to set
    /// @param value The new value
    inline void setMixingMatrix(const std::vector<int> &indices, std::complex<float> value)
    {
        NT_PROFILE();
        _mixingMatrix.setValue(indices, value);
    }

    /// @brief Set the baseline
    /// @param newBaseline new value
    virtual inline void setBaseline(float newBaseline)
    {

        NT_PROFILE();

        _baseline = newBaseline;

        _weightArgDenom = Tensor::scale(Tensor::scale(_energies, 2.0),
                                        std::complex<float>(1.0) / (std::complex<float>(-1.0J) * _baseline));
    }

    /// @}

    /// @{ Getters

    [[nodiscard]] inline float getBaseline() const
    {
        return _baseline;
    }

    /// @}

  private:
    // For calculating with alternate masses and mixing matrix, e.g. if using effective
    // values from massSolver
    [[nodiscard]] Tensor _calculateProbs(const Tensor &masses, const Tensor &mixingMatrix);

  protected:
    Tensor _mixingMatrix;
    Tensor _masses;
    Tensor _energies;
    Tensor _weightMatrix;
    Tensor _weightArgDenom;
    int _nGenerations;
    float _baseline;
    bool _antiNeutrino;

    std::shared_ptr<BaseMatterSolver> _matterSolver;
};

}; // namespace nuTens
