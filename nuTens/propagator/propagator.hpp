#pragma once

#include <memory>
#include <nuTens/propagator/base-matter-solver.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <vector>

/// @file propagator.hpp

class Propagator
{
    /*!
     * @class Propagator
     * @brief Neutrino oscillation probability calculator
     *
     * This class is used to propagate neutrinos over some baseline and calculate
     * the probability that they will oscillate to another flavour. A Propagator
     * can be configured using the Setters by assigning parameters (neutrino
     * masses and PMNS matrix elements). You can assign a matter solver (a
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
    Propagator(int nGenerations, float baseline) : _baseline(baseline), _nGenerations(nGenerations){};

    /// @brief Calculate the oscillation probabilities
    /// @param energies The energies of the neutrinos
    [[nodiscard]] Tensor calculateProbs(const Tensor &energies) const;

    /// @name Setters
    /// @{

    /// @brief Set a matter solver to use to deal with matter effects
    /// @param newSolver A derivative of BaseMatterSolver
    inline void setMatterSolver(std::shared_ptr<BaseMatterSolver> &newSolver)
    {
        NT_PROFILE();
        _matterSolver = std::move(newSolver);
        _matterSolver->setMasses(_masses);
        _matterSolver->setPMNS(_pmnsMatrix);
    }

    /// \todo Should add a check to tensors supplied to the setters to see how
    /// many dimensions they have, and if missing a batch dimension, add one.

    /// @brief Set the masses corresponding to the vacuum hamiltonian eigenstates
    /// @param newMasses The new masses to use. This tensor is expected to have a
    /// batch dimension + 1 more dimensions of size nGenerations. The batch
    /// dimension can (and probably should) be 1 and it will be broadcast to
    /// match the batch dimension of the energies supplied to calculateProbs().
    /// So dimension should be {1, nGenerations}.
    void setMasses(Tensor &newMasses)
    {
        _masses = newMasses;
        if (_matterSolver != nullptr)
        {
            _matterSolver->setMasses(newMasses);
        }
    }

    /// @brief Set a whole new PMNS matrix
    /// @param newPMNS The new matrix to use
    inline void setPMNS(Tensor &newPMNS)
    {
        NT_PROFILE();
        _pmnsMatrix = newPMNS;
        if (_matterSolver != nullptr)
        {
            _matterSolver->setPMNS(newPMNS);
        }
    }

    /// \todo add setPMNS(const std::vector<int> &indices, float value) methods
    /// to BaseMatterSolver? maybe have these setters in a base class of both
    /// Propagator and BaseMatterSolver ??

    /// @brief Set a single element of the PMNS matrix
    /// @param indices The index of the value to set
    /// @param value The new value
    inline void setPMNS(const std::vector<int> &indices, float value)
    {
        NT_PROFILE();
        _pmnsMatrix.setValue(indices, value);
    }

    /// @brief Set a single element of the PMNS matrix
    /// @param indices The index of the value to set
    /// @param value The new value
    inline void setPMNS(const std::vector<int> &indices, std::complex<float> value)
    {
        NT_PROFILE();
        _pmnsMatrix.setValue(indices, value);
    }

    /// @}

  private:
    // For calculating with alternate masses and PMNS, e.g. if using effective
    // values from massSolver
    [[nodiscard]] Tensor _calculateProbs(const Tensor &energies, const Tensor &masses, const Tensor &PMNS) const;

  private:
    Tensor _pmnsMatrix;
    Tensor _masses;
    int _nGenerations;
    float _baseline;

    std::shared_ptr<BaseMatterSolver> _matterSolver;
};