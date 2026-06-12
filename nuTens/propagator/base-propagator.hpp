#pragma once

#include <memory>
#include <nuTens/propagator/base-matter-solver.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <vector>

/// @file propagator.hpp

namespace nuTens
{

class BasePropagator
{
    /*!
     * @class BasePropagator
     * @brief ABC for oscillation probability calculators
     *
     * Any custom propagators should implement this base class
     */

  public:
    /// @brief Constructor
    /// @param nGenerations The number of generations the propagator should
    /// expect
    BasePropagator(int nGenerations) : _nGenerations(nGenerations){};

    /// @brief Destructor
    virtual ~BasePropagator() = default;
    /// @brief copy constructor
    BasePropagator(BasePropagator const &) = default;
    /// @brief copy assignment operator
    BasePropagator &operator=(BasePropagator const &) = default;
    /// @brief move constructor
    BasePropagator(BasePropagator &&) = default;
    /// @brief move assignment operator
    BasePropagator &operator=(BasePropagator &&) = default;

    /// @brief Calculate the oscillation probabilities
    [[nodiscard]] virtual inline Tensor calculateProbs()
    {
        NT_PROFILE();

        return _calculateProbs(_masses.get(), _mixingMatrix);
    }

    /// @name Setters
    /// @{

    /// @brief Set whether we are dealing with anti-neutrinos
    /// @param newValue
    inline BasePropagator &setAntiNeutrino(bool newValue)
    {
        NT_PROFILE();

        _antiNeutrino = newValue;

        return *this;
    }

    /// @brief Set a whole new mixing matrix
    /// @param newMatrix The new matrix to use
    virtual inline BasePropagator &setMixingMatrix(BaseMixingMatrix &newMatrix)
    {
        NT_PROFILE();

        _mixingMatrix = newMatrix;

        return *this;
    }

    /// @brief Set the masses corresponding to the vacuum hamiltonian eigenstates
    /// @param newMasses The new masses to use. This tensor is expected to have a
    /// batch dimension + 1 more dimensions of size nGenerations. The batch
    /// dimension can (and probably should) be 1 and it will be broadcast to
    /// match the batch dimension of the energies supplied to calculateProbs().
    /// So dimension should be {1, nGenerations}.
    virtual inline BasePropagator &setMasses(Tensor &newMasses)
    {
        NT_PROFILE();

        _masses = newMasses;

        return *this;
    }

    /// @}

  private:
    // For calculating with alternate masses and mixing matrix, e.g. if using effective
    // values from massSolver
    [[nodiscard]] virtual Tensor _calculateProbs() = 0;

  protected:
    std::shared_ptr<BaseMixingMatrix> _mixingMatrix;
    std::shared_ptr<Tensor> _masses;
    int _nGenerations;
    bool _antiNeutrino{false};
};

}; // namespace nuTens
