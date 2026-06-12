#pragma once

#include <memory>
#include <nuTens/propagator/base-matter-solver.hpp>
#include <nuTens/propagator/base-mixing-matrix.hpp>
#include <nuTens/propagator/base-propagator.hpp>
#include <nuTens/propagator/constants.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <vector>

/// @file propagator.hpp

namespace nuTens
{

class Propagator : public BasePropagator
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
    Propagator(int nGenerations) : BasePropagator(nGenerations){};

    /// @brief Destructor
    virtual ~Propagator() = default;
    /// @brief copy constructor
    Propagator(Propagator const &) = default;
    /// @brief copy assignment operator
    Propagator &operator=(Propagator const &) = default;
    /// @brief move constructor
    Propagator(Propagator &&) = default;
    /// @brief move assignment operator
    Propagator &operator=(Propagator &&) = default;

    /// @name Setters
    /// @{

    /// @brief Set a matter solver to use to deal with matter effects
    /// @param newSolver A derivative of BaseMatterSolver
    /// @warning Should be called *before* setMixingMatrix and setMasses
    virtual inline Propagator &setMatterSolver(const std::shared_ptr<BaseMatterSolver> &newSolver)
    {
        NT_PROFILE();
        _matterSolver = newSolver;

        _matterSolver->setAntiNeutrino(_antiNeutrino);

        if (_energies)
        {
            _matterSolver->setEnergies(_energies);
        }
        if (_masses)
        {
            _matterSolver->setMasses(_masses);
        }

        return *this;
    }

    /// @brief Set whether we are dealing with anti-neutrinos
    /// @param newValue
    inline Propagator &setAntiNeutrino(bool newValue)
    {
        NT_PROFILE();

        _antiNeutrino = newValue;

        if (_matterSolver)
        {
            _matterSolver->setAntiNeutrino(newValue);
        }

        return *this;
    }

    /// \todo Should add a check to tensors supplied to the setters to see how
    /// many dimensions they have, and if missing a batch dimension, add one.

    /// @brief Set the neutrino energies
    /// @param newEnergies The neutrino energies
    virtual inline Propagator &setEnergies(Tensor &newEnergies)
    {
        NT_PROFILE();

        _energies = newEnergies;

        _weightMatrix = Tensor::ones({_energies.getBatchDim(), _nGenerations, _nGenerations}, dtypes::kComplexFloat)
                            .requiresGrad(false);

        if (_matterSolver)
        {
            _matterSolver->setEnergies(newEnergies);
        }

        return *this;
    }

    /// @brief Set the masses corresponding to the vacuum hamiltonian eigenstates
    /// @param newMasses The new masses to use. This tensor is expected to have a
    /// batch dimension + 1 more dimensions of size nGenerations. The batch
    /// dimension can (and probably should) be 1 and it will be broadcast to
    /// match the batch dimension of the energies supplied to calculateProbs().
    /// So dimension should be {1, nGenerations}.
    virtual inline Propagator &setMasses(Tensor &newMasses)
    {
        NT_PROFILE();

        _masses = newMasses;
        if (_matterSolver != nullptr)
        {
            _matterSolver->setMasses(newMasses);
        }

        return *this;
    }

    /// @brief Set the baseline
    /// @param newBaseline new value
    inline Propagator &setBaseline(const Tensor &newBaseline)
    {

        NT_PROFILE();

        _baseline = newBaseline;

        _weightArgDenom = Tensor::scale(Tensor::scale(_energies, 2.0),
                                        std::complex<float>(1.0) / (std::complex<float>(-1.0J) * _baseline));

        return *this;
    }

    /// @}

  private:
    // For calculating with alternate masses and mixing matrix, e.g. if using effective
    // values from massSolver
    [[nodiscard]] Tensor _calculateProbs() override;

  protected:
    Tensor _weightMatrix;

    std::shared_ptr<Tensor> _baseline;
    std::shared_ptr<Tensor> _energies;
    std::shared_ptr<BaseMatterSolver> _matterSolver;
};

}; // namespace nuTens
