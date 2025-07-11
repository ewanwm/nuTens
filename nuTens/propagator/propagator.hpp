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
    inline Propagator(int nGenerations, float baseline) : _baseline(baseline), _nGenerations(nGenerations){};

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

    /// @brief Set the neutrino energies
    /// @param newEnergies The neutrino energies
    inline void setEnergies(Tensor &newEnergies)
    {
        NT_PROFILE();

        _energies = newEnergies;
        _weightMatrix = Tensor::ones({_energies.getBatchDim(), _nGenerations, _nGenerations}, NTdtypes::kComplexFloat)
                        .requiresGrad(false);
        _weightArgDenom = Tensor::scale(Tensor::scale(_energies, 2.0), std::complex<float>(1.0) / (std::complex<float>(-1.0J) * _baseline));

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
    inline void setMasses(Tensor &newMasses)
    {
        NT_PROFILE();

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


    /// @brief Calculate the oscillation probabilities
    [[nodiscard]] inline Tensor calculateProbs()
    {
        NT_PROFILE();

        Tensor ret;

        // if a matter solver was specified, use effective values for masses and PMNS
        // matrix, otherwise just use the "raw" ones
        if (_matterSolver != nullptr)
        {
            Tensor eigenVals =
                Tensor::zeros({1, _nGenerations, _nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);
            Tensor eigenVecs =
                Tensor::zeros({1, _nGenerations, _nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);

            _matterSolver->calculateEigenvalues(eigenVecs, eigenVals);
            Tensor effectiveMassesSq = Tensor::mul(eigenVals, Tensor::scale(_energies, 2.0));
            Tensor effectivePMNS = Tensor::matmul(_pmnsMatrix, eigenVecs);

            ret = _calculateProbs(effectiveMassesSq, effectivePMNS);
        }

        else
        {
            ret = _calculateProbs(Tensor::mul(_masses, _masses), _pmnsMatrix);
        }

        return ret;
    }

  private:
    // For calculating with alternate masses and PMNS, e.g. if using effective
    // values from massSolver
    [[nodiscard]] inline Tensor _calculateProbs(const Tensor &massesSq, const Tensor &PMNS)
    {
        NT_PROFILE();

        Tensor weightVector = Tensor::exp(
            Tensor::div(massesSq, _weightArgDenom));

        _weightMatrix.requiresGrad(false);
        for (int i = 0; i < _nGenerations; i++)
        {
            for (int j = 0; j < _nGenerations; j++)
            {
                _weightMatrix.setValue({"...", i, j}, weightVector.getValues({"...", j}));
            }
        }
        _weightMatrix.requiresGrad(true);

        Tensor sqrtProbabilities = Tensor::matmul(PMNS.conj(), Tensor::transpose(Tensor::mul(PMNS, _weightMatrix), 1, 2));

        return Tensor::pow(sqrtProbabilities.abs(), 2);
    }

  private:
    Tensor _pmnsMatrix;
    Tensor _masses;
    Tensor _energies;
    Tensor _weightMatrix;
    Tensor _weightArgDenom;
    int _nGenerations;
    float _baseline;

    std::shared_ptr<BaseMatterSolver> _matterSolver;

};