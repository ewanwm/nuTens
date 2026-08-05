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

    struct MassSqTensor : public nuTens::Tensor
    {
        /*!
         * @struct MassSqTensor
         * @brief Holds squared mass values
         */
        MassSqTensor() = default;
        explicit MassSqTensor(const nuTens::Tensor &tensor) : Tensor(tensor){};
    };
    struct MixingMatrixTensor : public nuTens::Tensor
    {
        /*!
         * @struct MixingMatrixTensor
         * @brief Holds mixing matrix
         */
        MixingMatrixTensor() = default;
        explicit MixingMatrixTensor(const nuTens::Tensor &tensor) : Tensor(tensor){};
    };

  public:
    /// @brief Constructor
    /// @param nGenerations The number of generations the propagator should
    /// expect
    Propagator(int nGenerations, dtypes::deviceType device = dtypes::kCPU)
        : _nGenerations(nGenerations), _device(device){};

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

    /// @brief Calculate the oscillation probabilities
    /// @param energies The energies of the neutrinos
    [[nodiscard]] virtual Tensor calculateProbs();

    /// @name Setters
    /// @{

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

    /// @brief Set a matter solver to use to deal with matter effects
    /// @param newSolver A derivative of BaseMatterSolver
    virtual inline Propagator &setMatterSolver(const std::shared_ptr<BaseMatterSolver> &newSolver)
    {
        NT_PROFILE();
        _matterSolver = newSolver;

        if (_energies.isInitialised())
        {
            _matterSolver->setEnergies(_energies);
        }

        if (_masses.isInitialised())
        {
            _matterSolver->setMasses(_masses);
        }
        if (_mixingMatrix.isInitialised())
        {
            _matterSolver->setMixingMatrix(_mixingMatrix);
        }
        _matterSolver->setAntiNeutrino(_antiNeutrino);

        return *this;
    }

    /// \todo Should add a check to tensors supplied to the setters to see how
    /// many dimensions they have, and if missing a batch dimension, add one.

    /// @brief Set the neutrino energies
    /// @param newEnergies The neutrino energies
    virtual Propagator &setEnergies(Tensor &newEnergies)
    {
        NT_PROFILE();

        if (newEnergies.getNdim() != 1)
        {
            throw std::invalid_argument("Energy tensor must be 1 dimensional (n_energies)");
        }

        if ((newEnergies.getDType() != dtypes::kComplexDouble) && (newEnergies.getDType() != dtypes::kComplexFloat))
        {
            throw std::invalid_argument("Energy tensor must have complex type");
        }

        if (newEnergies.getDevice() != _device)
        {
            NT_WARN(__FILE__, __LINE__,
                    "energy tensor is on a different device from propagator, this will likely cause you problems!!");
        }

        _energies = newEnergies;

        _weightMatrix = Tensor::ones({_energies.getBatchDim(), _nGenerations, _nGenerations}, dtypes::kComplexFloat)
                            .device(_device)
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

        if (newMasses.getNdim() != 2)
        {
            throw std::invalid_argument("Mass tensor must be 2 dimensional (n_batches, n_generations)");
        }

        if (newMasses.getDevice() != _device)
        {
            NT_WARN(__FILE__, __LINE__,
                    "mass tensor is on a different device from propagator, this will likely cause you problems!!");
        }

        if (newMasses.getShape()[1] != _nGenerations)
        {
            throw std::invalid_argument(
                "Mass tensor shape has wrong number of generations. Shape should be (n_batches, n_generations)");
        }

        _masses = newMasses;

        if (_matterSolver)
        {
            _matterSolver->setMasses(newMasses);
        }

        return *this;
    }

    /// @brief Set a whole new mixing matrix
    /// @param newMatrix The new matrix to use
    virtual inline Propagator &setMixingMatrix(Tensor &newMatrix)
    {
        NT_PROFILE();

        if (newMatrix.getNdim() != 3)
        {
            throw std::invalid_argument(
                "Mixing Matrix tensor must be 3 dimensional (n_batches, n_generations, n_generations)");
        }

        if (newMatrix.getDevice() != _device)
        {
            NT_WARN(
                __FILE__, __LINE__,
                "mixing matrix tensor is on a different device from propagator, this will likely cause you problems!!");
        }

        if ((newMatrix.getShape()[1] != _nGenerations) || (newMatrix.getShape()[2] != _nGenerations))
        {
            throw std::invalid_argument("Bad mixing matrix shape!!");
        }

        _mixingMatrix = newMatrix;

        if (_matterSolver)
        {
            _matterSolver->setMixingMatrix(newMatrix);
        }

        return *this;
    }

    /// @brief Set the baseline
    /// @param newBaseline new value
    inline Propagator &setBaseline(float newBaseline)
    {

        NT_PROFILE();

        _baseline = newBaseline;

        return *this;
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
    [[nodiscard]] Tensor _calculateProbs(const MassSqTensor &masses, const MixingMatrixTensor &mixingMatrix);

  protected:
    Tensor _mixingMatrix;
    Tensor _masses;
    Tensor _energies;

    Tensor _weightMatrix;
    Tensor _weightArgDenom;

    BaseMatterSolver::EigenvalTensor _eigenVals;
    BaseMatterSolver::EigenvecTensor _eigenVecs;

    int _nGenerations;
    float _baseline{NAN};
    bool _antiNeutrino{false};

    dtypes::deviceType _device;

    std::shared_ptr<BaseMatterSolver> _matterSolver;
};

}; // namespace nuTens
