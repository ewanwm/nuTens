#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/instrumentation.hpp>

/// @file base-matter-solver.hpp

namespace nuTens
{

class BaseMatterSolver
{
    /// @class BaseMatterSolver
    /// @brief Abstract base class for matter effect solvers

  public:
    BaseMatterSolver(int nGenerations, bool antiNeutrino) : antiNeutrino(antiNeutrino), nGenerations(nGenerations)
    {
    }

    /// @brief Destructor
    virtual ~BaseMatterSolver() = default;
    /// @brief copy constructor
    BaseMatterSolver(BaseMatterSolver const &) = default;
    /// @brief copy assignment operator
    BaseMatterSolver &operator=(BaseMatterSolver const &) = default;
    /// @brief move constructor
    BaseMatterSolver(BaseMatterSolver &&) = default;
    /// @brief move assignment operator
    BaseMatterSolver &operator=(BaseMatterSolver &&) = default;

    /// @name Setters
    /// @{

    /// @brief Set a new mixing matrix for this solver
    /// @param newMatrix The new matrix to set
    virtual inline BaseMatterSolver &setMixingMatrix(const Tensor &newMatrix)
    {
        NT_PROFILE();

        if (newMatrix.getNdim() != 3)
        {
            throw std::invalid_argument(
                "Mixing Matrix tensor must be 3 dimensional (n_batches, n_generations, n_generations)");
        }

        mixingMatrix = newMatrix;

        return *this;
    }

    /// @brief Set new mass eigenvalues for this solver
    /// @param newMasses The new masses
    virtual inline BaseMatterSolver &setMasses(const Tensor &newMasses)
    {
        NT_PROFILE();

        if (newMasses.getNdim() != 2)
        {
            throw std::invalid_argument("Mass tensor must be 2 dimensional (n_batches, n_generations)");
        }

        masses = newMasses;

        return *this;
    }

    inline virtual BaseMatterSolver &setEnergies(const Tensor &newEnergies)
    {

        if (newEnergies.getNdim() != 2)
        {
            throw std::invalid_argument("Energy tensor must be 2 dimensional (1, n_energies)");
        }

        NT_PROFILE();

        energies = newEnergies;
        energiesRed = energies.getValues({"..."});
        energiesRed.unsqueeze(-1);

        hamiltonian = Tensor::zeros({energies.getBatchDim(), nGenerations, nGenerations}, dtypes::kComplexFloat)
                          .requiresGrad(false);

        return *this;
    }

    /// @brief Set whether we are dealing with anti-neutrinos
    /// @param newValue
    virtual inline BaseMatterSolver &setAntiNeutrino(bool newValue)
    {
        NT_PROFILE();

        antiNeutrino = newValue;

        return *this;
    }

    /// @}

    /// @{getters

    /// @brief Get the current mixing matrix for this solver
    virtual inline const Tensor &getMixingMatrix()
    {
        NT_PROFILE();

        return mixingMatrix;
    }

    /// @brief Get the current mass eigenvalues for this solver
    virtual inline const Tensor &getMasses()
    {
        NT_PROFILE();

        return masses;
    }

    inline virtual const Tensor &getEnergies()
    {

        NT_PROFILE();

        return energies;
    }

    /// @brief Get whether we are dealing with anti-neutrinos
    virtual inline bool getAntiNeutrino()
    {
        NT_PROFILE();

        return antiNeutrino;
    }

    /// @}

    virtual void calculateEigenvalues(Tensor &eigenvectors, Tensor &eigenvalues) = 0;

  protected:
    bool antiNeutrino;
    int nGenerations;
    Tensor energies;
    Tensor energiesRed;
    Tensor hamiltonian;
    Tensor mixingMatrix;
    Tensor masses;
};

}; // namespace nuTens