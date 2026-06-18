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

        mixingMatrix = newMatrix;

        return *this;
    }

    /// @brief Set new mass eigenvalues for this solver
    /// @param newMasses The new masses
    virtual inline BaseMatterSolver &setMasses(const Tensor &newMasses)
    {
        assert((newMasses.getNdim() == 2));
        NT_PROFILE();

        masses = newMasses;

        return *this;
    }

    inline virtual BaseMatterSolver &setEnergies(const Tensor &newEnergies)
    {

        assert((newEnergies.getNdim() == 2));

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