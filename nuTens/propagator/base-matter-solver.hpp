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

    ~BaseMatterSolver() {};

    /// @name Setters
    /// @{

    /// @brief Set a new mixing matrix for this solver
    /// @param newMatrix The new matrix to set
    virtual inline void setMixingMatrix(const Tensor &newMatrix)
    {
        NT_PROFILE();

        mixingMatrix = newMatrix;
    }

    /// @brief Set new mass eigenvalues for this solver
    /// @param newMasses The new masses
    virtual inline void setMasses(const Tensor &newMasses)
    {
        assert((newMasses.getNdim() == 2));
        NT_PROFILE();

        masses = newMasses;
    }

    virtual void calculateEigenvalues(Tensor &eigenvectors, Tensor &eigenvalues) = 0;

    inline virtual void setEnergies(const Tensor &newEnergies)
    {

        assert((newEnergies.getNdim() == 2));

        NT_PROFILE();

        energies = newEnergies;
        energiesRed = energies.getValues({"..."});
        energiesRed.unsqueeze(-1);

        hamiltonian = Tensor::zeros({energies.getBatchDim(), nGenerations, nGenerations}, dtypes::kComplexFloat)
                          .requiresGrad(false);
    }

    /// @brief Set whether we are dealing with anti-neutrinos
    /// @param newValue
    virtual inline void setAntiNeutrino(bool newValue)
    {
        NT_PROFILE();

        antiNeutrino = newValue;
    }

    /// @}

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