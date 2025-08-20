#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/instrumentation.hpp>

/// @file base-matter-solver.hpp

namespace nuTens {

class BaseMatterSolver
{
    /// @class BaseMatterSolver
    /// @brief Abstract base class for matter effect solvers

  public:

    BaseMatterSolver(int nGenerations, bool antiNeutrino) 
    :
      antiNeutrino(antiNeutrino),
      nGenerations(nGenerations) 
      {}

    /// @name Setters
    /// @{
    virtual void setMixingMatrix(const Tensor &newMatrix) = 0;

    virtual void setMasses(const Tensor &newMasses) = 0;

    virtual void calculateEigenvalues(Tensor &eigenvectors, Tensor &eigenvalues) = 0;

    inline virtual void setEnergies(const Tensor &newEnergies) {
      
      assert((newEnergies.getNdim() == 2));
      
      NT_PROFILE();
      
      energies = newEnergies;
      energiesRed = energies.getValues({"..."});
      energiesRed.unsqueeze(-1);

      hamiltonian = Tensor::zeros({energies.getBatchDim(), nGenerations, nGenerations}, dtypes::kComplexFloat).requiresGrad(false);
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
};

};