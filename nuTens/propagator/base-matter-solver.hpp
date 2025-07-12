#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/instrumentation.hpp>

/// @file base-matter-solver.hpp

class BaseMatterSolver
{
    /// @class BaseMatterSolver
    /// @brief Abstract base class for matter effect solvers

  public:

    BaseMatterSolver(int nGenerations) 
    :
      nGenerations(nGenerations) {}

    /// @name Setters
    /// @{
    virtual void setPMNS(const Tensor &newPMNS) = 0;

    virtual void setMasses(const Tensor &newMasses) = 0;

    virtual void calculateEigenvalues(Tensor &eigenvectors, Tensor &eigenvalues) = 0;

    inline virtual void setEnergies(const Tensor &newEnergies) {
      
      assert((newEnergies.getNdim() == 2));
      
      NT_PROFILE();
      
      energies = newEnergies;
      energiesRed = energies.getValues({"...", 0});

      hamiltonian = Tensor::zeros({energies.getBatchDim(), nGenerations, nGenerations}, NTdtypes::kComplexFloat);
    }

    /// @}

  protected:

    int nGenerations;
    Tensor energies;
    Tensor energiesRed;
    Tensor hamiltonian;
};