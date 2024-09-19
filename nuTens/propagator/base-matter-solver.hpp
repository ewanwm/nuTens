#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/instrumentation.hpp>

/// @file base-matter-solver.hpp

class BaseMatterSolver
{
    /// @class BaseMatterSolver
    /// @brief Abstract base class for matter effect solvers

  public:
    /// @name Setters
    /// @{
    virtual void setPMNS(const Tensor &newPMNS) = 0;

    virtual void setMasses(const Tensor &newMasses) = 0;

    virtual void calculateEigenvalues(const Tensor &energies, Tensor &eigenvectors, Tensor &eigenvalues) = 0;

    /// @}
};