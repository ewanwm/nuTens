#pragma once

#include <nuTens/tensors/tensor.hpp>

class BaseMatterSolver {

    public:

        /// @name Setters
        /// @{
        virtual void setPMNS(const Tensor &newPMNS) = 0;

        virtual void setMasses(const Tensor &newMasses) = 0;

        virtual void calculateEigenvalues(const Tensor &energies, Tensor &eigenvectors, Tensor &eigenvalues) = 0;

        /// @}

        static constexpr float Groot2 = 1.52588e-4; //!< sqrt(2) G_fermi used in calculating hamiltonian

};