#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/propagator/base-matter-solver.hpp>
#include <vector>
#include <memory>

class Propagator{

    public:

        Propagator(int nGenerations, float baseline)
            :
            _baseline(baseline),
            _nGenerations(nGenerations)
            {};

        Tensor calculateProbs(const Tensor &energies) const;

        /// @name Setters
        /// @{        
        void setMasses(Tensor &newMasses){ _masses = newMasses; }

        inline void setMatterSolver(std::unique_ptr<BaseMatterSolver> &newSolver){ _matterSolver = std::move(newSolver); }

        inline void setPMNS(Tensor &newPMNS){ _PMNSmatrix = newPMNS; }

        inline void setPMNS(const std::vector<int> &indices, float value){ _PMNSmatrix.setValue(indices, value); }

        inline void setPMNS(const std::vector<int> &indices, std::complex<float> value){ _PMNSmatrix.setValue(indices, value); }
        /// @}

    private: 
        // For calculating with alternate masses and PMNS, e.g. if using effective values from massSolver
        Tensor _calculateProbs(const Tensor &energies, const Tensor &masses, const Tensor &PMNS) const;

    private:
        Tensor _PMNSmatrix;
        Tensor _masses;
        int _nGenerations;
        float _baseline;

        std::unique_ptr<BaseMatterSolver> _matterSolver;
};