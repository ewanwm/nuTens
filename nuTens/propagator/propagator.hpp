#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <vector>

class Propagator{

    public:

        Propagator(int nGenerations, float baseline)
            :
            baseline(baseline),
            nGenerations(nGenerations)
            {};

        Tensor calculateProbs(const Tensor &energies) const;

        /// @name Setters
        /// @{        
        void setMasses(Tensor &newMasses){ masses = newMasses; }

        inline void setPMNS(Tensor &newPMNS){ PMNSmatrix = newPMNS; }

        inline void setPMNS(const std::vector<int> &indices, float value){ PMNSmatrix.setValue(indices, value); }

        inline void setPMNS(const std::vector<int> &indices, std::complex<float> value){ PMNSmatrix.setValue(indices, value); }
        /// @}

    private:
        Tensor PMNSmatrix;
        Tensor masses;
        int nGenerations;
        float baseline;
};