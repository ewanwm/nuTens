#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <vector>

class VacuumPropagator{

    public:

        VacuumPropagator(int nGenerations, float baseline)
            :
            baseline(baseline),
            nGenerations(nGenerations)
            {};

        Tensor calculateProbs();

        /// @name Setters
        /// @{
        void setEnergies(Tensor newEnergies){ energies = newEnergies; }
        
        void setMasses(Tensor newMasses){ masses = newMasses; }

        inline void setPMNS(Tensor newPMNS){ PMNSmatrix = newPMNS; }

        inline void setPMNS(const std::vector<int> &indices, float value){ PMNSmatrix.setValue(indices, value); }

        inline void setPMNS(const std::vector<int> &indices, std::complex<float> value){ PMNSmatrix.setValue(indices, value); }
        /// @}

    private:
        Tensor PMNSmatrix;
        Tensor masses;
        Tensor energies;
        int nGenerations;
        float baseline;
};