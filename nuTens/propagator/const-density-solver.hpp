#pragma once

#include <nuTens/propagator/base-matter-solver.hpp>

class ConstDensityMatterSolver: public BaseMatterSolver{

    public:
        ConstDensityMatterSolver(int nGenerations, float baseline, float density)
        :
            nGenerations(nGenerations),
            baseline(baseline),
            density(density)
        {
            diagMassMatrix.zeros({1, nGenerations, nGenerations}, NTdtypes::kFloat);
        };

        
        inline void setPMNS(const Tensor &newPMNS) { 
            PMNS = newPMNS; 

            // construct the outer product of the electron neutrino row of the PMNS matrix used to construct the hamiltonian
            electronOuter = Tensor::scale(Tensor::outer(PMNS.getValue({0, 0,"..."}), PMNS.getValue({0, 0,"..."}).conj()), Groot2 * density);
        };

        inline void setMasses(const Tensor &newMasses) {
            masses = newMasses;

            // construct the diagonal mass^2 matrix used in the hamiltonian
            diagMassMatrix.requiresGrad(false);
            for (int i = 0; i < nGenerations; i++){
                float m_i = masses.getValue<float>({0, i});
                diagMassMatrix.setValue({0, i,i}, m_i * m_i / 2.0);
            };
            diagMassMatrix.requiresGrad(true);
        }
        
        void calculateEigenvalues(const Tensor &energies, Tensor &eigenvectors, Tensor &eigenvalues);

    private:
        Tensor PMNS;
        Tensor masses;
        Tensor diagMassMatrix;
        Tensor electronOuter;
        int nGenerations;
        float baseline;
        float density;
    
};