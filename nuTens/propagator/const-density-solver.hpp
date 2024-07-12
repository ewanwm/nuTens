#pragma once

#include <nuTens/propagator/base-matter-solver.hpp>
#include <nuTens/propagator/constants.hpp>

class ConstDensityMatterSolver: public BaseMatterSolver{

    /*!
    * @class ConstDensityMatterSolver
    * @brief Solver class for constant density material
    * 
    * This class is used to obtain effective mass eigenstates and an effective PMNS matrix due to matter effects for neutrinos passing through a block of material of constant density.
    * 
    * The method used here is to first construct the effective Hamiltonian 
    * \f{equation} 
    *   \frac{1}{2E} Diag(m^2_i) - \sqrt(2)G N_e \mathbf{U}_{ei} \otimes \mathbf{U}_{ie}^\dagger 
    * \f}
    * where \f$ \mathbf{U} \f$ is the supplied PMNS matrix and \f$ Diag(m^2_i) \f$ is a diagonal matrix with the specified mass eigenvalues on the diagonal. 
    * We then calculate the eigenvalues \f$ m_i^\prime \f$ and eigenvectors, summarised in the matrix \f$ V_{ij} \f$. 
    * These can then be passed to a propagator class to get the oscillation probabilities in the presence of such matter effects.
    * 
    * See \cite Barger for more details.
    * 
    */

    public:

        /// @brief Constructor 
        /// @arg nGenerations The number of neutrino generations this propagator should expect
        /// @arg density The electron density of the material to propagate in 
        ConstDensityMatterSolver(int nGenerations, float density)
        :
            nGenerations(nGenerations),
            density(density)
        {
            diagMassMatrix.zeros({1, nGenerations, nGenerations}, NTdtypes::kFloat);
        };

        /// @name Setters
        /// @{

        /// @brief Set a new PMNS matrix for this solver
        /// @param newPMNS The new matrix to set
        inline void setPMNS(const Tensor &newPMNS) { 
            PMNS = newPMNS; 

            // construct the outer product of the electron neutrino row of the PMNS matrix used to construct the hamiltonian
            electronOuter = Tensor::scale(Tensor::outer(PMNS.getValue({0, 0,"..."}), PMNS.getValue({0, 0,"..."}).conj()), Constants::Groot2 * density);
        };


        /// @brief Set new mass eigenvalues for this solver
        /// @param newMasses The new masses
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

        /// @}
        
        
        /// @brief Set new mass eigenvalues for this solver
        /// @param[in] energies Tensor of energies, expected to have a batch dimension and two further dimensions to make casting unambiguous i.e. shape should look like {Nbatches, 1, 1}.
        /// @param[out] eigenvectors The returned eigenvectors
        /// @param[out] eigenvalues The corresponding eigenvalues 
        void calculateEigenvalues(const Tensor &energies, Tensor &eigenvectors, Tensor &eigenvalues);

    private:
        Tensor PMNS;
        Tensor masses;
        Tensor diagMassMatrix;
        Tensor electronOuter;
        int nGenerations;
        float density;
    
};