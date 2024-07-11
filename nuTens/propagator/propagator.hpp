#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/propagator/base-matter-solver.hpp>
#include <vector>
#include <memory>

class Propagator{

    /*! 
    * @class Propagator 
    * @brief Neutrino oscillation probability calculator
    * 
    * This class is used to propagate neutrinos over some baseline and calculate the probability that they will oscillate to another flavour.
    * A Propagator can be configured using the Setters by assigning parameters (neutrino masses and PMNS matrix elements). 
    * You can assign a matter solver (a derivative of BaseMatterSolver) to deal with matter effects using setMatterSolver().
    * calculateProbs() can then be used to calculate energy dependent oscillation probabilities.
    *
    * (The specifics of this interface may change in the future)
    */

    public:

        /// @brief Constructor
        /// @param nGenerations The number of generations the propagator should expect
        /// @param baseline The baseline to propagate over
        Propagator(int nGenerations, float baseline)
            :
            _baseline(baseline),
            _nGenerations(nGenerations)
            {};

        /// @brief Calculate the oscillation probabilities
        /// @param energies The energies of the neutrinos
        Tensor calculateProbs(const Tensor &energies) const;

        /// @name Setters
        /// @{

        /// @brief Set a matter solver to use to deal with matter effects
        /// @param newSolver A derivative of BaseMatterSolver
        inline void setMatterSolver(std::unique_ptr<BaseMatterSolver> &newSolver){ _matterSolver = std::move(newSolver); }

        /// \todo Should add a check to tensors supplied to the setters to see how many dimensions they have, and if missing a batch dimension, add one.

        /// @brief Set the masses corresponding to the vacuum hamiltonian eigenstates
        /// @param newMasses The new masses to use. This tensor is expected to have a batch dimension + 1 more dimensions of size nGenerations. The batch dimension can (and probably should) be 1 and it will be broadcast to match the batch dimension of the energies supplied to calculateProbs(). So dimension should be {1, nGenerations}. 
        void setMasses(Tensor &newMasses){ _masses = newMasses; }

        /// @brief Set a whole new PMNS matrix
        /// @param newPMNS The new matrix to use
        inline void setPMNS(Tensor &newPMNS){ _PMNSmatrix = newPMNS; }

        /// @brief Set a single element of the PMNS matrix
        /// @param indices The index of the value to set
        /// @param value The new value
        inline void setPMNS(const std::vector<int> &indices, float value){ _PMNSmatrix.setValue(indices, value); }

        /// @brief Set a single element of the PMNS matrix
        /// @param indices The index of the value to set
        /// @param value The new value
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