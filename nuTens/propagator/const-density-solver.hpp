#pragma once

#include <nuTens/propagator/base-matter-solver.hpp>
#include <nuTens/propagator/constants.hpp>

/// @file const-density-solver.hpp

namespace nuTens
{

class ConstDensityMatterSolver : public BaseMatterSolver
{
    /*!
     * @class ConstDensityMatterSolver
     * @brief Solver class for constant density material
     *
     * This class is used to obtain effective mass eigenstates and an effective
     * mixing matrix due to matter effects for neutrinos passing through a block of
     * material of constant density.
     *
     * The method used here is to first construct the effective Hamiltonian
     * \f{equation}
     *   \frac{1}{2E} Diag(m^2_i) - \sqrt(2)G N_e \mathbf{U}_{ei} \otimes
     * \mathbf{U}_{ie}^\dagger \f} where \f$ \mathbf{U} \f$ is the supplied mixing
     * matrix and \f$ Diag(m^2_i) \f$ is a diagonal matrix with the specified
     * mass eigenvalues on the diagonal. We then calculate the eigenvalues \f$
     * m_i^\prime \f$ and eigenvectors, summarised in the matrix \f$ V_{ij} \f$.
     * These can then be passed to a propagator class to get the oscillation
     * probabilities in the presence of such matter effects.
     *
     * See \cite Barger for more details.
     *
     */

  public:
    /// @brief Constructor
    /// @arg nGenerations The number of neutrino generations this propagator
    /// should expect
    /// @arg density The electron density of the material to propagate in
    /// @arg antiNeutrino True if we are calculating effects for anti-neutrinos
    ConstDensityMatterSolver(int nGenerations, float density, bool antiNeutrino = false)
        : BaseMatterSolver(nGenerations, antiNeutrino), density(density)
    {
        diagMassMatrix = Tensor::zeros({1, nGenerations, nGenerations}, dtypes::kComplexFloat).requiresGrad(false);
    };

    /// @name Setters
    /// @{

    /// @brief Set whether we are dealing with anti-neutrinos
    /// @param newValue
    inline void setAntiNeutrino(bool newValue) override
    {
        NT_PROFILE();

        BaseMatterSolver::setAntiNeutrino(newValue);

        buildElectronOuterProduct();
    }

    /// @brief Set a new mixing matrix for this solver
    /// @param newMatrix The new matrix to set
    inline void setMixingMatrix(const Tensor &newMatrix) override
    {
        NT_PROFILE();

        mixingMatrix = newMatrix;

        buildElectronOuterProduct();
    };

    /// @brief Set new mass eigenvalues for this solver
    /// @param newMasses The new masses
    inline void setMasses(const Tensor &newMasses) override
    {
        assert((newMasses.getNdim() == 2));
        NT_PROFILE();

        masses = newMasses;

        Tensor m = masses.getValues({0, "..."});
        Tensor diag = Tensor::scale(Tensor::mul(m, m), 0.5);

        // construct the diagonal mass^2 matrix used in the hamiltonian
        diagMassMatrix = Tensor::diag(diag).requiresGrad(false).unsqueeze(0);
    }

    /// @brief set a new density
    /// @param newDensity the new value
    inline void setDensity(float newDensity)
    {
        /// @todo super inefficient to recalculate this here and also in
        /// setMixingMatrix. Would be good to have some _valuesChanged flag that causes
        /// these kind of things to be recalculated inside of calculateEigenvalues
        /// if any of the dependent variables changed e.g. mixing matrix, density, masses
        /// See also smilar problem in propagator::setBaseline

        NT_PROFILE();

        density = newDensity;

        buildElectronOuterProduct();
    }

    /// @}

    /// @{ Getters

    [[nodiscard]] inline float getDensity() const
    {
        return density;
    }

    /// @brief Calculate the hamiltonian eigenvalues and eigenvectors, i.e. the effective Mass^2 states and effective
    /// mixing matrix
    /// @param[out] eigenvectors The returned eigenvectors
    /// @param[out] eigenvalues The corresponding eigenvalues
    void calculateEigenvalues(Tensor &eigenvectors, Tensor &eigenvalues) override;

    /// construct the outer product of the electron row of the mixing matrix, used in building the hamiltonian, and
    /// return a copy of it potentially useful for debugging
    Tensor getElectronOuterProduct();

    /// construct the hamiltonian and return a copy of it
    /// potentially useful for debugging
    Tensor getHamiltonian();

  private:
    /// @brief construct the outer product of the electron neutrino row of the mixing
    /// matrix used to construct the hamiltonian
    void buildElectronOuterProduct();

    /// @brief Construct the hamiltonian
    void buildHamiltonian();

    Tensor diagMassMatrix;
    Tensor electronOuter;
    float density;
};

}; // namespace nuTens