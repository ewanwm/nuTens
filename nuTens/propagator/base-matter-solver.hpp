#pragma once

#include <nuTens/propagator/module-base.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/instrumentation.hpp>

/// @file base-matter-solver.hpp

namespace nuTens
{

class BaseMatterSolver : public ModuleBase
{
    /// @class BaseMatterSolver
    /// @brief Abstract base class for matter effect solvers

  public:
    struct EigenvecTensor : public nuTens::Tensor
    {
        /*!
         * @struct EigenvecTensor
         * @brief Holds matter solver eigenvectors
         */
        EigenvecTensor() = default;
        explicit EigenvecTensor(const nuTens::Tensor &tensor) : Tensor(tensor){};
    };
    struct EigenvalTensor : public nuTens::Tensor
    {
        /*!
         * @struct EigenvalTensor
         * @brief Holds matter solver eigenvalues
         */
        EigenvalTensor() = default;
        explicit EigenvalTensor(const nuTens::Tensor &tensor) : Tensor(tensor){};
    };

    BaseMatterSolver(int nGenerations, bool antiNeutrino, dtypes::deviceType device, long batchSize)
        : antiNeutrino(antiNeutrino), nGenerations(nGenerations), ModuleBase(batchSize, device, "BaseMatterSolver")
    {
    }

    /// @brief Destructor
    virtual ~BaseMatterSolver() = default;
    /// @brief copy constructor
    BaseMatterSolver(BaseMatterSolver const &) = default;
    /// @brief copy assignment operator
    BaseMatterSolver &operator=(BaseMatterSolver const &) = default;
    /// @brief move constructor
    BaseMatterSolver(BaseMatterSolver &&) = default;
    /// @brief move assignment operator
    BaseMatterSolver &operator=(BaseMatterSolver &&) = default;

    /// @name Setters
    /// @{

    /// @brief Set a new mixing matrix for this solver
    /// @param newMatrix The new matrix to set
    virtual inline BaseMatterSolver &setMixingMatrix(const Tensor &newMatrix)
    {
        NT_PROFILE();

        checkParameterShape(newMatrix, 2, {nGenerations, nGenerations}, mixingMatrix, "MixingMatrix");

        if (mixingMatrix.getDevice() != getDevice())
        {
            NT_WARN("mixing matrix tensor is on a different device from matter solver, this will likely cause you "
                    "problems!!");
        }

        return *this;
    }

    /// @brief Set new mass eigenvalues for this solver
    /// @param newMasses The new masses
    virtual inline BaseMatterSolver &setMasses(const Tensor &newMasses)
    {
        NT_PROFILE();

        checkParameterShape(newMasses, 1, {nGenerations}, masses, "Masses");

        if (masses.getDevice() != getDevice())
        {
            NT_WARN("mass tensor is on a different device from matter solver, this will likely cause you problems!!");
        }

        return *this;
    }

    /// @brief set new neutrino energies
    /// @param newEnergies new energy values
    inline virtual BaseMatterSolver &setEnergies(const Tensor &newEnergies)
    {

        NT_PROFILE();

        if (newEnergies.getNdim() != 1)
        {
            throw std::invalid_argument("Energy tensor must be 1 dimensional (n_energies)");
        }

        if (newEnergies.getDevice() != getDevice())
        {
            NT_WARN("energy tensor is on a different device from matter solver, this will likely cause you problems!!");
        }

        energies = newEnergies;

        hamiltonian = Tensor::zeros({energies.getBatchDim(), nGenerations, nGenerations}, dtypes::kComplexFloat)
                          .device(getDevice())
                          .requiresGrad(false);

        return *this;
    }

    /// @brief Set whether we are dealing with anti-neutrinos
    /// @param newValue
    virtual inline BaseMatterSolver &setAntiNeutrino(bool newValue)
    {
        NT_PROFILE();

        antiNeutrino = newValue;

        return *this;
    }

    /// @}

    /// @{getters

    /// @brief Get the current mixing matrix for this solver
    virtual inline const Tensor &getMixingMatrix()
    {
        NT_PROFILE();

        return mixingMatrix;
    }

    /// @brief Get the current mass eigenvalues for this solver
    virtual inline const Tensor &getMasses()
    {
        NT_PROFILE();

        return masses;
    }

    inline virtual const Tensor &getEnergies()
    {

        NT_PROFILE();

        return energies;
    }

    /// @brief Get whether we are dealing with anti-neutrinos
    virtual inline bool getAntiNeutrino()
    {
        NT_PROFILE();

        return antiNeutrino;
    }

    /// @}

    virtual void calculateEigenvalues(EigenvecTensor &eigenvectors, EigenvalTensor &eigenvalues) = 0;

  protected:
    bool antiNeutrino;
    int nGenerations;
    Tensor energies;
    Tensor hamiltonian;
    Tensor mixingMatrix;
    Tensor masses;
};

}; // namespace nuTens