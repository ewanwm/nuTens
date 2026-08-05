#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/instrumentation.hpp>

/// @file base-matter-solver.hpp

namespace nuTens
{

class BaseMatterSolver
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

    BaseMatterSolver(int nGenerations, bool antiNeutrino, dtypes::deviceType device)
        : antiNeutrino(antiNeutrino), nGenerations(nGenerations), _device(device)
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

        if (newMatrix.getNdim() != 3)
        {
            throw std::invalid_argument(
                "Mixing Matrix tensor must be 3 dimensional (n_batches, n_generations, n_generations)");
        }

        if (newMatrix.getDevice() != _device)
        {
            NT_WARN(__FILE__, __LINE__,
                    "mixing matrix tensor is on a different device from matter solver, this will likely cause you "
                    "problems!!");
        }

        if ((newMatrix.getShape()[1] != nGenerations) || (newMatrix.getShape()[2] != nGenerations))
        {
            throw std::invalid_argument("Bad mixing matrix shape!!");
        }

        mixingMatrix = newMatrix;

        return *this;
    }

    /// @brief Set new mass eigenvalues for this solver
    /// @param newMasses The new masses
    virtual inline BaseMatterSolver &setMasses(const Tensor &newMasses)
    {
        NT_PROFILE();

        if (newMasses.getNdim() != 2)
        {
            throw std::invalid_argument("Mass tensor must be 2 dimensional (n_batches, n_generations)");
        }

        if (newMasses.getDevice() != _device)
        {
            NT_WARN(__FILE__, __LINE__,
                    "mass tensor is on a different device from matter solver, this will likely cause you problems!!");
        }

        if (newMasses.getShape()[1] != nGenerations)
        {
            throw std::invalid_argument(
                "Mass tensor shape has wrong number of generations. Shape should be (n_batches, n_generations)");
        }

        masses = newMasses;

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

        if (newEnergies.getDevice() != _device)
        {
            NT_WARN(__FILE__, __LINE__,
                    "energy tensor is on a different device from matter solver, this will likely cause you problems!!");
        }

        energies = newEnergies;

        hamiltonian = Tensor::zeros({energies.getBatchDim(), nGenerations, nGenerations}, dtypes::kComplexFloat)
                          .device(_device)
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
    dtypes::deviceType _device;
    Tensor energies;
    Tensor hamiltonian;
    Tensor mixingMatrix;
    Tensor masses;
};

}; // namespace nuTens