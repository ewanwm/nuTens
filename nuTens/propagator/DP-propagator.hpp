#pragma once

#include <nuTens/propagator/constants.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/tensor.hpp>

/// @file base-matter-solver.hpp

namespace nuTens
{

/// @brief Solver based on Denton, Parke (2024) (https://arxiv.org/pdf/2405.02400)
/// assumes 3 flavour oscillations and dm^2_21 > 0
class DPpropagator : public Propagator
{

    template <typename T> struct fail : std::false_type
    {
    };

  public:
    DPpropagator(float baseline, bool antiNeutrino, float density, int NRiterations)
        : Propagator(3, baseline, antiNeutrino), NRiterations(NRiterations), _density(density) {};

    /// @{Setters

    inline void setTheta12(Tensor &newTheta12)
    {
        NT_PROFILE();

        theta12 = newTheta12;
    }
    inline void setTheta23(Tensor &newTheta23)
    {
        NT_PROFILE();

        theta23 = newTheta23;
    }
    inline void setTheta13(Tensor &newTheta13)
    {
        NT_PROFILE();

        theta13 = newTheta13;
    }
    inline void setDeltaCP(Tensor &newDeltaCP)
    {
        NT_PROFILE();

        deltaCP = newDeltaCP;
    }
    inline void setDmsp21(Tensor &newDmsq21)
    {
        NT_PROFILE();

        dmsq21 = newDmsq21;
    }
    inline void setDmsq31(Tensor &newDmsq31)
    {
        NT_PROFILE();

        dmsq31 = newDmsq31;
    }

    inline void setParameters(Tensor &newTheta12, Tensor &newTheta23, Tensor &newTheta13, Tensor &newDeltaCP,
                              Tensor &newDmsq21, Tensor &newDmsq31)
    {
        NT_PROFILE();

        theta12 = newTheta12;
        theta23 = newTheta23;
        theta13 = newTheta13;
        deltaCP = newDeltaCP;
        dmsq21 = newDmsq21;
        dmsq31 = newDmsq31;
    }

    /// @brief Set the neutrino energies
    /// @param newEnergies The neutrino energies
    inline void setEnergies(Tensor &newEnergies) override
    {
        NT_PROFILE();

        _energies = newEnergies;
    }

    /// @}

    /// @{Getters

    const Tensor &getTheta12()
    {
        NT_PROFILE();

        return theta12;
    }
    const Tensor &getTheta23()
    {
        NT_PROFILE();

        return theta23;
    }
    const Tensor &getTheta13()
    {
        NT_PROFILE();

        return theta13;
    }
    const Tensor &getDeltaCP()
    {
        NT_PROFILE();

        return deltaCP;
    }
    const Tensor &getDmsp21()
    {
        NT_PROFILE();

        return dmsq21;
    }
    const Tensor &getDmsq31()
    {
        NT_PROFILE();

        return dmsq31;
    }
    const Tensor &getEnergies()
    {
        NT_PROFILE();

        return _energies;
    }

    /// @}

    /// @brief Calculate the oscilaltion probabilities for the current set of parameters
    ///        and energies
    [[nodiscard]] Tensor calculateProbs();

    // shouldn't try to use a matter solver with this class since it internally
    // handles all matter effects
    template <typename T = bool> inline void setMatterSolver(const std::shared_ptr<BaseMatterSolver> &newSolver)
    {
        static_assert(fail<T>::value, "do not use for DP propagator");
    };

    // shouldn't use as this method requires us to directly set oscillation parameters
    template <typename T = bool> inline void setMixingMatrix(Tensor &newMatrix)
    {
        static_assert(fail<T>::value, "do not use for DP propagator");
    };

    // shouldn't use as this method requires us to directly set oscillation parameters
    template <typename T = bool> inline void setMasses(Tensor &newMasses)
    {
        static_assert(fail<T>::value, "do not use for DP propagator");
    };

  private:
    Tensor theta12 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor theta13 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor theta23 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

    Tensor deltaCP = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

    Tensor dmsq21 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor dmsq31 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

    int NRiterations;
    float _density;
};

}; // namespace nuTens