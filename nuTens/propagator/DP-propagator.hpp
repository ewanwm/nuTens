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
    DPpropagator(int NRiterations) : Propagator(3), NRiterations(NRiterations){};

    /// @{Setters

    inline DPpropagator &setBaseline(float newBaseline)
    {
        NT_PROFILE();

        _baseline = newBaseline;
        return *this;
    }
    inline DPpropagator &setDensity(float newDensity)
    {
        NT_PROFILE();

        _density = newDensity;
        return *this;
    }
    inline DPpropagator &setTheta12(Tensor &newTheta12)
    {
        NT_PROFILE();

        theta12 = newTheta12;
        return *this;
    }
    inline DPpropagator &setTheta23(Tensor &newTheta23)
    {
        NT_PROFILE();

        theta23 = newTheta23;
        return *this;
    }
    inline DPpropagator &setTheta13(Tensor &newTheta13)
    {
        NT_PROFILE();

        theta13 = newTheta13;
        return *this;
    }
    inline DPpropagator &setDeltaCP(Tensor &newDeltaCP)
    {
        NT_PROFILE();

        deltaCP = newDeltaCP;
        return *this;
    }
    inline DPpropagator &setDmsp21(Tensor &newDmsq21)
    {
        NT_PROFILE();

        dmsq21 = newDmsq21;
        return *this;
    }
    inline DPpropagator &setDmsq31(Tensor &newDmsq31)
    {
        NT_PROFILE();

        dmsq31 = newDmsq31;
        return *this;
    }
    inline DPpropagator &setAntiNeutrino(bool newValue)
    {
        NT_PROFILE();

        _antiNeutrino = newValue;
        return *this;
    }
    /// If true, the \theta_{ij}'s you provide will be interpreted as \sin^2(\theta_{ij}).
    /// This will shortcut some of the computations performed by this propagator
    /// and speed up calculation time.
    inline DPpropagator &setSinSquaredThetas(bool newValue)
    {
        NT_PROFILE();

        interpretSinSquaredThetas = newValue;
        return *this;
    }

    /// @brief Set all parameters at once
    /// @param newTheta12 New \theta_{12}
    /// @param newTheta23 New \theta_{23}
    /// @param newTheta13 New \theta_{13}
    /// @param newDeltaCP New \delta_{cp}
    /// @param newDmsq21  New \Delta m^2_{21}
    /// @param newDmsq31  New \Delta m^2_{31}
    /// @param sinSquaredThetas If true, the \theta_{ij}'s will be interpreted as \sin^2(\theta_{ij}) values
    ///                         This will shortcut some of the computations performed by this propagator
    ///                         and speed up calculation time
    inline void setParameters(Tensor &newTheta12, Tensor &newTheta23, Tensor &newTheta13, Tensor &newDeltaCP,
                              Tensor &newDmsq21, Tensor &newDmsq31, bool sinSquaredThetas = false)
    {
        NT_PROFILE();

        deltaCP = newDeltaCP;
        dmsq21 = newDmsq21;
        dmsq31 = newDmsq31;

        interpretSinSquaredThetas = sinSquaredThetas;

        // if user has provided sin^2(theta_ij) values, we just use those, otherwise
        // we need to calculate them
        if (interpretSinSquaredThetas)
        {

            sinSqTheta12 = newTheta12;
            sinSqTheta13 = newTheta13;
            sinSqTheta23 = newTheta23;
        }
        else
        {

            theta12 = newTheta12;
            theta13 = newTheta13;
            theta23 = newTheta23;
        }
    }

    /// @brief Set the neutrino energies
    /// @param newEnergies The neutrino energies
    inline DPpropagator &setEnergies(Tensor &newEnergies) override
    {
        NT_PROFILE();

        _energies = newEnergies;
        probsRet = Tensor::zeros({_energies.getShape()[0], 3, 3}).requiresGrad(false);

        return *this;
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
    [[nodiscard]] const float &getDensity() const
    {
        NT_PROFILE();

        return _density;
    }

    /// @}

    /// @brief Calculate the oscilaltion probabilities for the current set of parameters
    ///        and energies
    [[nodiscard]] Tensor calculateProbs() override;

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

    Tensor sinSqTheta12 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor sinSqTheta13 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor sinSqTheta23 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

    Tensor deltaCP = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

    Tensor dmsq21 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor dmsq31 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

    Tensor probsRet = Tensor::zeros({1});

    int NRiterations;
    float _density{0.0};

    // whether to interpret user specified \theta_{ij} values as \sin^2(\theta_{ij})
    // allowing shortcut in calculations
    bool interpretSinSquaredThetas = false;
};

}; // namespace nuTens