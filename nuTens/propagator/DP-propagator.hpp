#pragma once

#include <nuTens/tensors/tensor.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/propagator/constants.hpp>

namespace nuTens 
{

    static const Tensor one = Tensor::ones({1}).requiresGrad(false);

template <typename T>
struct fail : std::false_type 
{
};

/// @brief Solver based on Denton, Parke (https://arxiv.org/pdf/2405.02400)
/// assumes 3 flavour oscillations and dm^2_21 > 0
class DPpropagator : public Propagator
{

  public:

    DPpropagator(float baseline, bool antiNeutrino, float density, int NRiterations) 
    :
        Propagator(3, baseline, antiNeutrino),
        _NRiterations(NRiterations),
        _density(density)
    {};

    /// @{Setters

    inline void setTheta12(Tensor &newTheta12)
    {
        NT_PROFILE();
        
        _theta12 = newTheta12;
    }
    inline void setTheta23(Tensor &newTheta23)
    {
        NT_PROFILE();

        _theta23 = newTheta23;
    }
    inline void setTheta13(Tensor &newTheta13)
    {
        NT_PROFILE();

        _theta13 = newTheta13;
    }
    inline void setDeltaCP(Tensor &newDeltaCP)
    {
        NT_PROFILE();

        _deltaCP = newDeltaCP;
    }
    inline void setDmsp21(Tensor &newDmsq21)
    {
        NT_PROFILE();

        _dmsq21 = newDmsq21;
    }
    inline void setDmsq31(Tensor &newDmsq31)
    {
        NT_PROFILE();
        
        _dmsq31 = newDmsq31;
    }

    inline void setParameters(
        Tensor &newTheta12,
        Tensor &newTheta23,
        Tensor &newTheta13,
        Tensor &newDeltaCP,
        Tensor &newDmsq21,
        Tensor &newDmsq31
    )
    {
        NT_PROFILE();

        _theta12 = newTheta12;
        _theta23 = newTheta23;
        _theta13 = newTheta13;
        _deltaCP = newDeltaCP;
        _dmsq21 = newDmsq21;
        _dmsq31 = newDmsq31;

        // --------------------------------------------------------------- //
        // Calculate useful simple functions of the oscillation parameters //
        // --------------------------------------------------------------- //
        sinSqTheta12 = Tensor::pow(Tensor::sin(_theta12), 2.0);
        cosSqTheta12 = Tensor::pow(Tensor::cos(_theta12), 2.0);
        sinSqTheta13 = Tensor::pow(Tensor::sin(_theta13), 2.0);
        cosSqTheta13 = Tensor::pow(Tensor::cos(_theta13), 2.0);
        sinSqTheta23 = Tensor::pow(Tensor::sin(_theta23), 2.0);
        cosSqTheta23 = Tensor::pow(Tensor::cos(_theta23), 2.0);

        sinDeltaCP = Tensor::sin(_deltaCP);
        cosDeltaCP = Tensor::cos(_deltaCP);

        calculateIntermediate();
    }

    /// @brief Set the neutrino energies
    /// @param newEnergies The neutrino energies
    inline void setEnergies(Tensor &newEnergies) override
    {
        NT_PROFILE();

        _energies = newEnergies;
    }

    /// @}

    /// @brief Calculate the oscilaltion probabilities for the current set of parameters
    ///        and energies
    [[nodiscard]] Tensor calculateProbs();

    // shouldn't try to use a matter solver with this class since it internally
    // handles all matter effects
    template<typename T = bool>
    inline void setMatterSolver(const std::shared_ptr<BaseMatterSolver> &newSolver)
    {
        static_assert(fail<T>::value, "do not use for DP propagator");
    };

    // shouldn't use as this method requires us to directly set oscillation parameters
    template<typename T = bool>
    inline void setMixingMatrix(Tensor &newMatrix) 
    {
        static_assert(fail<T>::value, "do not use for DP propagator");
    };

    // shouldn't use as this method requires us to directly set oscillation parameters
    template<typename T = bool>
    inline void setMasses(Tensor &newMasses) 
    {
        static_assert(fail<T>::value, "do not use for DP propagator");
    };

    void calculateEigenvalues(Tensor &lambda1, Tensor &lambda2, Tensor &lambda3, Tensor &lambda21, Tensor &lambda31, Tensor &lambda32);

    /// Calculate intermediate some intermediate functions of the oscillation parameters
    /// only need to call this when new values are specified for osc parameters
    void calculateIntermediate();

  private:


    Tensor _theta12 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor _theta13 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor _theta23 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

    Tensor _deltaCP = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

    Tensor _dmsq21 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
    Tensor _dmsq31 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);


	Tensor sinSqTheta12;
	Tensor cosSqTheta12;
	Tensor sinSqTheta13;
	Tensor cosSqTheta13;
	Tensor sinSqTheta23;
	Tensor cosSqTheta23;

	Tensor sinDeltaCP;
	Tensor cosDeltaCP;

    Tensor Ue2sq;
    Tensor Ue3sq;
    
    Tensor Um3sq;
    
    Tensor Ut2sq;
    Tensor Jrr;
    Tensor Um2sq;
    Tensor Jmatter;
    Tensor Amatter;
    Tensor Dmsqee;
    
    Tensor Araw;
    Tensor See;
    Tensor Tmm;
    Tensor Tee;
    Tensor C;
    Tensor A;

    int _NRiterations;
    float _density;
};

}; // end namespace nuTens{