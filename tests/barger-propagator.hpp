#include <nuTens/propagator/constants.hpp>
#include <math.h>
#include <iostream>

// make a simple propagator based on Barger:
// [Vernon D. Barger, K. Whisnant, S. Pakvasa, and R. J. N. Phillips. Matter Effects on Three-Neutrino Oscillations. Phys. Rev. D, 22:2718, 1980.]
// for the purposes of testing so we can compare
// our propagators against something solid.

// doesn't need to be very efficient or very fancy.
// goal is more to be clear and simple so we can be confident in the results.
// also don't want to use any fancy classes like tensors, 
// just want to use vv simple c++ standard objects so is independent of the rest of the nuTens libraries.

namespace Testing{
    
    class TwoFlavourBarger{

    public:
        
        // set the parameters of this propagator
        // negative density values will be interpreted as propagating in vacuum
        inline void setParams(float m1, float m2, float theta, float baseline, float density = -999.9){
            _m1 = m1;
            _m2 = m2;
            _theta = theta;
            _baseline = baseline;
            _density = density;
        };

        
        // characteristic length in vacuum   
        inline float lv( float energy ) { return 4.0 * M_PI * energy / (_m1*_m1 - _m2*_m2); }
        
        // characteristic length in matter
        inline float lm() { return 2.0 * M_PI / (Constants::Groot2 * _density); }

        // calculate the modified rotation angle
        inline float calculateEffectiveAngle( float energy ){
            if (_density > 0.0)
                return std::atan2(std::sin(2.0 * _theta), (std::cos(2.0 * _theta) - lv(energy) / lm())) / 2.0;
            else
                return _theta;
        }
        
        // calculate the modified delta M^2
        inline float calculateEffectiveDm2( float energy ){
            if (_density > 0.0)
                return (_m1*_m1 - _m2*_m2) * std::sqrt( 1.0 - 2.0 * (lv(energy) / lm()) * std::cos(2.0 * _theta) + (lv(energy) / lm()) * (lv(energy) / lm()));
            else
                return (_m1*_m1 - _m2*_m2);
        }

        
        // get the good old 2 flavour PMNS matrix entries
        inline float getPMNSelement(float energy, int alpha, int beta){
            if ( (alpha > 1 || alpha < 0) || (beta > 1 || beta < 0)) {
                std::cerr << "ERROR: TwoFlavourBarger class only supports flavour indices of 0 or 1" << std::endl;
                std::cerr << "       you supplied alpha = " << alpha << ", " << "beta = " << beta << std::endl;
                std::cerr << "       " << __FILE__ << ": " << __LINE__ << std::endl;  
            }  

            float gamma = calculateEffectiveAngle( energy );

            if ( alpha == 0 && beta == 0 )
                return std::cos(gamma);
            else if (alpha == 1 && beta == 1)
                return std::cos(gamma);
            else if (alpha == 0 && beta == 1) 
                return -std::sin(gamma);
            else if (alpha == 1 && beta == 0)
                return std::sin(gamma);

            else{
                std::cerr << "ERROR: how did you get here????" << std::endl;
                std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
                throw;
            } 
        }

        // get the good old 2 flavour vacuum oscillation probability
        inline float calculateProb(float energy, int alpha, int beta){
            if ( (alpha > 1 || alpha < 0) || (beta > 1 || beta < 0)) {
                std::cerr << "ERROR: TwoFlavourBarger class only supports flavour indices of 0 or 1" << std::endl;
                std::cerr << "       you supplied alpha = " << alpha << ", " << "beta = " << beta << std::endl;
                std::cerr << "       " << __FILE__ << ": " << __LINE__ << std::endl;  
                throw;
            }

            // get the effective oscillation parameters
            // if in vacuum (_density <= 0.0) these should just return the "raw" values
            float gamma = calculateEffectiveAngle( energy );
            float dM2 = calculateEffectiveDm2( energy );

            // now get the actual probabilities
            float sin2Gamma = std::sin(2.0 * gamma);
            float sinPhi = std::sin(dM2 * _baseline / (4.0 * energy) );

            float offAxis = sin2Gamma * sin2Gamma * sinPhi * sinPhi;
            float onAxis = 1.0 - offAxis;

            if ( alpha == beta ) 
                return onAxis;
            else 
                return offAxis;

        }


    private:
        // oscillation parameters
        float _m1;
        float _m2;
        float _theta;

        // characteristic lengths in vacuum and matter   
        float _lv;
        float _lm;

        // other parameters
        float _baseline;
        float _density;

    };

}