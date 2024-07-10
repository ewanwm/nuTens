#include <nuTens/propagator/propagator.cpp>

// Get absolute relative difference between two floats:
//   | (f1 - f2) / f1 |
float relativeDiff(float f1, float f2){
    return std::abs((f1 - f2) / f1);
}

int main(){

    float m1 = 0.1, m2 = 0.5;
    float energy = 1.0;
    float baseline = 0.5;

    Tensor masses;
    masses.ones({1, 2}, NTdtypes::kFloat).requiresGrad(false);
    masses.setValue({0, 0}, m1);
    masses.setValue({0, 1}, m2);
    masses.requiresGrad(true);

    Tensor energies;
    energies.ones({1, 1}, NTdtypes::kFloat).requiresGrad(false);
    energies.setValue({0, 0}, energy);
    energies.requiresGrad(true);

    Propagator propagator(2, baseline);
    propagator.setMasses(masses);
    propagator.setEnergies(energies);

    
    float theta = -M_PI;

    // test that Propagator gives expected oscillation probabilites for a range of thetas
    for( int i = 0; i < 20; i++){

        theta += ( 2* M_PI / 20);

        // construct the PMNS matrix for current theta value
        Tensor PMNS;
        PMNS.ones({1, 2, 2}, NTdtypes::kComplexFloat).requiresGrad(false);
        PMNS.setValue({0, 0,0}, std::cos(theta));
        PMNS.setValue({0, 0,1}, -std::sin(theta));
        PMNS.setValue({0, 1,0}, std::sin(theta));
        PMNS.setValue({0, 1,1}, std::cos(theta));
        PMNS.requiresGrad(true);

        propagator.setPMNS(PMNS);

        Tensor probabilities = propagator.calculateProbs(energies);

        float sin2Theta = std::sin(2.0 * theta);
        float sinPhi = std::sin((m1 *m1 - m2 * m2) * baseline / (4.0 * energy) );

        float offAxis = sin2Theta * sin2Theta * sinPhi * sinPhi;
        float onAxis = 1.0 - offAxis;

        if ((relativeDiff(probabilities.getValue<float>({0, 0,0}), onAxis) > 0.00005)
            || (relativeDiff(probabilities.getValue<float>({0, 1,1}), onAxis) > 0.00005)
            || (relativeDiff(probabilities.getValue<float>({0, 0,1}), offAxis) > 0.00005)
            || (relativeDiff(probabilities.getValue<float>({0, 1,0}), offAxis) > 0.00005)
        ){
            std::cerr << "ERROR: 2 flavour probabilities from Propagator do not match expected values for theta = " << theta << std::endl;
            std::cerr << "Calculated probabilities: " << std::endl;
            std::cerr << probabilities << std::endl;

            std::cerr << std::endl;
            std::cerr << "'True' probabilities: " << std::endl;
            std::cerr << "  " << onAxis << "  " << offAxis << std::endl;
            std::cerr << "  " << offAxis << "  " << onAxis << std::endl;
            std::cerr << std::endl;
            std::cerr << "sin(2 theta) = " << sin2Theta << std::endl;
            std::cerr << "sin(Phi) = " << sinPhi << std::endl;

            return 1;
        }
    }

}