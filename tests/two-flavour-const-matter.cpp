#include <nuTens/propagator/const-density-solver.hpp>

// Get absolute relative difference between two floats:
//   | (f1 - f2) / f1 |
float relativeDiff(float f1, float f2){
    return std::abs((f1 - f2) / f1);
}

int main(){
    
    float m1 = 1.0, m2 = 2.0;
    float energy = 100.0;
    float density = 2.6;

    // 'standard' 2 flavour mass effect calculation values from Barger et. al
    float lv = 4.0 * M_PI * energy / (m1*m1 - m2*m2);
    float lm = 2.0 * M_PI / (BaseMatterSolver::Groot2 * density);

    // set the tensors we will use to calculate matter eigenvalues
    Tensor masses;
    masses.ones({1, 2}, NTdtypes::kFloat).requiresGrad(false);
    masses.setValue({0, 0}, m1);
    masses.setValue({0, 1}, m2);
    masses.requiresGrad(true);

    Tensor energies;
    energies.ones({1, 1,1}, NTdtypes::kFloat).requiresGrad(false);
    energies.setValue({0, 0}, energy);
    energies.requiresGrad(true);

    std::cout << "value tensors created" << std::endl;

    ConstDensityMatterSolver solver(2, density);
    
    std::cout << "solver created" << std::endl;

    // test that Propagator gives expected oscillation probabilites for a range of thetas
    for( int i = 0; i <= 20; i++){

        float theta = ( -1.0 + 2.0 * (float)i / 20.0) * 0.49 * M_PI;

        // the modified angle due to matter effects from Barger et. al.
        float alpha = std::atan2(std::sin(2.0 * theta), (std::cos(2.0 * theta) - lv / lm)) / 2.0;
        float dM2 = (m1*m1 - m2*m2) * std::sqrt( 1.0 - 2.0 * (lv / lm) * std::cos(2.0 * theta) + (lv / lm) * (lv / lm));

        // construct the PMNS matrix for current theta value
        Tensor PMNS;
        PMNS.ones({1, 2, 2}, NTdtypes::kComplexFloat).requiresGrad(false);
        PMNS.setValue({0, 0,0}, std::cos(theta));
        PMNS.setValue({0, 0,1}, -std::sin(theta));
        PMNS.setValue({0, 1,0}, std::sin(theta));
        PMNS.setValue({0, 1,1}, std::cos(theta));
        PMNS.requiresGrad(true);

        solver.setPMNS(PMNS);

        solver.setMasses(masses);

        Tensor eigenVals, eigenVecs;

        solver.calculateEigenvalues(energies, eigenVecs, eigenVals);
        
        std::cout << "######## theta = " << theta << " ########" << std::endl;
        
        std::cout << "Barger eigenvecs: " << std::endl;
        std::cout << "dM2 = " << dM2 << std::endl;
        std::cout << "alpha = " << alpha << std::endl;
        std::cout << "  " << std::cos(alpha) << "  " << - std::sin(alpha) << std::endl;
        std::cout << "  " << std::sin(alpha) << "  " << std::cos(alpha) << std::endl;

        std::cout << "Solver eigenvals: " << std::endl;
        std::cout << eigenVals << std::endl;
        float calcV1 = eigenVals.getValue<float>({0, 0});
        float calcV2 = eigenVals.getValue<float>({0, 1});
        std::cout << "dM2 = " << (calcV1 - calcV2) * 2.0 * energy << std::endl;
        
        Tensor PMNSeff = Tensor::matmul(PMNS, eigenVecs);
        std::cout << "effective PMNS: " << std::endl;
        std::cout << PMNSeff << std::endl << std::endl;

        if ((   relativeDiff(PMNSeff.getValue<float>({0, 0,0}), std::cos(alpha)) > 0.00005)
            || (relativeDiff(PMNSeff.getValue<float>({0, 1,1}), std::cos(alpha)) > 0.00005)
            || (relativeDiff(PMNSeff.getValue<float>({0, 0,1}), -std::sin(alpha)) > 0.00005)
            || (relativeDiff(PMNSeff.getValue<float>({0, 1,0}), std::sin(alpha)) > 0.00005)
        ){
            std::cerr << std::endl;
            std::cerr << "ERROR: 2 flavour effective PMNS does not match Barger values for theta = " << theta << std::endl;
            std::cerr << std::endl;

            return 1;
        }

        if( relativeDiff(dM2, (calcV1 - calcV2) * 2.0 * energy) > 0.00005 ){
            std::cerr << std::endl;
            std::cerr << "ERROR: calculated effective delta M^2 value for 2 flavours does not match with Barger value for theta = " << theta << std::endl;
            std::cerr << std::endl;
            
            return 1;
        }
    }
}