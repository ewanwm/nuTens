#include <nuTens/propagator/vacuum-propagator.hpp>


Tensor Propagator::calculateProbs(const Tensor &energies){
    Tensor weightMatrix;
    weightMatrix.ones({1,nGenerations, nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);
    
    for(int i = 0; i < nGenerations; i++){
        weightMatrix.setValue({0, i, "..."}, Tensor::exp(Tensor::div(Tensor::scale(Tensor::mul(masses, masses), -1.0j * baseline), Tensor::scale(energies, 2.0))));
    }
    weightMatrix.requiresGrad(true);

    Tensor sqrtProbabilities = Tensor::matmul(PMNSmatrix.conj(), Tensor::transpose(Tensor::mul(PMNSmatrix, weightMatrix), 1, 2));
    Tensor probabilities = Tensor::mul(sqrtProbabilities.abs(), sqrtProbabilities.abs());

    return probabilities;
}
