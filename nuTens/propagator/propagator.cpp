#include <nuTens/propagator/propagator.hpp>


Tensor Propagator::calculateProbs(const Tensor &energies) const {
    
    // if a matter solver was specified, use effective values for masses and PMNS matrix, otherwise just use the "raw" ones
    if( _matterSolver != nullptr ){

        Tensor eigenVals, eigenVecs;
        _matterSolver->calculateEigenvalues(energies, eigenVecs, eigenVals);
        Tensor effectiveMassesSq = Tensor::mul(eigenVals, Tensor::scale(energies, 2.0));
        Tensor effectivePMNS = Tensor::matmul(_PMNSmatrix, eigenVecs);

        return _calculateProbs(energies, effectiveMassesSq, effectivePMNS);
    }

    else{
        return _calculateProbs(energies, Tensor::mul(_masses, _masses), _PMNSmatrix);
    }
}

Tensor Propagator::_calculateProbs(const Tensor &energies, const Tensor &massesSq, const Tensor &PMNS) const {
    Tensor weightMatrix;
    weightMatrix.ones({1,_nGenerations, _nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);
    
    for(int i = 0; i < _nGenerations; i++){
        weightMatrix.setValue({0, i, "..."}, Tensor::exp(Tensor::div(Tensor::scale(massesSq, -1.0j * _baseline), Tensor::scale(energies, 2.0))));
    }
    weightMatrix.requiresGrad(true);

    Tensor sqrtProbabilities = Tensor::matmul(PMNS.conj(), Tensor::transpose(Tensor::mul(PMNS, weightMatrix), 1, 2));
    
    return Tensor::mul(sqrtProbabilities.abs(), sqrtProbabilities.abs());
}
