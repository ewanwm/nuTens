#include <nuTens/propagator/propagator.hpp>

using namespace nuTens;

Tensor Propagator::calculateProbs()
{
    NT_PROFILE();

    Tensor ret;

    // if a matter solver was specified, use effective values for masses and mixing
    // matrix, otherwise just use the "raw" ones
    if (_matterSolver)
    {
        Tensor eigenVals =
            Tensor::zeros({1, _nGenerations, _nGenerations}, dtypes::kComplexFloat).requiresGrad(false);
        Tensor eigenVecs =
            Tensor::zeros({1, _nGenerations, _nGenerations}, dtypes::kComplexFloat).requiresGrad(false);

        _matterSolver->calculateEigenvalues(eigenVecs, eigenVals);
        Tensor effectiveMassesSq = Tensor::mul(eigenVals, Tensor::scale(_energies, 2.0));
        Tensor effectiveMixingMatrix = Tensor::matmul(_mixingMatrix, eigenVecs);

        ret = _calculateProbs(effectiveMassesSq, effectiveMixingMatrix);
    }

    else
    {
        ret = _calculateProbs(Tensor::mul(_masses, _masses), _mixingMatrix);
    }

    return ret;
}

Tensor Propagator::_calculateProbs(const Tensor &massesSq, const Tensor &mixingMatrix)
{
    NT_PROFILE();

    Tensor weightVector = Tensor::exp(
        Tensor::div(massesSq, _weightArgDenom));

    _weightMatrix.requiresGrad(false);
    for (int i = 0; i < _nGenerations; i++)
    {
        for (int j = 0; j < _nGenerations; j++)
        {
            _weightMatrix.setValue({"...", i, j}, weightVector.getValues({"...", j}));
        }
    }
    _weightMatrix.requiresGrad(true);

    Tensor sqrtProbabilities;
    
    if (_antiNeutrino) {
        sqrtProbabilities = Tensor::matmul(mixingMatrix, Tensor::transpose(Tensor::mul(mixingMatrix.conj(), _weightMatrix), 1, 2));
    }
    else {
        sqrtProbabilities = Tensor::matmul(mixingMatrix.conj(), Tensor::transpose(Tensor::mul(mixingMatrix, _weightMatrix), 1, 2));
    }
    return Tensor::pow(sqrtProbabilities.abs(), 2);
}
