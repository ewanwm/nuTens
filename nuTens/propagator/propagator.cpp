#include <nuTens/propagator/propagator.hpp>

using namespace nuTens;

Tensor Propagator::calculateProbs()
{
    NT_PROFILE();

    Tensor ret;
    Propagator::MassSqTensor massesSq;
    Propagator::MixingMatrixTensor mixingMatrix;

    // if a matter solver was specified, use effective values for masses and mixing
    // matrix, otherwise just use the "raw" ones
    if (_matterSolver)
    {
        Tensor eigenVals = Tensor::zeros({1, _nGenerations, _nGenerations}, dtypes::kComplexFloat).requiresGrad(false);
        Tensor eigenVecs = Tensor::zeros({1, _nGenerations, _nGenerations}, dtypes::kComplexFloat).requiresGrad(false);

        _matterSolver->calculateEigenvalues(eigenVecs, eigenVals);
        massesSq = Propagator::MassSqTensor(eigenVals * _energies * 2.0);
        mixingMatrix = Propagator::MixingMatrixTensor(Tensor::matmul(_mixingMatrix, eigenVecs));
    }

    else
    {
        massesSq = Propagator::MassSqTensor(_masses * _masses);
        mixingMatrix = Propagator::MixingMatrixTensor(_mixingMatrix);
    }

    return _calculateProbs(massesSq, mixingMatrix);
}

Tensor Propagator::_calculateProbs(const Propagator::MassSqTensor &massesSq,
                                   const Propagator::MixingMatrixTensor &mixingMatrix)
{
    NT_PROFILE();

    // basically exp { - i m^2 L / 2 E }
    Tensor weightVector = Tensor::exp(Tensor::div(massesSq, _weightArgDenom));

    // turn it into a matrix with the right shape
    _weightMatrix.requiresGrad(false);
    for (int i = 0; i < _nGenerations; i++)
    {
        _weightMatrix.setValue({"...", i}, weightVector);
    }
    _weightMatrix.requiresGrad(true);

    Tensor matrixA;
    Tensor matrixB;

    if (_antiNeutrino)
    {
        matrixA = Tensor::mul(mixingMatrix.conj(), Tensor::transpose(_weightMatrix, 1, 2));
        matrixB = Tensor::transpose(mixingMatrix, 1, 2);
    }
    else
    {
        matrixA = Tensor::mul(mixingMatrix, Tensor::transpose(_weightMatrix, 1, 2));
        matrixB = Tensor::transpose(mixingMatrix.conj(), 1, 2);
    }

    Tensor sqrtProbabilities = Tensor::matmul(matrixA, matrixB);

    return Tensor::pow(sqrtProbabilities.abs(), 2);
}
