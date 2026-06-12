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
        Tensor eigenVals = Tensor::zeros({1, _nGenerations, _nGenerations}, dtypes::kComplexFloat).requiresGrad(false);
        Tensor eigenVecs = Tensor::zeros({1, _nGenerations, _nGenerations}, dtypes::kComplexFloat).requiresGrad(false);

        _matterSolver->calculateEigenvalues(eigenVecs, eigenVals);
        Tensor effectiveMassesSq = Tensor::mul(eigenVals, Tensor::scale(_energies, 2.0));
        Tensor effectiveMixingMatrix = Tensor::matmul(_mixingMatrix, eigenVecs);

        ret = _calculateProbs(effectiveMassesSq, effectiveMixingMatrix);
    }

    else
    {
        ret = _calculateProbs(_masses * _masses, _mixingMatrix);
    }

    return ret;
}

Tensor Propagator::_calculateProbs(const Tensor &massesSq, const Tensor &mixingMatrix)
{
    NT_PROFILE();

    Tensor weightArgDenom =
        _energies * 2.0 * std::complex<float>(1.0) / (std::complex<float>(-1.0J) * _baseline * constants::twoPi);

    // basically exp { - i m^2 L / 2 E }
    Tensor weightVector = Tensor::exp(massesSq / weightArgDenom);

    // turn it into a matrix with the right shape
    for (int i = 0; i < _nGenerations; i++)
    {
        _weightMatrix.setValue({"...", i}, weightVector);
    }

    Tensor matrixA;
    Tensor matrixB;

    if (_antiNeutrino)
    {
        matrixA = mixingMatrix.conj() * Tensor::transpose(_weightMatrix, 1, 2);
        matrixB = Tensor::transpose(mixingMatrix, 1, 2);
    }
    else
    {
        matrixA = mixingMatrix * Tensor::transpose(_weightMatrix, 1, 2);
        matrixB = Tensor::transpose(mixingMatrix.conj(), 1, 2);
    }

    Tensor sqrtProbabilities = Tensor::matmul(matrixA, matrixB);

    return Tensor::pow(sqrtProbabilities.abs(), 2);
}
