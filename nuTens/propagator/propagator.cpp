#include <nuTens/propagator/propagator.hpp>

Tensor Propagator::calculateProbs(const Tensor &energies) const
{
    Tensor ret;

    // if a matter solver was specified, use effective values for masses and PMNS
    // matrix, otherwise just use the "raw" ones
    if (_matterSolver != nullptr)
    {
        Tensor eigenVals;
        Tensor eigenVecs;
        _matterSolver->calculateEigenvalues(energies, eigenVecs, eigenVals);
        Tensor effectiveMassesSq = Tensor::mul(eigenVals, Tensor::scale(energies, 2.0));
        Tensor effectivePMNS = Tensor::matmul(_PMNSmatrix, eigenVecs);

        ret = _calculateProbs(energies, effectiveMassesSq, effectivePMNS);
    }

    else
    {
        ret = _calculateProbs(energies, Tensor::mul(_masses, _masses), _PMNSmatrix);
    }

    return ret;
}

Tensor Propagator::_calculateProbs(const Tensor &energies, const Tensor &massesSq, const Tensor &PMNS) const
{
    Tensor weightMatrix;
    weightMatrix.ones({energies.getBatchDim(), _nGenerations, _nGenerations}, NTdtypes::kComplexFloat)
        .requiresGrad(false);

    Tensor weightVector =
        Tensor::exp(Tensor::div(Tensor::scale(massesSq, -1.0J * _baseline), Tensor::scale(energies, 2.0)));

    for (int i = 0; i < _nGenerations; i++)
    {
        for (int j = 0; j < _nGenerations; j++)
        {
            weightMatrix.setValue({"...", i, j}, weightVector.getValue({"...", j}));
        }
    }
    weightMatrix.requiresGrad(true);

    Tensor sqrtProbabilities = Tensor::matmul(PMNS.conj(), Tensor::transpose(Tensor::mul(PMNS, weightMatrix), 1, 2));

    return Tensor::mul(sqrtProbabilities.abs(), sqrtProbabilities.abs());
}
