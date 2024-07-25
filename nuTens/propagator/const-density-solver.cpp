#include <nuTens/propagator/const-density-solver.hpp>

void ConstDensityMatterSolver::calculateEigenvalues(const Tensor &energies, Tensor &eigenvectors, Tensor &eigenvalues)
{
    Tensor hamiltonian;
    hamiltonian.zeros({energies.getBatchDim(), nGenerations, nGenerations}, NTdtypes::kComplexFloat);

    for (int i = 0; i < nGenerations; i++)
    {
        for (int j = 0; j < nGenerations; j++)
        {
            hamiltonian.setValue({"...", i, j},
                                 Tensor::div(diagMassMatrix.getValue({0, i, j}), energies.getValue({"...", 0})) -
                                     electronOuter.getValue({i, j}));
        }
    }

    eigenvectors.zeros({1, nGenerations, nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);
    eigenvalues.zeros({1, nGenerations, nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);

    Tensor::eig(hamiltonian, eigenvectors, eigenvalues);
}