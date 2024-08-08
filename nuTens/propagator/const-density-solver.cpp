#include <nuTens/propagator/const-density-solver.hpp>

void ConstDensityMatterSolver::calculateEigenvalues(const Tensor &energies, Tensor &eigenvectors, Tensor &eigenvalues)
{
    NT_PROFILE();
    Tensor hamiltonian = Tensor::zeros({energies.getBatchDim(), nGenerations, nGenerations}, NTdtypes::kComplexFloat);

    for (int i = 0; i < nGenerations; i++)
    {
        for (int j = 0; j < nGenerations; j++)
        {
            hamiltonian.setValue({"...", i, j},
                                 Tensor::div(diagMassMatrix.getValue({i, j}), energies.getValue({"...", 0})) -
                                     electronOuter.getValue({i, j}));
        }
    }

    Tensor::eig(hamiltonian, eigenvectors, eigenvalues);
}