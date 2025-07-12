#include <nuTens/propagator/const-density-solver.hpp>

void ConstDensityMatterSolver::calculateEigenvalues(Tensor &eigenvectors, Tensor &eigenvalues)
{
    NT_PROFILE();
    
    for (int i = 0; i < nGenerations; i++)
    {
        for (int j = 0; j < nGenerations; j++)
        {
            hamiltonian.setValue({"...", i, j},
                                 Tensor::div(diagMassMatrix.getValues({i, j}), energiesRed) -
                                     electronOuter.getValues({i, j}));
        }
    }

    Tensor::eigh(hamiltonian, eigenvectors, eigenvalues);
}