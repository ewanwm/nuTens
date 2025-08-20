#include <nuTens/propagator/const-density-solver.hpp>

using namespace nuTens;

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

    Tensor::eigh(hamiltonian, eigenvalues, eigenvectors);
}

void ConstDensityMatterSolver::buildElectronOuterProduct() 
{

    NT_PROFILE();

    if (antiNeutrino)
    {
        electronOuter =
            Tensor::scale(Tensor::outer(mixingMatrix.getValues({0, 0, "..."}).conj(), mixingMatrix.getValues({0, 0, "..."})),
                          -nuTens::constants::Groot2 * density);
        
    }

    else 
    {
        electronOuter =
            Tensor::scale(Tensor::outer(mixingMatrix.getValues({0, 0, "..."}), mixingMatrix.getValues({0, 0, "..."}).conj()),
                          nuTens::constants::Groot2 * density);
    }
}
