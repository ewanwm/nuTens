#include <nuTens/propagator/const-density-solver.hpp>

using namespace nuTens;

void ConstDensityMatterSolver::calculateEigenvalues(Tensor &eigenvectors, Tensor &eigenvalues)
{
    NT_PROFILE();

    hamiltonian.setValue({"..."}, (Tensor::div(diagMassMatrix, energiesRed) - electronOuter));

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
<<<<<<< HEAD
}
=======

    electronOuter.unsqueeze(0);
}
>>>>>>> bd1b1ae (now create hamiltonian with more tensor-y operations rather than in for loop, hopefully saving some time)
