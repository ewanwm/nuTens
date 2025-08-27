#include <nuTens/propagator/const-density-solver.hpp>

using namespace nuTens;

void ConstDensityMatterSolver::calculateEigenvalues(Tensor &eigenvectors, Tensor &eigenvalues)
{
    NT_PROFILE();

    buildHamiltonian();

    Tensor::eigh(hamiltonian, eigenvalues, eigenvectors);
}

void ConstDensityMatterSolver::buildHamiltonian()
{

    NT_PROFILE();

    hamiltonian.setValue({"..."}, (Tensor::div(diagMassMatrix, energiesRed) - electronOuter));
}

Tensor ConstDensityMatterSolver::getHamiltonian()
{

    NT_PROFILE();

    buildHamiltonian();

    return hamiltonian;
}

Tensor ConstDensityMatterSolver::getElectronOuterProduct()
{

    NT_PROFILE();

    buildElectronOuterProduct();

    return electronOuter;
}

void ConstDensityMatterSolver::buildElectronOuterProduct()
{

    NT_PROFILE();

    if (antiNeutrino)
    {
        electronOuter = Tensor::scale(
            Tensor::outer(mixingMatrix.getValues({0, 0, "..."}).conj(), mixingMatrix.getValues({0, 0, "..."})),
            -nuTens::constants::Groot2 * density);
    }

    else
    {
        electronOuter = Tensor::scale(
            Tensor::outer(mixingMatrix.getValues({0, 0, "..."}).conj(), mixingMatrix.getValues({0, 0, "..."})),
            nuTens::constants::Groot2 * density);
    }

    electronOuter.unsqueeze(0);
}
