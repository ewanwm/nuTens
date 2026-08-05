#include <nuTens/propagator/const-density-solver.hpp>

using namespace nuTens;

void ConstDensityMatterSolver::calculateEigenvalues(EigenvecTensor &eigenvectors, EigenvalTensor &eigenvalues)
{
    NT_PROFILE();

    buildHamiltonian();

    Tensor::eigh(hamiltonian, eigenvalues, eigenvectors);
}

void ConstDensityMatterSolver::buildHamiltonian()
{

    NT_PROFILE();

    if (!energies.isInitialised())
    {
        throw std::runtime_error("No energies set for matter solver!!");
    }
    if (!masses.isInitialised())
    {
        throw std::runtime_error("No masses set for matter solver!!");
    }

    Tensor energiesRed = energies.getValues({"..."}).unsqueeze(-1).unsqueeze(-1);

    hamiltonian.setValue({"..."}, (Tensor::div(diagMassMatrix, energiesRed) - getElectronOuterProduct()));
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

    if (!mixingMatrix.isInitialised())
    {
        throw std::runtime_error("No mixing matrix set for matter solver!!");
    }

    Tensor electronRow = mixingMatrix.getValues({0, 0, "..."});

    if (antiNeutrino)
    {
        electronOuter = -nuTens::constants::Groot2 * density * Tensor::outer(electronRow.conj(), electronRow);
    }

    else
    {
        electronOuter = nuTens::constants::Groot2 * density * Tensor::outer(electronRow.conj(), electronRow);
    }
}
