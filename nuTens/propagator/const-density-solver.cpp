#include <nuTens/propagator/const-density-solver.hpp>

void ConstDensityMatterSolver::calculateEigenvalues(const Tensor &energies, Tensor &eigenvectors, Tensor &eigenvalues){

    Tensor hamiltonian;

    for ( int i = 0; i < nGenerations; i++){
        hamiltonian = Tensor::div(diagMassMatrix, energies) - electronOuter;
    }

    eigenvectors.zeros({1, nGenerations, nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);
    eigenvalues.zeros({1, nGenerations, nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);

    Tensor::eig(hamiltonian, eigenvalues, eigenvectors);

}