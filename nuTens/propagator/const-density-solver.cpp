#include <nuTens/propagator/const-density-solver.hpp>

void ConstDensityMatterSolver::calculateEigenvalues(const Tensor &energies, Tensor &eigenvectors, Tensor &eigenvalues){

    Tensor hamiltonian;

    for ( int i = 0; i < nGenerations; i++){
        hamiltonian = Tensor::div(diagMassMatrix, energies) - electronOuter;
    }

    //std::cout << "Hamiltonian: " << std::endl;
    //std::cout << "real: " << hamiltonian.real() << std::endl;
    //std::cout << "imag: " << hamiltonian.imag() << std::endl;

    //std::cout << "mass term: " << Tensor::div(diagMassMatrix, energies) << std::endl;
    //std::cout << "PMNS term: " << Tensor::scale(electronOuter, Groot2*density) << std::endl;

    eigenvectors.zeros({1, nGenerations, nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);
    eigenvalues.zeros({1, nGenerations, nGenerations}, NTdtypes::kComplexFloat).requiresGrad(false);

    Tensor::eig(hamiltonian, eigenvalues, eigenvectors);

}