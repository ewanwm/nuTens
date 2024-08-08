#include <nuTens/propagator/const-density-solver.hpp>
#include <tests/barger-propagator.hpp>
#include <tests/test-utils.hpp>

using namespace Testing;

int main()
{
    NT_PROFILE_BEGINSESSION("two-flavour-const-matter-test");

    NT_PROFILE();

    float m1 = 1.0;
    float m2 = 2.0;
    float energy = 100.0;
    float density = 2.6;

    // set the tensors we will use to calculate matter eigenvalues
    Tensor masses = Tensor({m1, m2}, NTdtypes::kFloat).addBatchDim().requiresGrad(true);

    Tensor energies = Tensor::ones({1, 1, 1}, NTdtypes::kFloat).requiresGrad(false);
    energies.setValue({0, 0}, energy);
    energies.requiresGrad(true);

    std::cout << "value tensors created" << std::endl;

    ConstDensityMatterSolver tensorSolver(2, density);

    std::cout << "tensorSolver created" << std::endl;

    TwoFlavourBarger bargerProp{};

    // test that Propagator gives expected oscillation probabilites for a range
    // of thetas
    for (int i = 0; i <= 20; i++)
    {
        float theta = (-1.0 + 2.0 * (float)i / 20.0) * 0.49 * M_PI;

        bargerProp.setParams(m1, m2, theta, /*baseline=*/-999.9, density);

        // construct the PMNS matrix for current theta value
        Tensor PMNS = Tensor::ones({1, 2, 2}, NTdtypes::kComplexFloat).requiresGrad(false);
        PMNS.setValue({0, 0, 0}, std::cos(theta));
        PMNS.setValue({0, 0, 1}, std::sin(theta));
        PMNS.setValue({0, 1, 0}, -std::sin(theta));
        PMNS.setValue({0, 1, 1}, std::cos(theta));
        PMNS.requiresGrad(true);

        tensorSolver.setPMNS(PMNS);
        tensorSolver.setMasses(masses);

        Tensor eigenVals;
        Tensor eigenVecs;
        tensorSolver.calculateEigenvalues(energies, eigenVecs, eigenVals);

        std::cout << "######## theta = " << theta << " ########" << std::endl;

        // first check that the effective dM^2 from the tensor solver is what we
        // expect
        std::cout << "tensorSolver eigenvals: " << std::endl;
        std::cout << eigenVals << std::endl;
        auto calcV1 = eigenVals.getValue<float>({0, 0});
        auto calcV2 = eigenVals.getValue<float>({0, 1});
        float effDm2 = (calcV1 - calcV2) * 2.0 * energy;

        TEST_EXPECTED(effDm2, bargerProp.calculateEffectiveDm2(energy),
                      "effective dM^2 for theta == " + std::to_string(theta), 0.00001)

        // now check the actual PMNS matrix entries
        Tensor PMNSeff = Tensor::matmul(PMNS, eigenVecs);
        std::cout << "effective PMNS: " << std::endl;
        std::cout << PMNSeff << std::endl << std::endl;

        TEST_EXPECTED(PMNSeff.getValue<float>({0, 0, 0}), bargerProp.getPMNSelement(energy, 0, 0),
                      "PMNS[0,0] for theta == " + std::to_string(theta), 0.00001)

        TEST_EXPECTED(PMNSeff.getValue<float>({0, 1, 1}), bargerProp.getPMNSelement(energy, 1, 1),
                      "PMNS[1,1] for theta == " + std::to_string(theta), 0.00001)

        TEST_EXPECTED(PMNSeff.getValue<float>({0, 0, 1}), bargerProp.getPMNSelement(energy, 0, 1),
                      "PMNS[0,1] for theta == " + std::to_string(theta), 0.00001)

        TEST_EXPECTED(PMNSeff.getValue<float>({0, 1, 0}), bargerProp.getPMNSelement(energy, 1, 0),
                      "PMNS[1,0] for theta == " + std::to_string(theta), 0.00001)

        std::cout << "###############################" << std::endl << std::endl;
    }

    NT_PROFILE_ENDSESSION();
}