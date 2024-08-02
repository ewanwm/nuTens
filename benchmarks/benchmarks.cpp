
#include <benchmark/benchmark.h>
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/tensor.hpp>

Tensor buildPMNS(const Tensor &theta12, const Tensor &theta13, const Tensor &theta23, const Tensor &deltaCP)
{
    // set up the three matrices to build the PMNS matrix
    Tensor M1, M2, M3;
    M1.zeros({1, 3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);
    M2.zeros({1, 3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);
    M3.zeros({1, 3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);

    M1.setValue({0, 0, 0}, 1.0);
    M1.setValue({0, 1, 1}, Tensor::cos(theta23));
    M1.setValue({0, 1, 2}, Tensor::sin(theta23));
    M1.setValue({0, 2, 1}, -Tensor::sin(theta23));
    M1.setValue({0, 2, 2}, Tensor::cos(theta23));
    M1.requiresGrad(true);

    M2.setValue({0, 1, 1}, 1.0);
    M2.setValue({0, 0, 0}, Tensor::cos(theta13));
    M2.setValue({0, 0, 2}, Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(deltaCP, -1.0J))));
    M2.setValue({0, 2, 0}, -Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(deltaCP, 1.0J))));
    M2.setValue({0, 2, 2}, Tensor::cos(theta13));
    M2.requiresGrad(true);

    M3.setValue({0, 2, 2}, 1.0);
    M3.setValue({0, 0, 0}, Tensor::cos(theta12));
    M3.setValue({0, 0, 1}, Tensor::sin(theta12));
    M3.setValue({0, 1, 0}, -Tensor::sin(theta12));
    M3.setValue({0, 1, 1}, Tensor::cos(theta12));
    M3.requiresGrad(true);

    // Build PMNS
    Tensor PMNS = Tensor::matmul(M1, Tensor::matmul(M2, M3));
    PMNS.requiresGrad(true);

    return PMNS;
}

static void batchedOscProbs(const Propagator &prop, Tensor &energies, int batchSize, int nBatches)
{
    for (int _ = 0; _ < nBatches; _++)
    {
        // set random energy values
        for (int i = 0; i < batchSize; i++)
        {
            // set to random energy between 0 and 10000.0 MeV
            energies.setValue({i, 0}, ((float)std::rand() / (float)RAND_MAX) * 10000.0);
        }

        // calculate the osc probabilities
        // static_cast<void> to discard the return value that we're not supposed to discard :)
        static_cast<void>(prop.calculateProbs(energies).sum());
    }
}

static void BM_vacuumOscillations(benchmark::State &state)
{

    // set up the inputs
    Tensor energies;
    energies.zeros({state.range(0), 1}, NTdtypes::kFloat).requiresGrad(false);

    Tensor masses;
    masses.ones({1, 3}, NTdtypes::kFloat).requiresGrad(false);
    masses.setValue({0, 0}, 0.1);
    masses.setValue({0, 1}, 0.2);
    masses.setValue({0, 2}, 0.3);

    Tensor theta23, theta13, theta12, deltaCP;
    theta23.ones({1}, NTdtypes::kComplexFloat).requiresGrad(false).setValue({0}, 0.23);
    theta13.ones({1}, NTdtypes::kComplexFloat).requiresGrad(false).setValue({0}, 0.13);
    theta12.ones({1}, NTdtypes::kComplexFloat).requiresGrad(false).setValue({0}, 0.12);
    deltaCP.ones({1}, NTdtypes::kComplexFloat).requiresGrad(false).setValue({0}, 0.5);

    Tensor PMNS = buildPMNS(theta12, theta13, theta23, deltaCP);

    // set up the propagator
    Propagator vacuumProp(3, 100.0);
    vacuumProp.setPMNS(PMNS);
    vacuumProp.setMasses(masses);

    // seed the random number generator for the energies
    std::srand(123);

    for (auto _ : state)
    {
        // This code gets timed
        batchedOscProbs(vacuumProp, energies, state.range(0), state.range(1));
    }
}

// Register the function as a benchmark
BENCHMARK(BM_vacuumOscillations)->Name("Vacuum Osc")->Args({1 << 10, 1 << 10});
// Run the benchmark
BENCHMARK_MAIN();