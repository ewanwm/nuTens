
#include <benchmark/benchmark.h>
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/propagator/DP-propagator.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <nuTens/propagator/pmns-matrix.hpp>

using namespace nuTens;

// The random seed to use for the RNG
// want this to be fixed for reproducibility
const int randSeed = 123;

/// get random double between 0.0 and 1.0
double randomDouble()
{
    return (double)rand() / (RAND_MAX + 1.);
}

static void batchedOscProbs(
    Propagator &prop, 
    PMNSmatrix &matrix,
    AccessedTensor<float, 2, dtypes::kCPU> &masses, 
    long nBatches)
{
    for (int _ = 0; _ < nBatches; _++)
    {

        // set random values of the oscillation parameters
        masses.setValue(randomDouble(), 0, 0);
        masses.setValue(randomDouble(), 0, 1);
        masses.setValue(randomDouble(), 0, 2);

        matrix.setParameterValues(
            /*theta12=*/randomDouble(),
            /*theta13=*/randomDouble(),
            /*theta23=*/randomDouble(),
            /*deltaCP=*/randomDouble() * 2.0 * M_PI
        );

        prop.setMixingMatrix(matrix.build());
        prop.setMasses(masses);

        // calculate the osc probabilities
        // static_cast<void> to discard the return value that we're not supposed to discard :)
        static_cast<void>(prop.calculateProbs().sum());
    }
}

static void BM_vacuumOscillations(benchmark::State &state)
{

    NT_PROFILE_BEGINSESSION("Benchmark-vacuum-oscillations");
    NT_PROFILE();

    // make some random test energies
    Tensor energies =
        Tensor::scale(Tensor::rand({state.range(0), 1}).dType(dtypes::kFloat).requiresGrad(false), 10000.0).hasBatchDim(true) +
        Tensor({100.0});

    energies = energies.hasBatchDim(true);

    // set up the inputs
    auto masses = AccessedTensor<float, 2, dtypes::kCPU>::zeros({1, 3});
    PMNSmatrix PMNS;

    // set up the propagator
    Propagator vacuumProp(3, 295000.0);
    vacuumProp.setEnergies(energies);

    // seed the random number generator for the energies
    std::srand(randSeed);

    // linter gets angry about this as _ is never used :)))
    // NOLINTNEXTLINE
    for (auto _ : state)
    {
        // This code gets timed
        batchedOscProbs(vacuumProp, PMNS, masses, state.range(1));
    }

    NT_PROFILE_ENDSESSION();
}

static void BM_constMatterOscillations(benchmark::State &state)
{
    
    NT_PROFILE_BEGINSESSION("Benchmark-const-density-oscillations");

    NT_PROFILE();
    
    // make some random test energies
    Tensor energies =
        Tensor::scale(Tensor::rand({state.range(0), 1}).dType(dtypes::kFloat).requiresGrad(false), 10000.0) +
        Tensor({100.0});

    energies = energies.hasBatchDim(true);

    // set up the inputs
    auto masses = AccessedTensor<float, 2, dtypes::kCPU>::zeros({1, 3});
    PMNSmatrix PMNS;

    // set up the propagator
    Propagator matterProp(3, 295000.0);
    auto matterSolver = std::make_shared<ConstDensityMatterSolver>(3, 2.6);
    matterProp.setMatterSolver(matterSolver);
    matterProp.setEnergies(energies);

    // seed the random number generator for the energies
    std::srand(randSeed);

    // linter gets angry about this as _ is never used :)))
    // NOLINTNEXTLINE
    for (auto _ : state)
    {
        // This code gets timed
        batchedOscProbs(matterProp, PMNS, masses, state.range(1));
    }

    NT_PROFILE_ENDSESSION();
}


static void BM_DPpropOscillations(benchmark::State &state)
{
    
    NT_PROFILE_BEGINSESSION("Benchmark-DP-propagator");

    NT_PROFILE();
    
    // make some random test energies
    Tensor energies =
        Tensor::scale(Tensor::rand({state.range(0), 1}).dType(dtypes::kFloat).requiresGrad(false), 10000.0) +
        Tensor({100.0});

    energies = energies.hasBatchDim(true);
    
    auto dmsq21 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false);
    auto dmsq31 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false);
    auto theta23 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false);
    auto theta13 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false);
    auto theta12 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false);
    auto deltaCP = Tensor::zeros({1}).dType(dtypes::kComplexFloat).requiresGrad(false);

    // set up the propagator
    DPpropagator dpProp(3, 295000.0, 2.6, 5);
    
    dpProp.setEnergies(energies);

    // seed the random number generator for the energies
    std::srand(randSeed);

    // linter gets angry about this as _ is never used :)))
    // NOLINTNEXTLINE
    for (auto _ : state)
    {
        // This code gets timed
        for (int _ = 0; _ < state.range(1); _++)
        {
            // set random values of the oscillation parameters
            dmsq21.setValue(randomDouble(), 0);
            dmsq31.setValue(randomDouble(), 0);

            theta23.setValue(randomDouble(), 0);
            theta13.setValue(randomDouble(), 0);
            theta12.setValue(randomDouble(), 0);

            deltaCP.setValue({0}, Tensor::scale(Tensor::rand({1}), 2.0 * 3.1415));

            dpProp.setParameters(theta12, theta23, theta13, deltaCP, dmsq21, dmsq31);

            // calculate the osc probabilities
            // static_cast<void> to discard the return value that we're not supposed to discard :)
            static_cast<void>(dpProp.calculateProbs().sum());
        }
    }
    NT_PROFILE_ENDSESSION();
}


// Register the function as a benchmark
// NOLINTNEXTLINE
BENCHMARK(BM_vacuumOscillations)->Name("Vacuum Oscillations")->Args({1 << 10, 1 << 10});

// Register the function as a benchmark
// NOLINTNEXTLINE
BENCHMARK(BM_constMatterOscillations)->Name("Const Density Oscillations")->Args({1 << 10, 1 << 10});

// Register the function as a benchmark
// NOLINTNEXTLINE
BENCHMARK(BM_DPpropOscillations)->Name("DP Propagator Const Density Oscillations")->Args({1 << 10, 1 << 10});

// Run the benchmark
// NOLINTNEXTLINE
BENCHMARK_MAIN();
