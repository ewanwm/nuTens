
#include <benchmark/benchmark.h>
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/tensor.hpp>

// The random seed to use for the RNG
// want this to be fixed for reproducibility
const int randSeed = 123;

// set the PMNS parameters to use
// Will very likely change the benchmark so that energies are fixed
// and these get randomised but for now just set them here
const float m1 = 0.1;
const float m2 = 0.2;
const float m3 = 0.3;

const float th12 = 0.12;
const float th23 = 0.23;
const float th13 = 0.13;

const float dcp = 0.5;

Tensor buildPMNS(const Tensor &theta12, const Tensor &theta13, const Tensor &theta23, const Tensor &deltaCP)
{
    // set up the three matrices to build the PMNS matrix
    Tensor M1 = Tensor::zeros({1, 3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);
    Tensor M2 = Tensor::zeros({1, 3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);
    Tensor M3 = Tensor::zeros({1, 3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);

    M1.setValue({0, 0, 0}, 1.0);
    M1.setValue({0, 1, 1}, Tensor::cos(theta23));
    M1.setValue({0, 1, 2}, Tensor::sin(theta23));
    M1.setValue({0, 2, 1}, -Tensor::sin(theta23));
    M1.setValue({0, 2, 2}, Tensor::cos(theta23));
    M1.requiresGrad(true);

    M2.setValue({0, 1, 1}, 1.0);
    M2.setValue({0, 0, 0}, Tensor::cos(theta13));
    std::complex<float> i(0.0, 1.0);
    M2.setValue({0, 0, 2}, Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(deltaCP, -i))));
    M2.setValue({0, 2, 0}, -Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(deltaCP, i))));
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

static void batchedOscProbs(const Propagator &prop, long batchSize, long nBatches)
{
    for (int _ = 0; _ < nBatches; _++)
    {

        Tensor energies =
            Tensor::scale(Tensor::rand({batchSize, 1}).dType(NTdtypes::kFloat).requiresGrad(false), 10000.0) +
            Tensor({100.0});

        // calculate the osc probabilities
        // static_cast<void> to discard the return value that we're not supposed to discard :)
        static_cast<void>(prop.calculateProbs(energies).sum());
    }
}

static void BM_vacuumOscillations(benchmark::State &state)
{

    // set up the inputs
    Tensor masses = Tensor({m1, m2, m3}, NTdtypes::kFloat).requiresGrad(false).addBatchDim();

    Tensor theta23 = Tensor({th23}).dType(NTdtypes::kComplexFloat).requiresGrad(false);
    Tensor theta13 = Tensor({th13}).dType(NTdtypes::kComplexFloat).requiresGrad(false);
    Tensor theta12 = Tensor({th12}).dType(NTdtypes::kComplexFloat).requiresGrad(false);
    Tensor deltaCP = Tensor({dcp}).dType(NTdtypes::kComplexFloat).requiresGrad(false);

    Tensor PMNS = buildPMNS(theta12, theta13, theta23, deltaCP);

    // set up the propagator
    Propagator vacuumProp(3, 100.0);
    vacuumProp.setPMNS(PMNS);
    vacuumProp.setMasses(masses);

    // seed the random number generator for the energies
    std::srand(randSeed);

    // linter gets angry about this as _ is never used :)))
    // NOLINTNEXTLINE
    for (auto _ : state)
    {
        // This code gets timed
        batchedOscProbs(vacuumProp, state.range(0), state.range(1));
    }
}

static void BM_constMatterOscillations(benchmark::State &state)
{

    // set up the inputs
    Tensor masses = Tensor({m1, m2, m3}, NTdtypes::kFloat).requiresGrad(false).addBatchDim();

    Tensor theta23 = Tensor({th23}).dType(NTdtypes::kComplexFloat).requiresGrad(false);
    Tensor theta13 = Tensor({th13}).dType(NTdtypes::kComplexFloat).requiresGrad(false);
    Tensor theta12 = Tensor({th12}).dType(NTdtypes::kComplexFloat).requiresGrad(false);
    Tensor deltaCP = Tensor({dcp}).dType(NTdtypes::kComplexFloat).requiresGrad(false);

    Tensor PMNS = buildPMNS(theta12, theta13, theta23, deltaCP);

    // set up the propagator
    Propagator matterProp(3, 100.0);
    std::shared_ptr<BaseMatterSolver> matterSolver = std::make_shared<ConstDensityMatterSolver>(3, 2.6);
    matterProp.setPMNS(PMNS);
    matterProp.setMasses(masses);
    matterProp.setMatterSolver(matterSolver);

    // seed the random number generator for the energies
    std::srand(randSeed);

    // linter gets angry about this as _ is never used :)))
    // NOLINTNEXTLINE
    for (auto _ : state)
    {
        // This code gets timed
        batchedOscProbs(matterProp, state.range(0), state.range(1));
    }
}

// Register the function as a benchmark
// NOLINTNEXTLINE
BENCHMARK(BM_vacuumOscillations)->Name("Vacuum Oscillations")->Args({1 << 10, 1 << 10});

// Register the function as a benchmark
// NOLINTNEXTLINE
BENCHMARK(BM_constMatterOscillations)->Name("Const Density Oscillations")->Args({1 << 10, 1 << 10});

// Run the benchmark
// NOLINTNEXTLINE
BENCHMARK_MAIN();
