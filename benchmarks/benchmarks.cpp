
#include <benchmark/benchmark.h> // NOLINT
#include <nuTens/propagator/DP-propagator.hpp>
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/constants.hpp>
#include <nuTens/propagator/pmns-matrix.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/propagator/units.hpp>
#include <nuTens/tensors/autograd.hpp>
#include <nuTens/tensors/tensor.hpp>

using namespace nuTens;

// the baseline to calculate oscillations at
constexpr float baseline = 295 * units::km;
// the electron density to use in calculations
constexpr float density = 2.6;
// uded for setting the scale and position of the energy distribution
constexpr float energyScale = 1 * units::GeV;
constexpr float energyOffset = 100 * units::eV;
// number of NR iterations to use for the DP propagator
constexpr int DPpropNRiterations = 5;

// The random seed to use for the RNG
// want this to be fixed for reproducibility
const int randSeed = 123;

/// get random double between 0.0 and 1.0
double randomDouble()
{
    return (double)rand() / (RAND_MAX + 1.);
}

/// get random double between 0.0 and 1.0
float randomFloat()
{
    return (float)rand() / ((float)RAND_MAX + 1.F);
}

static void batchedOscProbs(Propagator &prop, PMNSmatrix &matrix, Tensor *masses, long nBatches,
                            dtypes::deviceType device)
{
    for (int _ = 0; _ < nBatches; _++)
    {

        // set random values of the oscillation parameters
        masses->setValue({0, 0}, randomDouble());
        masses->setValue({0, 1}, randomDouble());
        masses->setValue({0, 2}, randomDouble());

        matrix.setTheta12(randomFloat())
            .setTheta13(randomFloat())
            .setTheta23(randomFloat())
            .setDeltaCP(randomFloat() * (float)constants::twoPi);

        prop.setMixingMatrix(matrix.build());
        prop.setMasses(masses->device(device));

        // calculate the osc probabilities
        // static_cast<void> to discard the return value that we're not supposed to discard :)
        static_cast<void>(prop.calculateProbs().sum());

        // move back to the cpu so we can set their values
        masses->device(dtypes::kCPU);
    }
}

static void propagatorBenchmark(benchmark::State &state, bool inMatter, dtypes::deviceType device = dtypes::kCPU)
{
    // make some random test energies
    Tensor energies =
        Tensor::rand({state.range(0)}).dType(dtypes::kComplexFloat).requiresGrad(false).device(device) * energyScale +
        energyOffset;

    // set up the inputs
    auto masses = AccessedTensor<float, 2, dtypes::kCPU>::zeros({1, 3});
    PMNSmatrix PMNS(device);

    // set up the propagator
    Propagator prop = Propagator(3, device).setBaseline(baseline);
    prop.setEnergies(energies);

    if (inMatter)
    {
        auto matterSolver = std::make_shared<ConstDensityMatterSolver>(3, device);
        matterSolver->setDensity(density);
        prop.setMatterSolver(matterSolver);
    }

    // seed the random number generator for the energies
    std::srand(randSeed);

    // linter gets angry about this as _ is never used :)))
    // NOLINTNEXTLINE
    for (auto _ : state)
    {
        // This code gets timed
        batchedOscProbs(prop, PMNS, &masses, state.range(1), device);
    }
}

static void DPpropagatorBenchmark(benchmark::State &state, dtypes::deviceType device)
{
    // make some random test energies
    Tensor energies =
        Tensor::rand({state.range(0)}).dType(dtypes::kComplexFloat).requiresGrad(false).device(device) * energyScale +
        energyOffset;

    std::unique_ptr<Tensor> dmsq21;
    std::unique_ptr<Tensor> dmsq31;
    std::unique_ptr<Tensor> sinSqTheta23;
    std::unique_ptr<Tensor> sinSqTheta13;
    std::unique_ptr<Tensor> sinSqTheta12;
    std::unique_ptr<Tensor> deltaCP;

    if (device == dtypes::kGPU)
    {
        dmsq21 =
            std::make_unique<Tensor>(Tensor::zeros({1}).dType(dtypes::kFloat).device(dtypes::kGPU).requiresGrad(false));
        dmsq31 =
            std::make_unique<Tensor>(Tensor::zeros({1}).dType(dtypes::kFloat).device(dtypes::kGPU).requiresGrad(false));
        sinSqTheta23 =
            std::make_unique<Tensor>(Tensor::zeros({1}).dType(dtypes::kFloat).device(dtypes::kGPU).requiresGrad(false));
        sinSqTheta13 =
            std::make_unique<Tensor>(Tensor::zeros({1}).dType(dtypes::kFloat).device(dtypes::kGPU).requiresGrad(false));
        sinSqTheta12 =
            std::make_unique<Tensor>(Tensor::zeros({1}).dType(dtypes::kFloat).device(dtypes::kGPU).requiresGrad(false));
        deltaCP = std::make_unique<Tensor>(
            Tensor::zeros({1}).dType(dtypes::kComplexFloat).device(dtypes::kGPU).requiresGrad(false));
    }
    else if (device == dtypes::kCPU)
    {
        dmsq21 = std::make_unique<Tensor>(AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false));
        dmsq31 = std::make_unique<Tensor>(AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false));
        sinSqTheta23 = std::make_unique<Tensor>(AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false));
        sinSqTheta13 = std::make_unique<Tensor>(AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false));
        sinSqTheta12 = std::make_unique<Tensor>(AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, false));
        deltaCP = std::make_unique<Tensor>(Tensor::zeros({1}).dType(dtypes::kComplexFloat).requiresGrad(false));
    }

    // set up the propagator
    DPpropagator dpProp = DPpropagator(/*NRiterations=*/DPpropNRiterations, device)
                              .setBaseline(baseline)
                              .setAntiNeutrino(false)
                              .setDensity(density);

    dpProp.setEnergies(energies);

    dpProp.setSinSquaredThetas(true);
    dpProp.setTheta12(*sinSqTheta12)
        .setTheta23(*sinSqTheta23)
        .setTheta13(*sinSqTheta13)
        .setDeltaCP(*deltaCP)
        .setDmsq21(*dmsq21)
        .setDmsq31(*dmsq31);

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
            dmsq21->setValue({0}, randomDouble());
            dmsq31->setValue({0}, randomDouble());

            sinSqTheta23->setValue({0}, randomDouble());
            sinSqTheta13->setValue({0}, randomDouble());
            sinSqTheta12->setValue({0}, randomDouble());

            deltaCP->setValue({0}, Tensor::rand({1}) * constants::twoPi);

            // calculate the osc probabilities
            // static_cast<void> to discard the return value that we're not supposed to discard :)
            static_cast<void>(dpProp.calculateProbs().sum());
        }
    }
}

static void BM_vacuumOscillations(benchmark::State &state)
{

    NT_PROFILE_BEGINSESSION("Benchmark-vacuum-oscillations");
    NT_PROFILE();

    propagatorBenchmark(state, /*inMatter=*/false);

    NT_PROFILE_ENDSESSION();
}

static void BM_vacuumOscillationsGPU(benchmark::State &state)
{

    if (!Tensor::gpuAvailable())
    {
        throw std::runtime_error("No GPU available");
    }

    NT_PROFILE_BEGINSESSION("Benchmark-vacuum-oscillations");
    NT_PROFILE();

    propagatorBenchmark(state, /*inMatter=*/false, /*device=*/dtypes::kGPU);

    NT_PROFILE_ENDSESSION();
}

static void BM_constMatterOscillations(benchmark::State &state)
{

    NT_PROFILE_BEGINSESSION("Benchmark-const-density-oscillations");

    NT_PROFILE();

    propagatorBenchmark(state, /*inMatter=*/true);

    NT_PROFILE_ENDSESSION();
}

static void BM_constMatterOscillationsGPU(benchmark::State &state)
{

    if (!Tensor::gpuAvailable())
    {
        throw std::runtime_error("No GPU available");
    }

    NT_PROFILE_BEGINSESSION("Benchmark-const-density-oscillations-GPU");

    NT_PROFILE();

    propagatorBenchmark(state, /*inMatter=*/true, /*device=*/dtypes::kGPU);

    NT_PROFILE_ENDSESSION();
}

static void BM_vacuumOscillationsNoGrad(benchmark::State &state)
{

    NT_PROFILE_BEGINSESSION("Benchmark-vacuum-oscillations-noGrad");
    NT_PROFILE();

    // disable gradient calculations
    auto noGradGuard = autograd::NoGrad();

    propagatorBenchmark(state, /*inMatter=*/false);

    NT_PROFILE_ENDSESSION();
}

static void BM_vacuumOscillationsNoGradGPU(benchmark::State &state)
{

    NT_PROFILE_BEGINSESSION("Benchmark-vacuum-oscillations-noGrad");
    NT_PROFILE();

    // disable gradient calculations
    auto noGradGuard = autograd::NoGrad();

    propagatorBenchmark(state, /*inMatter=*/false, dtypes::kGPU);

    NT_PROFILE_ENDSESSION();
}

static void BM_constMatterOscillationsNoGrad(benchmark::State &state)
{

    NT_PROFILE_BEGINSESSION("Benchmark-const-density-oscillations-noGrad");

    NT_PROFILE();

    // disable gradient calculations
    auto noGradGuard = autograd::NoGrad();

    propagatorBenchmark(state, /*inMatter=*/true);

    NT_PROFILE_ENDSESSION();
}

static void BM_constMatterOscillationsNoGradGPU(benchmark::State &state)
{

    if (!Tensor::gpuAvailable())
    {
        throw std::runtime_error("No GPU available");
    }

    NT_PROFILE_BEGINSESSION("Benchmark-const-density-oscillations-noGrad-GPU");

    NT_PROFILE();

    // disable gradient calculations
    auto noGradGuard = autograd::NoGrad();

    propagatorBenchmark(state, /*inMatter=*/true, /*device=*/dtypes::kGPU);

    NT_PROFILE_ENDSESSION();
}

static void BM_DPpropOscillations(benchmark::State &state)
{

    NT_PROFILE_BEGINSESSION("Benchmark-DP-propagator");

    NT_PROFILE();

    DPpropagatorBenchmark(state, dtypes::kCPU);

    NT_PROFILE_ENDSESSION();
}

static void BM_DPpropOscillationsNoGrad(benchmark::State &state)
{

    NT_PROFILE_BEGINSESSION("Benchmark-DP-propagator-noGrad");

    NT_PROFILE();

    // disable gradient calculations
    auto noGradGuard = autograd::NoGrad();

    DPpropagatorBenchmark(state, dtypes::kCPU);

    NT_PROFILE_ENDSESSION();
}

static void BM_DPpropOscillationsGPU(benchmark::State &state)
{

    if (!Tensor::gpuAvailable())
    {
        throw std::runtime_error("No GPU available");
    }

    NT_PROFILE_BEGINSESSION("Benchmark-DP-propagator-gpu");

    NT_PROFILE();

    DPpropagatorBenchmark(state, dtypes::kGPU);

    NT_PROFILE_ENDSESSION();
}

static void BM_DPpropOscillationsNoGradGPU(benchmark::State &state)
{

    if (!Tensor::gpuAvailable())
    {
        throw std::runtime_error("No GPU available");
    }

    NT_PROFILE_BEGINSESSION("Benchmark-DP-propagator-noGrad-gpu");

    NT_PROFILE();

    // disable gradient calculations
    auto noGradGuard = autograd::NoGrad();

    DPpropagatorBenchmark(state, dtypes::kGPU);

    NT_PROFILE_ENDSESSION();
}

constexpr std::initializer_list<long int> range = {1 << 10, 1 << 10};

// Register the function as a benchmark
// NOLINTNEXTLINE
BENCHMARK(BM_vacuumOscillations)->Name("Vacuum Oscillations")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_vacuumOscillationsNoGrad)->Name("Vacuum Oscillations noGrad")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_constMatterOscillations)->Name("Const Density Oscillations")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_constMatterOscillationsNoGrad)->Name("Const Density Oscillations noGrad")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_DPpropOscillations)->Name("DP Propagator Const Density Oscillations")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_DPpropOscillationsNoGrad)->Name("DP Propagator Const Density Oscillations noGrad")->Args(range);

// only compile following benchmarks if GPU benchmarking was explicitly enabled
#if BENCHMARK_GPU

// NOLINTNEXTLINE
BENCHMARK(BM_vacuumOscillationsGPU)->Name("Vacuum Oscillations GPU")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_vacuumOscillationsNoGradGPU)->Name("Vacuum Oscillations noGrad GPU")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_constMatterOscillationsGPU)->Name("Const Density Oscillations GPU")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_constMatterOscillationsNoGradGPU)->Name("Const Density Oscillations noGrad GPU")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_DPpropOscillationsGPU)->Name("DP Propagator Const Density Oscillations GPU")->Args(range);

// NOLINTNEXTLINE
BENCHMARK(BM_DPpropOscillationsNoGradGPU)->Name("DP Propagator Const Density Oscillations noGrad GPU")->Args(range);

#endif

// Run the benchmark
// NOLINTNEXTLINE
BENCHMARK_MAIN();
