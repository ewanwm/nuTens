
#include <benchmark/benchmark.h>
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/propagator/propagator.hpp>
#include <nuTens/tensors/tensor.hpp>

// The random seed to use for the RNG
// want this to be fixed for reproducibility
const int randSeed = 123;

const std::complex<float> imagUnit(0.0, 1.0);

/// get random double between 0.0 and 1.0
double randomDouble()
{
    return (double)rand() / (RAND_MAX + 1.);
}

class PMNSmatrix
{
  public:
    PMNSmatrix()
    {
        // set up the three matrices to build the PMNS matrix
        _m1 = Tensor::zeros({1, 3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);
        _m2 = Tensor::zeros({1, 3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);
        _m3 = Tensor::zeros({1, 3, 3}, NTdtypes::kComplexFloat).requiresGrad(false);
    }

    void build(const Tensor &theta12, const Tensor &theta13, const Tensor &theta23, const Tensor &deltaCP)
    {
        _m1.requiresGrad(false);
        _m2.requiresGrad(false);
        _m3.requiresGrad(false);

        _m1.setValue({0, 0, 0}, 1.0);
        _m1.setValue({0, 1, 1}, Tensor::cos(theta23));
        _m1.setValue({0, 1, 2}, Tensor::sin(theta23));
        _m1.setValue({0, 2, 1}, -Tensor::sin(theta23));
        _m1.setValue({0, 2, 2}, Tensor::cos(theta23));
        _m1.requiresGrad(true);

        _m2.setValue({0, 1, 1}, 1.0);
        _m2.setValue({0, 0, 0}, Tensor::cos(theta13));
        _m2.setValue({0, 0, 2}, Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(deltaCP, -imagUnit))));
        _m2.setValue({0, 2, 0}, -Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(deltaCP, imagUnit))));
        _m2.setValue({0, 2, 2}, Tensor::cos(theta13));
        _m2.requiresGrad(true);

        _m3.setValue({0, 2, 2}, 1.0);
        _m3.setValue({0, 0, 0}, Tensor::cos(theta12));
        _m3.setValue({0, 0, 1}, Tensor::sin(theta12));
        _m3.setValue({0, 1, 0}, -Tensor::sin(theta12));
        _m3.setValue({0, 1, 1}, Tensor::cos(theta12));
        _m3.requiresGrad(true);

        // Build PMNS
        matrix = Tensor::matmul(_m1, Tensor::matmul(_m2, _m3));
        matrix.requiresGrad(true);
    }

    Tensor matrix;

  private:
    Tensor _m1;
    Tensor _m2;
    Tensor _m3;
};

static void batchedOscProbs(
    Propagator &prop, 
    PMNSmatrix &matrix, 
    AccessedTensor<float, 1, NTdtypes::kCPU> &theta23, 
    AccessedTensor<float, 1, NTdtypes::kCPU> &theta13, 
    AccessedTensor<float, 1, NTdtypes::kCPU> &theta12,
    Tensor &deltaCP, 
    AccessedTensor<float, 2, NTdtypes::kCPU> &masses, 
    long nBatches)
{
    for (int _ = 0; _ < nBatches; _++)
    {

        // set random values of the oscillation parameters
        masses.setValue(randomDouble(), 0, 0);
        masses.setValue(randomDouble(), 0, 1);
        masses.setValue(randomDouble(), 0, 2);

        theta23.setValue(randomDouble(), 0);
        theta13.setValue(randomDouble(), 0);
        theta12.setValue(randomDouble(), 0);

        deltaCP.setValue({0}, Tensor::scale(Tensor::rand({1}), 2.0 * 3.1415));

        // calculate new values of the PMNS matrix
        matrix.build(theta12, theta13, theta23, deltaCP);

        prop.setPMNS(matrix.matrix);
        prop.setMasses(masses);

        // calculate the osc probabilities
        // static_cast<void> to discard the return value that we're not supposed to discard :)
        static_cast<void>(prop.calculateProbs().sum());
    }
}

static void BM_vacuumOscillations(benchmark::State &state)
{
    // make some random test energies
    Tensor energies =
        Tensor::scale(Tensor::rand({state.range(0), 1}).dType(NTdtypes::kFloat).requiresGrad(false), 10000.0).hasBatchDim(true) +
        Tensor({100.0});

    energies = energies.hasBatchDim(true);

    // set up the inputs
    auto masses = AccessedTensor<float, 2, NTdtypes::kCPU>::zeros({1, 3});

    auto theta23 = AccessedTensor<float, 1, NTdtypes::kCPU>::zeros({1}, false);
    auto theta13 = AccessedTensor<float, 1, NTdtypes::kCPU>::zeros({1}, false);
    auto theta12 = AccessedTensor<float, 1, NTdtypes::kCPU>::zeros({1}, false);
    auto deltaCP = Tensor::zeros({1}).dType(NTdtypes::kComplexFloat).requiresGrad(false);

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
        batchedOscProbs(vacuumProp, PMNS, theta23, theta13, theta12, deltaCP, masses, state.range(1));
    }
}

static void BM_constMatterOscillations(benchmark::State &state)
{
    // make some random test energies
    Tensor energies =
        Tensor::scale(Tensor::rand({state.range(0), 1}).dType(NTdtypes::kFloat).requiresGrad(false), 10000.0) +
        Tensor({100.0});

    energies = energies.hasBatchDim(true);

    // set up the inputs
    auto masses = AccessedTensor<float, 2, NTdtypes::kCPU>::zeros({1, 3});

    auto theta23 = AccessedTensor<float, 1, NTdtypes::kCPU>::zeros({1}, false);
    auto theta13 = AccessedTensor<float, 1, NTdtypes::kCPU>::zeros({1}, false);
    auto theta12 = AccessedTensor<float, 1, NTdtypes::kCPU>::zeros({1}, false);
    auto deltaCP = Tensor::zeros({1}).dType(NTdtypes::kComplexFloat).requiresGrad(false);

    PMNSmatrix PMNS;
    PMNS.build(theta12, theta13, theta23, deltaCP);

    // set up the propagator
    Propagator matterProp(3, 295000.0);
    matterProp.setPMNS(PMNS.matrix);
    matterProp.setMasses(masses);

    std::shared_ptr<BaseMatterSolver> matterSolver = std::make_shared<ConstDensityMatterSolver>(3, 2.6);

    matterProp.setMatterSolver(matterSolver);

    matterProp.setEnergies(energies);

    // seed the random number generator for the energies
    std::srand(randSeed);

    // linter gets angry about this as _ is never used :)))
    // NOLINTNEXTLINE
    for (auto _ : state)
    {
        // This code gets timed
        batchedOscProbs(matterProp, PMNS, theta23, theta13, theta12, deltaCP, masses, state.range(1));
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
