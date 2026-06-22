#pragma once

#include <gtest/gtest.h> // NOLINT
// alias the gtest "testing" namespace
namespace gtest = ::testing;

#include <iostream>
#include <nuTens/propagator/base-matter-solver.hpp>
#include <nuTens/propagator/const-density-solver.hpp>
#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>

using namespace nuTens;

/// BaseMatterSolver dummy implementation so we can test the base
/// class methods
class DummyMatterSolver : public BaseMatterSolver
{
  public:
    DummyMatterSolver(int nGenerations, bool antiNeutrino) : BaseMatterSolver(nGenerations, antiNeutrino){};

    void calculateEigenvalues(Tensor &eigenvectors, Tensor &eigenvalues) override{};
};