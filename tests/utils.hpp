#pragma once

/// @file utils.hpp
/// define some helpful utilities for testing

#include <gtest/gtest.h> // NOLINT
// alias the gtest "testing" namespace
namespace gtest = ::testing;

namespace nuTens::testing
{

#define SKIP_GPU(device)                                                                                               \
    if (!Tensor::gpuAvailable() && device == dtypes::kGPU)                                                             \
    {                                                                                                                  \
        GTEST_SKIP() << "No GPU available. SKIPPING!";                                                                 \
    }

} // end namespace nuTens::testing