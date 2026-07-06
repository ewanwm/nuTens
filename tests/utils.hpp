#pragma once

/// @file utils.hpp
/// define some helpful utilities for testing

#include <gtest/gtest.h> // NOLINT
// alias the gtest "testing" namespace
namespace gtest = ::testing;

namespace nuTens::testing
{

// SKIP_GPU() has to be a macro as GTEST_SKIP() cannot be called from inside a function and needs to be called directly
// within the test body!!!

/// @brief Skips the current test (using GTEST_SKIP()) if the passed device is GPU and no GPU device can be found
/// @param[in] device The device to check
// NOLINTNEXTLINE
#define SKIP_GPU(device)                                                                                               \
    if (!Tensor::gpuAvailable() && (device) == dtypes::kGPU)                                                           \
    {                                                                                                                  \
        GTEST_SKIP() << "No GPU available. SKIPPING!";                                                                 \
    }

} // end namespace nuTens::testing