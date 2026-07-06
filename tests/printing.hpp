#pragma once

/// @file printing.hpp
/// provides utility functions for nicely printing nuTens classes
/// within the gtest framework

#include <nuTens/tensors/dtypes.hpp>

namespace nuTens::dtypes
{
void PrintTo(const deviceType &device, std::ostream *oStream)
{
    if (device == kCPU)
    {
        *oStream << "CPU";
    }
    else if (device == kGPU)
    {
        *oStream << "GPU";
    }
}

void PrintTo(const scalarType &dtype, std::ostream *oStream)
{
    if (dtype == kFloat)
    {
        *oStream << "Float";
    }
    else if (dtype == kDouble)
    {
        *oStream << "Double";
    }
    else if (dtype == kComplexFloat)
    {
        *oStream << "Complex Float";
    }
    else if (dtype == kComplexDouble)
    {
        *oStream << "Complex Double";
    }
}
} // namespace nuTens::dtypes