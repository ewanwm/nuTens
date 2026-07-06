#pragma once

/// @file printing.hpp
/// provides utility functions for nicely printing nuTens classes
/// within the gtest framework

namespace nuTens::dtypes
{
void PrintTo(const deviceType &device, std::ostream *os)
{
    if (device == kCPU)
    {
        *os << "CPU";
    }
    else if (device == kGPU)
    {
        *os << "GPU";
    }
}

void PrintTo(const scalarType &dtype, std::ostream *os)
{
    if (dtype == kFloat)
    {
        *os << "Float";
    }
    else if (dtype == kDouble)
    {
        *os << "Double";
    }
    else if (dtype == kComplexFloat)
    {
        *os << "Complex Float";
    }
    else if (dtype == kComplexDouble)
    {
        *os << "Complex Double";
    }
}
} // namespace nuTens::dtypes