#pragma once

#include <cmath>

#include <iostream>

// Some helpful utility functions for testing

namespace Testing
{

// Get absolute relative difference between two floats:
//   | (f1 - f2) / f1 |
float relativeDiff(float f1, float f2)
{
    return std::abs((f1 - f2) / f1);
}

} // namespace Testing

// ###########################
// #### Some handy macros ####
// ###########################

// use when we want to check if a value is equal to some expectation
// threshold is fractional difference that is considdered "too different"
// will print out some useful information then fail the test
// NOLINTNEXTLINE: Wants me to make this a constexpr... but i prefer it like this so...
#define TEST_EXPECTED(value, expectation, varName, threshold)                                                          \
    {                                                                                                                  \
        if (Testing::relativeDiff((value), (expectation)) > (threshold))                                               \
        {                                                                                                              \
            std::cerr << "bad " << (varName) << std::endl;                                                             \
            std::cerr << "Got: " << (value);                                                                           \
            std::cerr << "; Expected: " << (expectation);                                                              \
            std::cerr << std::endl;                                                                                    \
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;                                                     \
            return 1;                                                                                                  \
        }                                                                                                              \
    }
