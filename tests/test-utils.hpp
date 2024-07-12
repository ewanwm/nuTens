#include <math.h>

// Some helpful utility functions for testing

namespace Testing{

    // Get absolute relative difference between two floats:
    //   | (f1 - f2) / f1 |
    float relativeDiff(float f1, float f2){
        return std::abs((f1 - f2) / f1);
    }
    
}