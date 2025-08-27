#pragma once

#include <nuTens/tensors/tensor.hpp>

namespace nuTens
{

class BaseMixingMatrix
{
  public:
    /// @brief Should construct and return the mixing matrix
    virtual Tensor &build() = 0;

    /// destructor
    virtual ~BaseMixingMatrix() = default;
};

}; // namespace nuTens
