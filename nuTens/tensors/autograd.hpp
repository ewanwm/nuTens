#pragma once

#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/instrumentation.hpp>
#include <nuTens/utils/logging.hpp>

#if USE_PYTORCH
#include <torch/torch.h>
#endif

namespace nuTens::autograd
{

/// @brief compute the gradient of "final" with respect to "leaf"
nuTens::Tensor grad(const nuTens::Tensor &final, const nuTens::Tensor &leaf);

} // namespace nuTens::autograd