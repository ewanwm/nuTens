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

class NoGrad
{
    /*!
     * @brief Disable autograd calculations within some scope, improving performance where you are not interested in
     * calculating gradients Instantiate at the start of the scope where you want to disable gradient calculations like
     * so \code{.cpp} #include <nuTens/tensors/tensor.hpp>
     * ...
     *   {
     *       auto noGrad = nuTens::NoGrad();
     *       ...
     *       < speedy non differentiated code >
     *       ...
     *   }
     * \endcode
     */

  public:
    NoGrad()
    {
#if USE_PYTORCH
        guard = std::make_unique<c10::InferenceMode>();
#endif
    };

  private:
#if USE_PYTORCH
    std::unique_ptr<c10::InferenceMode> guard;
#endif
};

} // namespace nuTens::autograd