#include <nuTens/tensors/autograd.hpp>

namespace nuTens::autograd
{

nuTens::Tensor grad(const nuTens::Tensor &final, const nuTens::Tensor &leaf)
{
    NT_PROFILE();

    return nuTens::Tensor::fromTorchTensor(torch::autograd::grad(
        {final.getTensor()}, {leaf.getTensor()}, /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/true)[0]);
}

} // namespace nuTens::autograd