from __future__ import annotations
import nuTens._pyNuTens.tensor
__all__: list[str] = ['NoGrad', 'grad']
class NoGrad:
    def __init__(self) -> None:
        ...
def grad(value: nuTens._pyNuTens.tensor.Tensor, leaf: nuTens._pyNuTens.tensor.Tensor) -> nuTens._pyNuTens.tensor.Tensor:
    """
    Get the gradient of a value with respect to some leaf tensor
    """
