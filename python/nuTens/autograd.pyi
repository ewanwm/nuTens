from __future__ import annotations
import nuTens.pyNuTens.tensor
__all__: list[str] = ['NoGrad', 'grad']
class NoGrad:
    def __init__(self) -> None:
        ...
def grad(value: nuTens.pyNuTens.tensor.Tensor, leaf: nuTens.pyNuTens.tensor.Tensor) -> nuTens.pyNuTens.tensor.Tensor:
    """
    Get the gradient of a value with respect to some leaf tensor
    """
