from __future__ import annotations
import collections.abc
import nuTens._pyNuTens.dtype
import nuTens.dtype
import torch
import typing
__all__: list[str] = ['Tensor', 'add', 'cos', 'cumsum', 'div', 'exp', 'gpu_available', 'log', 'matmul', 'mul', 'outer', 'pow', 'scale', 'sin', 'sum', 'transpose']
class Tensor:
    """
    Tensor defines a basic interface for creating and manipulating tensors.To create tensors you should use the static constructor methods.
    Alternatively you can chain together multiple property setters.
    
    For example
    
    .. code-block::
    
        from nuTens.tensor import Tensor, dtype
        tensor = Tensor.ones([3,3], dtype.scalar_type.float, dtype.device_type.cpu)
    
    will get you a 3x3 tensor of floats that lives on the CPU.
    
    This is equivalent to
    
    .. code-block::
        tensor = Tensor.ones([3,3]).dtype(dtype.scalar_type.float).device(dtype.device_type.cpu);
    
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def diag(diagonal: Tensor) -> Tensor:
        """
        Create a tensor with specified values along the diagonal
        """
    @staticmethod
    def eye(n: typing.SupportsInt | typing.SupportsIndex, dtype: nuTens._pyNuTens.dtype.scalar_type = nuTens.dtype.scalar_type.float, device: nuTens._pyNuTens.dtype.device_type = nuTens.dtype.device_type.cpu, requires_grad: bool = True) -> Tensor:
        """
        Create a tensor initialised with an identity matrix
        """
    @staticmethod
    def from_torch_tensor(arg0: torch.Tensor) -> Tensor:
        """
        construct a nuTens Tensor from a pytorch tensor
        """
    @staticmethod
    def ones(shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], dtype: nuTens._pyNuTens.dtype.scalar_type = nuTens.dtype.scalar_type.float, device: nuTens._pyNuTens.dtype.device_type = nuTens.dtype.device_type.cpu, requires_grad: bool = True) -> Tensor:
        """
        Create a tensor initialised with ones
        """
    @staticmethod
    def rand(shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], dtype: nuTens._pyNuTens.dtype.scalar_type = nuTens.dtype.scalar_type.float, device: nuTens._pyNuTens.dtype.device_type = nuTens.dtype.device_type.cpu, requires_grad: bool = True) -> Tensor:
        """
        Create a tensor initialised with random values
        """
    @staticmethod
    def zeros(shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], dtype: nuTens._pyNuTens.dtype.scalar_type = nuTens.dtype.scalar_type.float, device: nuTens._pyNuTens.dtype.device_type = nuTens.dtype.device_type.cpu, requires_grad: bool = True) -> Tensor:
        """
        Create a tensor initialised with zeros
        """
    @typing.overload
    def __add__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __add__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
        ...
    def __eq__(self, arg0: Tensor) -> bool:
        ...
    def __getstate__(self) -> tuple[torch.Tensor]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], dtype: nuTens._pyNuTens.dtype.scalar_type = nuTens.dtype.scalar_type.float, device: nuTens._pyNuTens.dtype.device_type = nuTens.dtype.device_type.cpu, requires_grad: bool = True) -> None:
        ...
    @typing.overload
    def __init__(self, array: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], requires_grad: bool = True) -> None:
        """
        Construct a tensor from a numpy array
        """
    @typing.overload
    def __init__(self, array_like: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64], requires_grad: bool = True) -> None:
        """
        Construct a tensor from an "array like" object
        """
    def __mul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
        ...
    def __neg__(self) -> Tensor:
        ...
    def __radd__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
        ...
    def __repr__(self) -> str:
        ...
    def __rmul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __truediv__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
        ...
    def abs(self) -> Tensor:
        """
        Get element-wise magnitudes of a complex tensor
        """
    def add_batch_dim(self) -> Tensor:
        """
        Add a batch dimension to the start of this tensor if it doesn't have one already
        """
    def angle(self) -> Tensor:
        """
        Get element-wise phases of a complex tensor
        """
    def backward(self) -> None:
        """
        Do the backward propagation from this tensor
        """
    def conj(self) -> Tensor:
        """
        Get complex conjugate of a complex tensor
        """
    def device(self, new_device: nuTens._pyNuTens.dtype.device_type) -> Tensor:
        """
        Set the device that the tensor lives on
        """
    def dtype(self, new_dtype: nuTens._pyNuTens.dtype.scalar_type) -> Tensor:
        """
        Set the data type of the tensor
        """
    def get_device(self) -> nuTens._pyNuTens.dtype.device_type:
        """
        Get the device that this tensor lives on
        """
    def get_dtype(self) -> nuTens._pyNuTens.dtype.scalar_type:
        """
        Get the type of the data contained within this tensor
        """
    def get_requires_grad(self) -> bool:
        """
        Get whether or not this tensor will collect gradients
        """
    def get_shape(self) -> list[int]:
        """
        Get the shape of this tensor
        """
    def get_value(self, indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> int | float | float | complex | complex:
        """
        Get the data stored at a particular index of the tensor
        """
    def get_values(self, indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex | str]) -> Tensor:
        """
        Get the subset of values in this tensor at a specified location
        """
    def grad(self) -> Tensor:
        """
        Get the accumulated gradient stored in this tensor after calling backward()
        """
    @typing.overload
    def has_batch_dim(self) -> bool:
        """
        Check Whether or not the first dimension should be interpreted as a batch dim for this tensor
        """
    @typing.overload
    def has_batch_dim(self, new_value: bool) -> Tensor:
        """
        Set Whether or not the first dimension should be interpreted as a batch dim for this tensor
        """
    def imag(self) -> Tensor:
        """
        Get imaginary part of a complex tensor
        """
    def is_initialised(self) -> bool:
        """
        Check if this tensor has been initialised yet
        """
    def numpy(self) -> numpy.ndarray:
        """
        Get a numpy array with the contents of the tensor
        """
    def real(self) -> Tensor:
        """
        Get real part of a complex tensor
        """
    def requires_grad(self, new_value: bool) -> Tensor:
        """
        Set Whether or not this tensor requires gradient to be calculated
        """
    @typing.overload
    def set_value(self, indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex | str], value: Tensor) -> None:
        """
        Set a value at a specific index of this tensor
        """
    @typing.overload
    def set_value(self, indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a value at a specific index of this tensor
        """
    @typing.overload
    def set_value(self, indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a value at a specific index of this tensor
        """
    @typing.overload
    def set_value(self, indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], value: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a value at a specific index of this tensor
        """
    @typing.overload
    def set_value(self, indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], value: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a value at a specific index of this tensor
        """
    def to_string(self) -> str:
        """
        get a summary of this tensor as a string
        """
    def torch_tensor(self) -> torch.Tensor:
        """
        Get the pytorch tensor that lives inside this tensor. Only available if using the pytorch backend...
        """
    def unsqueeze(self, dim: typing.SupportsInt | typing.SupportsIndex) -> Tensor:
        """
        add an extra dimension to this tensor at the specified location
        """
    def zero_grad(self) -> None:
        """
        Zero out the accumulated gradient for this tensor
        """
def add(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Element-wise addition
    """
def cos(tensor_1: Tensor) -> Tensor:
    """
    Element-wise trigonometric cosine function
    """
def cumsum(tensor_1: Tensor, dimensions: typing.SupportsInt | typing.SupportsIndex) -> Tensor:
    """
    Get the cumulative sum over particular dimensions
    """
def div(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Element-wise division
    """
def exp(tensor_1: Tensor) -> Tensor:
    """
    Take element-wise exponential of a tensor
    """
def gpu_available() -> bool:
    """
    Returns true if there is an available GPU, False if not
    """
def log(tensor_1: Tensor) -> Tensor:
    """
    Take element-wise natural log of a tensor
    """
def matmul(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Matrix multiplication
    """
def mul(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Element-wise multiplication
    """
def outer(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Tensor outer product
    """
@typing.overload
def pow(tensor_1: Tensor, power: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
    """
    Raise to scalar power
    """
@typing.overload
def pow(tensor_1: Tensor, power: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
    """
    Raise to scalar power
    """
@typing.overload
def scale(tensor_1: Tensor, scalar: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
    """
    Scalar multiplication
    """
@typing.overload
def scale(tensor_1: Tensor, scalar: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
    """
    Scalar multiplication
    """
def sin(tensor_1: Tensor) -> Tensor:
    """
    Element-wise trigonometric sine function
    """
@typing.overload
def sum(tensor_1: Tensor) -> Tensor:
    """
    Get the sum of all values in a tensor
    """
@typing.overload
def sum(tensor_1: Tensor, dimensions: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> Tensor:
    """
    Get the sum over particular dimensions
    """
def transpose(tensor_1: Tensor, index_1: typing.SupportsInt | typing.SupportsIndex, index_2: typing.SupportsInt | typing.SupportsIndex) -> Tensor:
    """
    Get the matrix transpose
    """
