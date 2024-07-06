## Tensors

We define a fairly barebones abstracted interface for manipulating tensors so that we can support multible libraries (though currently only PyTorch is supported)

[tensor.hpp](tensor.hpp) defines the interface. The implementations for each tensor library are then defined in \<library\>-tensor.cpp. Which implementation gets compiled is decided at compile time depending on which library is found by CMake. 

[dtypes.hpp](dtpes.hpp) defines various types used in this library, which are then translated to library specific types in the implementation cpp files using maps.