## Python Interface

nuTens provides a python interface that can be installed manually or via [pyPi](https://pypi.org/). [binding.cpp](binding.cpp) defines the python bindings of many of the nuTens c++ objects, which are then compiled into a python module using [pybind11](https://github.com/pybind/pybind11) (see [CMakeLists.txt](CMakeLists.txt)).

The [nuTens](nuTens) folder defines the python module and any pure python extension code for nuTens should go in there.