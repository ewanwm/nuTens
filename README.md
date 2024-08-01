<a name="nutens"></a>
# <img src="doc/nuTens-logo.png" alt="nuTens" class="right" align="top" width="400"/>

nuTens is a software library which uses [tensors](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) to efficiently calculate neutrino oscillation probabilities. 

[![CI badge](https://github.com/ewanwm/nuTens/actions/workflows/CI-build-and-test.yml/badge.svg)](https://github.com/ewanwm/nuTens/actions/workflows/CI-build-and-test.yml)
[![Code - Doxygen](https://img.shields.io/badge/Code-Doxygen-2ea44f)](https://ewanwm.github.io/nuTens/index.html)
[![test - coverage](https://codecov.io/github/ewanwm/nuTens/graph/badge.svg?token=PJ8C8CX37O)](https://codecov.io/github/ewanwm/nuTens)
[![cpp - linter](https://github.com/ewanwm/nuTens/actions/workflows/cpp-linter.yaml/badge.svg)](https://github.com/ewanwm/nuTens/actions/workflows/cpp-linter.yaml)


## Installation
### Requirements

- CMake - Should work with most modern versions. If you wish to use precompiled headers to speed up build times you will need CMake > 3.16.
- Compiler with support for c++17 standard - Tested with gcc
- [PyTorch](https://pytorch.org/) - The recommended way to install is using PyTorch_requirements.txt:
```
  pip install -r PyTorch_requirements.txt
```
(or see [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for instructions on how to build yourself)

### Installation
Assuming PyTorch was built using pip, [nuTens](#nutens) can be built using
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make <-j Njobs>
```

(installation with a non-pip install of PyTorch have not been tested but should be possible)

### Verifying Installation
Once [nuTens](#nutens) has been built, you can verify your installation by running
```
make test
```


## Feature Wishlist
- [x] Support PyTorch in tensor library
- [x] Vacuum oscillation calculations
- [x] Constant matter density propagation
- [x] Basic test suite
- [x] Basic CI
- [x] Doxygen documentation with automatic deployment
- [x] Add test coverage checks into CI
- [x] Integrate linting ( [cpp-linter](https://github.com/cpp-linter)? )
- [x] Add instrumentation library for benchmarking and profiling
- [ ] Add suite of benchmarking tests
- [ ] Integrate benchmarks into CI ( maybe use [hyperfine](https://github.com/sharkdp/hyperfine) and [bencher](https://bencher.dev/) for this? )
- [ ] Add proper unit tests
- [ ] Expand CI to include more platforms
- [ ] Add support for modules (see [PyTorch doc](https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html))
- [ ] Propagation in variable matter density
- [ ] Add support for Tensorflow backend
- [ ] Add python interface 

