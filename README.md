![Logo](https://github.com/ewanwm/nuTens/raw/main/nuTens-logo.png)
<a name="nutens"></a>

<div align="center">

  <a href="">[![DOI](https://zenodo.org/badge/824239795.svg)](https://doi.org/10.5281/zenodo.15873397)</a>
  <a href="">[![GitHub Release](https://img.shields.io/github/v/release/ewanwm/nuTens?color=blue)](https://github.com/ewanwm/nuTens/releases)</a>
  <a href="">[![PyPI - Version](https://img.shields.io/pypi/v/nuTens?color=blue)](https://pypi.org/project/nuTens/)</a>
  <a href="">[![GitHub License](https://img.shields.io/github/license/ewanwm/nuTens?color=green)](https://github.com/ewanwm/nuTens/blob/main/LICENSE)</a>

  <a href="">[![CI badge](https://github.com/ewanwm/nuTens/actions/workflows/CI-cpp.yml/badge.svg)](https://github.com/ewanwm/nuTens/actions/workflows/CI-cpp.yml)</a>
  <a href="">[![pip](https://github.com/ewanwm/nuTens/actions/workflows/CI-Python.yaml/badge.svg)](https://github.com/ewanwm/nuTens/actions/workflows/CI-Python.yaml)</a>
  <a href="">[![test - coverage](https://codecov.io/github/ewanwm/nuTens/graph/badge.svg?token=PJ8C8CX37O)](https://codecov.io/github/ewanwm/nuTens)</a>
  <a href="">[![cpp - linter](https://github.com/ewanwm/nuTens/actions/workflows/Lint-cpp.yaml/badge.svg)](https://github.com/ewanwm/nuTens/actions/workflows/Lint-cpp.yaml)</a>

</div>

nuTens is an engine for calculating neutrino oscillation proabilities using [tensors](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) which allow it to be fast, flexible, and fully differentiable. 

See the [full documentation](https://ewanwm.github.io/nuTens/) for more details.

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

### Python

nuTens provides a python interface for it's high level functionality. The Pypi release of nuTens can be found [here](https://pypi.org/project/nuTens/) and can be installed using 
```
pip install nuTens
```

#### Manual Installation 

The python interface can be installed manually after cloning the repository using pip by running
```
pip install .
```
in the root directory of nuTens

Additionally, the nuTens python module can be installed by specifying the CMake option
```
cmake -DNT_ENABLE_PYTHON=ON <other options> <source dir>
```
during configuration and then doing `make && make install`

#### Known Issues

When trying to run using the python interface you may get complaints relating to not being able to locate `libtorch.so` or `libtorch_cpu.so` library files. If so running

```
export LD_LIBRARY_PATH=`python3 -c 'import os;import torch;print(os.path.abspath(torch.__file__)[:-11])'`/lib:$LD_LIBRARY_PATH
```

should allow these files to be found


## Usage

A few simple example scripts using nuTens are available [here](https://github.com/ewanwm/nuTens/tree/main/examples)



## Benchmarks
nuTens uses [Googles benchmark library](https://github.com/google/benchmark) to perform benchmarking and tracks the results uing [Bencher](https://bencher.dev). Each benchmark consists of calculating neutrino oscillations for 1024 random variations of parameters in the 3 flavour formalism for 1024 neutrino energies in vacuum and in constant density matter:

<p align="center">  
<a
  href="https://bencher.dev/perf/nutens?lower_value=false&upper_value=false&lower_boundary=false&upper_boundary=false&x_axis=date_time&branches=9fb1fa7d-4e90-4889-a370-8488dea67849&testbeds=49818c12-6c02-42a2-bbbb-697a772d8991&benchmarks=700b0d80-ef19-4fac-bc84-45d558df1801&measures=fc8c0fd1-3b41-4ce7-826c-74843c2ea71c&start_time=1718212890927&tab=plots&plots_search=36aa4017-86a3-47ff-8c39-b77045d5268b&key=true&reports_per_page=4&branches_per_page=8&testbeds_per_page=8&benchmarks_per_page=8&plots_per_page=8&reports_page=1&branches_page=1&testbeds_page=1&benchmarks_page=1&plots_page=1">
  <img
    src="https://api.bencher.dev/v0/projects/nutens/perf/img?branches=9fb1fa7d-4e90-4889-a370-8488dea67849&testbeds=49818c12-6c02-42a2-bbbb-697a772d8991&benchmarks=700b0d80-ef19-4fac-bc84-45d558df1801&measures=fc8c0fd1-3b41-4ce7-826c-74843c2ea71c&start_time=1718212890927&title=Const+Density+Osc+Benchmark"
  title="Const Density Osc Benchmark" 
  alt="Const Density Osc Benchmark for nuTens - Bencher" /></a>
</p>

<p align="center">
<a 
  href="https://bencher.dev/perf/nutens?lower_value=false&upper_value=false&lower_boundary=false&upper_boundary=false&x_axis=date_time&branches=9fb1fa7d-4e90-4889-a370-8488dea67849&testbeds=49818c12-6c02-42a2-bbbb-697a772d8991&benchmarks=bd0cdb00-102a-422a-a672-7f297e65fd7e&measures=fc8c0fd1-3b41-4ce7-826c-74843c2ea71c&start_time=1718212962301&tab=plots&plots_search=097d254e-f328-4643-9e51-7b37436df615&key=true&reports_per_page=4&branches_per_page=8&testbeds_per_page=8&benchmarks_per_page=8&plots_per_page=8&reports_page=1&branches_page=1&testbeds_page=1&benchmarks_page=1&plots_page=1">
  <img
    src="https://api.bencher.dev/v0/projects/nutens/perf/img?branches=9fb1fa7d-4e90-4889-a370-8488dea67849&testbeds=49818c12-6c02-42a2-bbbb-697a772d8991&benchmarks=bd0cdb00-102a-422a-a672-7f297e65fd7e&measures=fc8c0fd1-3b41-4ce7-826c-74843c2ea71c&start_time=1718212962301&title=Vacuum+Osc+Benchmark" 
  title="Vacuum Osc Benchmark" 
  alt="Vacuum Osc Benchmark for nuTens - Bencher" 
/></a>

</p>


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
- [x] Add suite of benchmarking tests
- [x] Integrate benchmarks into CI ( maybe use [hyperfine](https://github.com/sharkdp/hyperfine) and [bencher](https://bencher.dev/) for this? )
- [ ] Add proper unit tests
- [x] Expand CI to include more platforms
- [ ] Add support for modules (see [PyTorch doc](https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html))
- [ ] Propagation in variable matter density
- [ ] Add support for Tensorflow backend
- [x] Add python interface 

