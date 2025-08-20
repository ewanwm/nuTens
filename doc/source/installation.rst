
============
Installation
============

Installing Using Pip
--------------------

A pypi distribution of nuTens is provided, the page for which can be found `here <https://pypi.org/project/nuTens/>`_.

This means that you can install nuTens via pip using the command
 
.. code::

    pip install nuTens


.. note::

    The python interface can be installed manually after cloning the repository using pip by running

    .. code::
    
        pip install .
    
    in the root directory of nuTens.

    This may be useful if you are developing nuTens.


Installing From Source
----------------------

Step 1 is to get your hands on a copy of the nuTens source code.

To do this you can either clone the repository from github

.. code:: 

    git clone git@github.com:ewanwm/nuTens.git

and checkout the release you want

.. code::

    git checkout tags/v<X.Y.Z>

or visit the `releases <https://github.com/ewanwm/nuTens/releases>`_ page.

Requirements
^^^^^^^^^^^^

CMake 
"""""
Should work with most modern versions. If you wish to use precompiled headers to speed up build times you will need CMake > 3.16.

Compiler 
""""""""
Requires compiler with support for c++17 standard - Tested with gcc and clang

PyTorch
"""""""
Currently pytorch is the only "backend" supported by nuTens. See https://pytorch.org/ for instructions on how to install it yourself, or you can use the requirements file provided as part of nuTens:

.. code::

    pip install -r PyTorch_requirements.txt

Building
^^^^^^^^

Create a build directory

.. code::

    mkdir build
    cd build

Configure using cmake (see :ref:`cmake-config-options` for more information on available build options). 
One particularly important option to be aware of here is ``NT_ENABLE_PYTHON=<ON/OFF>`` which determines whether or not to build the nuTens python interface.

.. note::
    If you build the python interface using this method (specifying ``-DNT_ENABLE_PYTHON=ON`` during cmake configuration) you will need to tell python where it can find the nuTens python module by setting the ``PYTHONPATH`` environment variable:

    .. code::

        export PYTHONPATH=<path to nuTens build directory>:$PYTHONPATH

.. code::

    cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` <path to source code>

Now build!

.. code::

    make <-j Njobs> && make install

Verifying Your Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you've installed nuTens, you can verify your installation by running

.. code::

    make test

and (if you have installed the python interface)

.. code::

    pip install pytest
    pytest tests

Known Issues
------------

When trying to run using the python interface you may get complaints relating to not being able to locate `libtorch.so` or `libtorch_cpu.so` library files. If so running

.. code::
    
    export LD_LIBRARY_PATH=`python3 -c 'import os;import torch;print(os.path.abspath(torch.__file__)[:-11])'`/lib:$LD_LIBRARY_PATH


should allow these files to be found


.. _cmake-config-options:

CMake Configuration Options
---------------------------

==========================  ============================================================================  =======
Option                      Description                                                                   Default
==========================  ============================================================================  =======
NT_USE_TORCH                Use torch as the backend for dealing with tensors                             ON
NT_TORCH_FROM_PIP           If it is not found, torch will be installed using pip                         ON
NT_ALLOW_GLOBAL_PYTHON_ENV  Allow installing pip packages in global python environment (Not recommended)  OFF
NT_COMPILE_TESTS            Whether or not to compile the test library                                    ON
NT_ENABLE_PYTHON            Enable compilation of the python interface                                    OFF
==========================  ============================================================================  =======

For Developers
^^^^^^^^^^^^^^

These options are a bit more "advanced" and probably only of interest to anyone actually writing nuTens library code.

======================  =====================================================================================================================================================================================   =======
Option                  Description                                                                                                                                                                             Default
======================  =====================================================================================================================================================================================   =======
NT_TORCH_FROM_SCRATCH   If it is not found, torch will be compiled from scratch using CPM (very slow but maybe useful for debugging builds)                                                                      OFF
NT_ENABLE_BENCHMARKING  Whether or not to compile benchmark executables                                                                                                                                         OFF
NT_BUILD_TIMING         Whether or not to time the build process                                                                                                                                                OFF
NT_LOG_LEVEL            Set the log level to one of <SILENT ERROR WARNING INFO DEBUG TRACE>                                                                                                                     INFO
NT_PROFILING            Enable profiling (see :ref:`profiling`)                                                                                                                                                 OFF
NT_TEST_COVERAGE        Add flags to allow checking of test coverage                                                                                                                                            OFF
NT_USE_PCH              Use precompiled headers to speed up the build process                                                                                                                                   OFF
BUILD_SHARED_LIBS       Whether or not to build shared libraries or static (not compatible with NT_ENABLE_PYTHON which requires static libraries. If both are specified BUILD SHARED_LIBS will be set to OFF)   ON
======================  =====================================================================================================================================================================================   =======

Building Against nuTens
-----------------------