
.. _testing:

Testing 🧪
==========

Tests live in the `tests <https://github.com/ewanwm/nuTens/tree/main/tests>`_ directory. For c++ tests we use Googles `GoogleTest <http://google.github.io/googletest/primer.html>`_ framework and for python tests we use `pytest <https://docs.pytest.org/en/stable/>`_ in order to organise our tests and more easily track them.
Tests are run as part of our continuous integration and generally for every pull request all tests should pass (i.e. your new change should not break the behaviour of existing code).

If you are adding a new feature to nuTens, you should also include some tests of your new feature.
These tests should ideally cover all of the new code that you are adding and should be as "simple" as possible i.e. they should rely as little as possible on other parts of the code.
Try to test the smallest possible "units" of your code on their own, ideally against externally calculated values (unit tests), and also the overall behaviour of your new feature (integration tests).

If your new feature has a python interface, new python tests should also be added.

Coverage
--------

The coverage (how much of the source code appears in tests) of our test suite is tracked using `CodeCov <https://codecov.io/github/ewanwm/nuTens>`_. 
We would ideally like for this to be as close as possible to 100% (though as you can see we aren't quite there yet).

For this reason, new features should be covered as close as possible to 100%. 
You will be able to see the coverage of added code in any pull request.
Unless there is a good reason for not doing so, new code should only be merged if it is fully covered by new tests.

.. _profiling:

Profiling 📊
============

nuTens is instrumented with a custom profiler. Compiling with the cmake option 

.. code::
    
    -D NT_PROFILING=ON

will enable the profiling information. You can then profile an application by adding the following to the start of the main() function:

.. code::

    int main()
    {
        NT_PROFILE_BEGINSESSION("<name of the profile>");

        NT_PROFILE();

        // ...
        // the rest of your application
        // ...

and adding the NT_PROFILE_ENDSESSION() to the end:

.. code::

        // ...
        // the rest of your application
        // ...

        NT_PROFILE_ENDSESSION()
    
    }

Now after running that application, a file will be produced called "<name of the profile>.json" containing profile information. 

You can view the contents of this file in a browser.

If using firefox, go to 

https://profiler.firefox.com/

If using chrome, open chrome and type 

.. code::
    
    chrome://tracing

You can then drag and drop the json profile into the profiler.

.. _benchmarking:

Benchmarking 📉
===============

nuTens uses `Googles benchmark library <https://github.com/google/benchmark>`_ to perform benchmarking. 

To compile the benchmark executable you will need to configure nuTens with the cmake option

.. code::

    -DNT_ENABLE_BENCHMARK=ON

Then after make installing, there will be a ``benchmarks/`` directory in the build directory containing the ``benchmark`` executable.

You can run this with all command line options available that are specified in the `benchmark user guide <https://github.com/google/benchmark/blob/main/docs/user_guide.md>`_.

This is useful to run to compare speed after your changes to the code with the main branch (see below). In fact, this is done as part of the continuous integration process.

Code that causes serious performance regressions in the benchmarks should be very carefully considered before being committed to the main branch.

Results
-------

the main branch is benchmarked every time there is a commit, and the results are tracked uing `Bencher <https://bencher.dev>`_. Each benchmark consists of calculating neutrino oscillations for 1024 random variations of parameters in the 3 flavour formalism for 1024 neutrino energies in vacuum and in constant density matter:

.. raw:: html

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

Precompiled Headers 🗿
======================

nuTens has the option to use precompiled headers which can significantyly speed up build times.
To enable this feature use the :code:`NT_USE_PCH` cmake option:

.. code::

    cmake [other options] -DNT_USE_PCH=ON <path to src>
    