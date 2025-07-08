
Profiling
=========

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


Now after running that application, a file will be produced called "<name of the profile>.json" containing profile information. 

You can view the contents of this file in a browser.

If using firefox, go to 

https://profiler.firefox.com/

If using chrome, open chrome and type 

chrome://tracing

You can then drag and drop the json profile into the profiler.

