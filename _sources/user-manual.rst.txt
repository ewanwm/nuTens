
.. _user-manual:

==============
User Manual 📗
==============

Tensors
-------

.. _Indexing:

Indexing
^^^^^^^^

.. _Accessed Tensors:

Accessed Tensors
^^^^^^^^^^^^^^^^

.. _gpu:

GPU
---

.. _batched-oscillation-calculations:

Batching Oscillation calculations
---------------------------------

.. warning::
    Not really properly working at the moment `:)` 


.. _dp-propagator:

Denton-Parke Propagator
------------------------

The Denton-Parke (DP) propagator is an implementation of the `nufast algorithm <https://arxiv.org/pdf/2405.02400>`_ using tensors to allow it to be automatically differentiated.
It is implemented in the DPpropagator class.
This propagator is less flexible than the general Propagator class: It can only be used to calculate 3 flavour oscillations in the usual PMNS parameterisation.
However what it lacks in flexibility it makes up for in speed. 
It leverages the `Eigenvector-eigenvalue identity <https://www.ams.org/journals/bull/2022-59-01/S0273-0979-2021-01722-8/>`_ to very quickly calculate the effective PMNS matrix in the presence of matter.

It's use is slightly different to the usual Propagator, as it requires no additional matter solver to be provided.

A basic usage example looks like 

.. tabs::

    .. code-tab:: c++

        // set up tensors for the oscillation parameters and energies
        Tensor theta23 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
        Tensor theta13 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
        Tensor theta12 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
        Tensor deltaCP = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
        Tensor dmsq21 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);
        Tensor dmsq31 = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, false);

        Tensor energies = Tensor::ones({1, 1}, dtypes::kComplexFloat).requiresGrad(false).hasBatchDim(true);

        // instantiate the propagator
        DPpropagator dpPropagator = DPpropagator(baseline, false, density, 10);

        // link parameters to the propagator
        dpPropagator.setEnergies(energies);
        dpPropagator.setParameters(theta12, theta23, theta13, deltaCP, dmsq21, dmsq31);

        // set values of the parameters
        theta23.setValue({0}, 0.4 * M_PI);
        theta13.setValue({0}, 0.3 * M_PI);
        theta12.setValue({0}, 0.2 * M_PI);

        dmsq21.setValue({0}, m1 * m1 - m2 * m2);
        dmsq31.setValue({0}, m1 * m1 - m3 * m3);

        deltaCP.setValue({0}, dcp);

        // calculate probabilities
        Tensor dpProbabilities = dpPropagator.calculateProbs();

    .. code-tab:: py
        
        .. warning::

            DPpropagator not yet tested for python, but should probably work


.. _autograd:

Automatic Differentiation
-------------------------

Using Backward()
^^^^^^^^^^^^^^^^

Calling `backward()` on an endpoint tensor in a computation will do backpropagation through the computation graph and populate the gradient attribute of all tensors which habe the attribute `requiresGrad == true`.
This can then be accessed via the `grad()` method.

e.g.

.. tabs::

    .. code-tab:: c++

        #include <nuTens/tensors/tensor.hpp>

        // set up the input tensor
        Tensor input = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU, /*requiresGrad=*/true);
        
        // perform some computation
        Tensor output = Tensor::exp(input * 3.0);

        // perform backpropagation
        output.backward();

        // get the accumulated gradient from the input tensor
        Tensor gradient = input.grad();

    .. code-tab:: py
        
        from nuTens.tensor import Tensor
        
        # set up the input tensor
        input = Tensor.ones([1], requires_grad = True)
        
        # perform some computation
        output = Tensor.exp(input * 3.0)

        # perform backpropagation
        output.backward()

        # get the accumulated gradient from the input tensor
        gradient = input.grad()


Using Autograd Module
^^^^^^^^^^^^^^^^^^^^^

The `autograd::grad()` method can be used to calculate the derivative of one tensor with respect to another.

e.g.

.. tabs::

    .. code-tab:: c++

        #include <nuTens/tensors/tensor.hpp>
        #include <nuTens/tensors/autograd.hpp>

        // set up the input tensor
        Tensor input = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU, /*requiresGrad=*/true);
        
        // perform some computation
        Tensor output = Tensor::exp(input * 3.0);

        // get the gradient
        Tensor gradient = autograd::grad(output, input);

    .. code-tab:: py
        
        from nuTens.tensor import Tensor
        from nuTens import autograd
        
        # set up the input tensor
        input = Tensor.ones([1], requires_grad = True)
        
        # perform some computation
        output = Tensor.exp(input * 3.0)

        # get the gradient
        gradient = autograd.grad(output, input)


Higher Order Derivatives
^^^^^^^^^^^^^^^^^^^^^^^^

`autograd::grad()` can also be used to calculate higher order Derivatives

e.g.

.. tabs::

    .. code-tab:: c++

        #include <nuTens/tensors/tensor.hpp>
        #include <nuTens/tensors/autograd.hpp>

        // set up the input tensor
        Tensor input = Tensor::ones({1}, dtypes::kFloat, dtypes::kCPU, /*requiresGrad=*/true);
        
        // perform some computation
        Tensor output = Tensor::exp(input * 3.0);

        // get the gradient
        Tensor gradient = autograd::grad(output, input);
        Tensor secondDeriv = autograd::grad(gradient, input);

    .. code-tab:: py
        
        from nuTens.tensor import Tensor
        from nuTens import autograd
        
        # set up the input tensor
        input = Tensor.ones([1], requires_grad = True)
        
        # perform some computation
        output = Tensor.exp(input * 3.0)

        # get the gradient
        gradient = autograd.grad(output, input)
        second_deriv = autograd.grad(gradient, input)

Disabling Autograd calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are not interested in computing gradients, you may wish to disable the automatic differentiation machinery as it can incur performance penalties.
This can be done with the NoGrad guard class.

You can disable autograd related computations in a specific scope like:

.. tabs::

    .. code-tab:: c++

        #include <nuTens/tensors/tensor.hpp>
        #include <nuTens/tensors/autograd.hpp>

        {
            // while this object exists, i.e. within the current scope
            // no computations related to automatic differentiation will be performed 
            auto noGrad = autograd::NoGrad();

            // ... Do some computation ...
        
        }

        // now autodiff will be re-enabled
    
    .. code-tab:: py

        from nuTens.tensor import Tensor
        from nuTens import autograd

        def foo():
            
            # while this object exists, i.e. within the current scope
            # no computations related to automatic differentiation will be performed 
            noGrad = autograd.NoGrad()

            # ... Do some computation ...
        
        }

        # now autodiff will be re-enabled
