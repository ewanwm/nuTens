
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
