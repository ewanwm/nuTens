
==========
Propagator
==========

In the example previously, we saw how we can calculate oscillation probabilities for two flavours using tensore "by hand". 
But the point of nuTens is to perform these calculations for you so that you don't have to!

This is exactly the job of the Propagator class.
In this section we will cover the basics of using this class and provide some examples.

Configurng the Propagator
-------------------------

We instantiate a Propagator, specifying the number of neutrino generations, and a baseline, like so:

.. tabs::

    .. code-tab:: c++

        #include <nuTens/propagator/propagator.hpp>

        Propagator propagator(/*nGenerations=*/3, /*baseline=*/295*units::km);

    .. code-tab:: py

        from nuTens.propagator import Propagator
        from nuTens import units

        propagator = Propagator(n_generations = 3, baseline = 295.0 * units.km)

if we decide later that we want to change the baseline, we can always change it like so:

.. tabs::

    .. code-tab:: c++

        propagator.setBaseline(600 * units::km);

    .. code-tab:: py

        propagator.set_baseline(600 * units::km)

.. warning::

    It is likely that in the near future the baseline will be changed to a Tensor so that we can calculate gradients with respect to it.
    This may change the interface here.

Setting a Mixing Matrix
^^^^^^^^^^^^^^^^^^^^^^^

We will also need to specify a mixing matrix

.. tabs::

    .. code-tab:: c++

        #include <nuTens/tensors/tensor.hpp>

        Tensor theta = Tensor::zeros(/*shape=*/{1}, /*dtype=*/NTdtypes::kFloat, /*device=*/NTdtypes::kCPU, /*requiresGrad=*/true);

    .. code-tab:: py

        from nuTens.tensor import Tensor
        from nuTens import dtype

        theta = Tensor.zeros(shape=[1], dtype=dtype.scalar_type.float, device=dtype.device_type.cpu, requires_grad=True)



Calculating Oscillation probabilities
-------------------------------------

.. note::
    We specify a 1 at the start of the shape. This is to allow for batched calculations (see :ref:`batched-oscillation-calculations`).
    This extra dummy dimension is essentially the same idea as batch dimensions in machine learning, for those familiar.

Oscillation Spectrum Example
----------------------------

*calculate oscillations for a range of energies (specified with a tensor), then plot the spectrum, and the derivative of the spectrum wrt L/E*

Other Propagators
-----------------

The Propagator class described above is the most generic one offered by nuTens.
Your needs however may be more specific. 
As such nuTens offers some other specialised propagators :

* The :ref:`dp-propagator` class offers reduced flexibility but far greater speed for 3 flavour oscillations