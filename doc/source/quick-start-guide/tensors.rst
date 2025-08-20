
=======
Tensors
=======

The interface for tensors in nuTens is *heavily* inspired by pyTorch tensors. This is mainly because nuTens tensors are essentially little more than a wrapper around a pytorch tensor. 
Thus, it is recommended that you familiarise yourself a bit with pytorch and pytorch tensors. 
They provide some very nice resources to do this including `videos and tutorials <https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html>`_ which are very helpful.

Here we will give a specific example of using nuTens tensors, with more of a focus on neutrino oscillations, by "manually" calculating oscillation probabilities for the simple case of two neutrino flavours.

Creating Tensors
----------------

The Tensor class provides a number of static creator methods to build tensors. These are the preferred way to initialise tensors and you should avoid directly creating them yourself.
The creator methods are 

================= ====================================== ===================
c++               Python                                 Description
================= ====================================== ===================
Tensor::eye       :py:meth:`nuTens.tensor.Tensor.eye`    Creates an identity tensor (ones along the diagonal, zeros everywhere else)
Tensor::rand      :py:meth:`nuTens.tensor.Tensor.rand`   Creates a tensor with random elements (in the range [0.0, 1.0])
Tensor::diag      :py:meth:`nuTens.tensor.Tensor.diag`   Creates a tensor with specified values along the diagonal
Tensor::zeros     :py:meth:`nuTens.tensor.Tensor.zeros`  Creates a tensor filled with zeros
Tensor::ones      :py:meth:`nuTens.tensor.Tensor.ones`   Creates a tensor filled with ones
================= ====================================== ===================

Let's create a single entry tensor that we will use as the mixing angle for our 2-flavour neutrino mixing example.
We have to specify the shape, data type, and a device for the tensor to live on, and also specify whether or not gradients will be required for this tensor (these can be changed later with setter methods).
Whether or not gradients are needed for a tensor generally depends where it sits in the computational "chain", generally this should only be true for tensors which sit at the very start of a computation, e.g. the input oscillation parameters for your oscillation models.
Basically for anyhing you will want to know the gradient of some other quantity with respect to, this should be true.

.. tabs::

    .. code-tab:: c++

        #include <nuTens/tensors/tensor.hpp>

        Tensor theta = Tensor::zeros(/*shape=*/{1}, /*dtype=*/NTdtypes::kFloat, /*device=*/NTdtypes::kCPU, /*requiresGrad=*/true);

    .. code-tab:: py

        from nuTens.tensor import Tensor
        from nuTens import dtype

        theta = Tensor.zeros(shape=[1], dtype=dtype.scalar_type.float, device=dtype.device_type.cpu, requires_grad=True)

As mentioned above, we can change the dtype, device and requiresGrad options using the provided setter methods

==================== ============================================= ===================
c++                  Python                                        Description
==================== ============================================= ===================
Tensor::dType        :py:meth:`nuTens.tensor.Tensor.dtype`         Set the data type of the tensor
Tensor::device       :py:meth:`nuTens.tensor.Tensor.device`        Move the tensor to another device
Tensor::requiresGrad :py:meth:`nuTens.tensor.Tensor.requires_grad` Set whether gradients are required for this tensor
==================== ============================================= ===================

These setter methods can be chained together to create tensors, as an alternative way of achieving the same result above

.. tabs::

    .. code-tab:: c++

        #include <nuTens/tensors/tensor.hpp>

        Tensor theta = Tensor::zeros(/*shape=*/{1}).dtype(NTdtypes::kFloat).device(NTdtypes::kCPU).requiresGrad(true);

    .. code-tab:: py
        
        from nuTens.tensor import Tensor
        from nuTens import dtype

        theta = Tensor.zeros(shape=[1]).dtype(dtype.scalar_type.float).device(dtype.device_type.cpu).requires_grad(True)


Setting Values
--------------

Setting single values of a tensor is simply a case of calling the setValue method (set_value in python), providing the index of the element, and the new value.
Note however, that if the tensor has the `requires gradient` attribute set to true, then it must be set to false before altering the values (then it can be re-enabled after setting is done).

Lets set the value in our :math:`\theta` tensor to :math:`\frac{ \pi }{ 8 }`:

.. tabs::

    .. code-tab:: c++

        theta.requiresGrad(false);
        
        theta.setValue({0}, M_PI / 8.0);
        
        theta.requiresGrad(true);

    .. code-tab:: py
        
        import math as m
        
        theta.requires_grad(Talse)

        theta.set_value([0], m.pi / 8.0)
        
        theta.requires_grad(True)

.. note::
    As in pyTorch, nuTens supports more complex "slicing" of tensors too. 
    You can read more about this in the :ref:`Indexing` section of the user manual.

If you are using the c++ interface, and will be performing a large number of these setValue operations on single elements of your tensor, for example inside of a loop, it may be faster to use an AccessedTensor. 
You can read more about these in :ref:`Accessed Tensors`

We can inspect our tensor to check that the value has been set by printing it using the `toString` method (`to_string` in python):

.. tabs::

    .. code-tab:: c++

        std::cout << tensor.toString() << std::endl;

    .. code-tab:: py
        
        print(tensor.to_string())

which will give us 

.. code::

    0.7854
    [ CPUFloatType{1} ]

Retrieving Values
-----------------

Individual values can be retrieved from tensors using the `getValue` (`get_value` in python) method:

.. note::
    In c++ this method is templated, so that the return type can be known at compile time, but in python there is no such condition.

.. tabs::

    .. code-tab:: c++

        float thetaValue = theta.getValue<float>({0});

        std::cout << "theta as float = " << thetaValue << std::endl;

    .. code-tab:: py
        
        theta_value = theta.get_value([0])
        
        print(f"theta as float = {theta_value}")

.. code::

    theta as float = 0.7853981852531433

.. note::
    As with setting values, slicing is also implemented for retrieving values.
    See :ref:`Indexing` for more details.

Units
-----

It is generally keeps things nice and simple to work with natural units in calculating neutrino oscillations. 
To this end, nuTens provides a number of standard units (which are really just conversion factors to eV).
These can be found in :doc:`../breathe-apidoc/file/units_8hpp.html` header file for c++, and the :py:mod:`nuTens.units` module in python.

Tensor Operations
-----------------

nuTens exposes a subset of mathematical operations on tensors that are useful for neutrino oscillations.
These typically have the signature ``Tensor::<function>(tensor1)`` for unary operators like ``sin`` and ``cos``, or ``Tensor::<function>(tensor1, tensor2)`` for binary operators like addition or matrix multiplication, or ``Tensor::<function>(tensor1, scalar)`` for functions operating on a tensor and a scalar like scalar multiplication or division.

.. note::
    See the Tensor API page (:doc:`c++<../breathe-apidoc/class/classTensor>`, :doc:`python<../python-api/tensor>`) for a full list of available operations.

Here we will use some of these operations to calculate oscillation probabilities for our simple 2-flavour case.

Lets calculate the two flavour oscillation probability from flavour a to flavour b using

.. math::

    P_{a \rightarrow b}(E) = \sin^2(2\theta)\sin^2 \left( \frac{\Delta m^2 L}{ 4 E } \right)

using a baseline of 295 km and an energy of 0.5 GeV, with a :math:`\Delta m^2` of :math:`20 \times 10 ^{-4}` eV

and lets define 

.. math::

    \Phi \equiv \frac{\Delta m^2 L} { 4 E }

for convenience

Using nuTens tensors this looks like:

.. tabs::

    .. code-tab:: c++

        #include <nuTens/propagator/units.hpp>

        Tensor sin2Theta = Tensor::sin(Tensor::scale(theta, 2.0));

        Tensor sinSquared2Theta = Tensor::pow(sin2Theta, 2.0);
        
        // for now we will just use floats for the other variables 
        // to keep things (relatively) simple
        
        float dm2 = 0.5 * Units::eV;
        float baseline = 295 * Units::km;
        float energy = 0.5 * Units::GeV;

        float phi = dm2 * baseline / ( 4.0 * energy );

        float sinSquaredPhi = std::sin(phi) * std::sin(phi);

        Tensor oscProb = Tensor::Scale(sinSquared2Theta, sinSquaredPhi);

        std::cout << "oscillation probability = " << oscProb.toString() << std::endl;

    .. code-tab:: py
        
        from nuTens import units

        sin_2_theta = tensor.sin(tensor.scale(theta, 2.0))

        sin_squared_2_theta = tensor.pow(sin_2_theta, 2.0)
        
        # for now we will just use floats for the other variables 
        # to keep things (relatively) simple

        dm2 = 0.5 * units.eV
        baseline = 295 * units.km
        energy = 0.5 * units.GeV

        phi = dm2 * baseline / ( 4.0 * energy )

        sin_squared_phi = m.sin(phi) * m.sin(phi)

        osc_prob = tensor.scale(sin_squared_2_theta, sin_squared_phi)

        print(f"oscillation probability = {osc_prob.to_string()}")

which gives us

.. code::
        
    oscillation probability =  0.4973
    [ CPUFloatType{1} ]

as expected.

Automatic Differentiation
-------------------------

One of the most powerful tools offered by nuTens is the ability to perform automatic differentiation and extract the gradient of a calculated quantity with respect to any of its inputs.
This is built on top of pytorch and so the process of extracting these gradients is very similar to the process there.

In the example we have been using so far, we may want to extract the derivative of :math:`P_{a \rightarrow b}` with respect to :math:`\theta`.
In order to do this, we would first need to make sure that the ``requires gradient`` property of our :math:`\theta` tensor has been set (which we have already done).

We then call the :code:`.backward()` method on the quantity we want to differentiate (the :math:`P_{a \rightarrow b}` tensor)

.. tabs::

    .. code-tab:: c++

        oscProb.backward();

    .. code-tab:: py
        
        osc_prob.backward()

this performs backpropagation through the computational graph defined by our tensor computations, and will fill the gradients of any tensors that have the ``requires gradient`` property set.

We can then access this gradient 

.. tabs::

    .. code-tab:: c++

        Tensor gradient = theta.grad();

        std::cout << "gradient = " << gradient.toString() << std::endl;

    .. code-tab:: py
        
        gradient = theta.grad()
        
        print(f"gradient = {gradient.to_string()}")

.. code::

    gradient =  1.9893
    [ CPUFloatType{1} ]

Just to be sure this is right, we can calculate the probaility using our brains too:

.. math::

    \frac {\partial P_{a \rightarrow b} } { \partial \theta } = 4 \cos ( 2 \theta ) \sin ( 2 \theta ) \sin^2 \left( \frac{\Delta m^2 L}{ 4 E } \right)

.. math::

    = 4 \cos \left( \frac {\pi} {4} \right) \sin \left( \frac {\pi} {4} \right)  \sin^2 ( 1.4975... )
    
.. math::
    
    = 1.9893...

viola! it's the same!

This example is pretty simple, and we could have just calculated it by hand. 
However, there is almost no limit to how complex we can make these calculations.
We will see in the following parts of this tutorial how this can be used to create a fully differentioble neutrino oscillation model.

Full scripts containing the 2-neutrino oscillation calculation here can be found in "simple-tensor.[py/cpp]" in the `examples folder <https://github.com/ewanwm/nuTens/tree/main/examples>`_.

Oscillation Spectrum Example
----------------------------

*calculate oscillations for a range of energies (specified with a tensor), then plot the spectrum, and the derivative of the spectrum wrt L/E*