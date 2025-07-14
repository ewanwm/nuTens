
=======
Tensors
=======

The interface for tensors in nuTens is *heavily* inspired by pyTorch tensors. This is mainly because nuTens tensors are essentially little more than a wrapper around a pytorch tensor. 
Thus, it is recommended that you familiarise yourself a bit with pytorch and pytorch tensors. 
They provide some very nice resources to do this including `videos and tutorials <https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html>`_ which are very helpful.

Here we will give some examples of using nuTens tensors, with more of a focus on neutrino oscillations.

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

Let's create a 2 x 2 matrix that we will use as out 2-flavour neutrino mixing matrix.
We have to specify the shape, data type, and a device for the tensor to live on, and also specify whether or not gradients will be required for this tensor (these can be changed later with setter methods).

.. tabs::

    .. code-tab:: c++

        #include <nuTens/tensors/tensor.hpp>

        Tensor PMNS = Tensor::zeros(/*shape=*/{1, 2, 2}, /*dtype=*/NTdtypes::kComplexFloat, /*device=*/NTdtypes::kCPU, /*requiresGrad=*/true);

    .. code-tab:: py

        from nuTens.tensors import Tensor, dtype

        pmns = Tensor.zeros(shape=[1, 2, 2], dtype=dtype.scalar_type.complex_float, device=dtype.device_type.cpu, requires_grad=True)

.. note::
    We specify a 1 at the start of the shape. This is to allow for batched calculations (see :ref:`batched-oscillation-calculations`).
    This extra dummy dimension is essentially the same idea as batch dimensions in machine learning, for those familiar.

As mentioned above, we can change the dtype, devide and requiresGrad options using the provided setter methods


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

        Tensor PMNS = Tensor::zeros(/*shape=*/{1, 2, 2}).dtype(NTdtypes::kComplexFloat).device(NTdtypes::kCPU).requiresGrad(true);

    .. code-tab:: py

        from nuTens.tensors import Tensor, dtype

        pmns = Tensor.zeros(shape=[1, 2, 2]).dtype(dtype.scalar_type.complex_float).device(dtype.device_type.cpu).requires_grad(True)


Setting Values
--------------

Tensor Operations
-----------------

Automatic Differentiation
-------------------------