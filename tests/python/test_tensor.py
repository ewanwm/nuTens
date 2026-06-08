import unittest
import math as m
import nuTens as nt
from nuTens.tensor import Tensor, matmul, scale
from nuTens import dtype

import numpy as np
import typing

import unittest
import math as m
import numpy as np
import nuTens as nt

class TestTensorNumpy(unittest.TestCase):

    def test_numpy_to_tensor(self):
        """ Test that we can make tensor from numpy array
        """

        array = np.array([[0.0, 1.0], [2.0, 3.0]], dtype = np.float32)
        tensor = Tensor(array, requires_grad = False)

        self.assertEqual(tensor.get_value([0, 0]), array[0, 0],
                        f"numpy conversion failed")
        self.assertEqual(tensor.get_value([0, 1]), array[0, 1],
                        f"numpy conversion failed")
        self.assertEqual(tensor.get_value([1, 0]), array[1, 0],
                        f"numpy conversion failed")
        self.assertEqual(tensor.get_value([1, 1]), array[1, 1],
                        f"numpy conversion failed")

    def test_tensor_to_numpy(self):
        """ Test that we can make a numpy array from a tensor
        """

        tensor = Tensor.zeros([2, 2], dtype=dtype.scalar_type.float, requires_grad=False)

        tensor.set_value([0,1], 1.0)
        tensor.set_value([1,0], 2.0)
        tensor.set_value([1,1], 3.0)

        array = tensor.numpy()

        self.assertEqual(tensor.get_value([0, 0]), array[0, 0],
                        f"numpy conversion failed")
        self.assertEqual(tensor.get_value([0, 1]), array[0, 1],
                        f"numpy conversion failed")
        self.assertEqual(tensor.get_value([1, 0]), array[1, 0],
                        f"numpy conversion failed")
        self.assertEqual(tensor.get_value([1, 1]), array[1, 1],
                        f"numpy conversion failed")

class TestMaths(unittest.TestCase):
    """ check math functions
    """

    one = Tensor.ones([1], dtype=dtype.scalar_type.float)

    def test_scale(self):
        self.assertEqual((self.one * 2.0).get_value([0]), 2.0,
                        f"1 * 2 != 2")
