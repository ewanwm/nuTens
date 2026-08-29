import pickle

import unittest

from nuTens.tensor import Tensor
from nuTens import dtype

class TestSerialisation(unittest.TestCase):
    """ check serialisation of tensors
    """

    def test_scale(self):

        one = Tensor.ones([3, 3], dtype=dtype.scalar_type.float)

        with open("test-tensor.pickle", "wb") as file:

            pickle.dump(one, file)

        with open("test-tensor.pickle", "rb") as file:
            loaded_one = pickle.load(file)

        print(one)
        print(loaded_one)

        self.assertEqual(
            loaded_one, one,
            f"unpickled tensor does not match original!"
        )
