import unittest
from itertools import product
ranges = lambda *args: product(*[range(arg) for arg in args])
import xerus as xe
import numpy as np


class TestTensor(unittest.TestCase):
    def setUp(self):
        self.dimension = [10, 15, 20]
        self.ranks = [1, 8, 4, 1]

    def test_create_tensors(self):
        ten_init = xe.Tensor()
        ten_init_dim = xe.Tensor([3,4,5])
        self.assertAlmostEqual(	ten_init_dim[0,0,0], 0)
        def fun(ind):
            return ind[0] + ind[1] + ind[2]   
        ten_init_fun = xe.Tensor.from_function([7,7,7], fun)
        self.assertAlmostEqual(	ten_init_fun[3,4,5], 12)        
        self.assertAlmostEqual(	ten_init_fun.order(), 3)




