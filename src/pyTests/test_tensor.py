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
        self.assertAlmostEqual(	ten_init_dim.frob_norm(), 0)

        def fun(ind):
            return ind[0] + ind[1] + ind[2]   
        ten_init_fun = xe.Tensor.from_function([7,7,7], fun)
        print(ten_init_fun.one_norm())
        self.assertAlmostEqual(	ten_init_fun[3,4,5], 12)        
        self.assertAlmostEqual(	ten_init_fun.order(), 3)
        self.assertAlmostEqual(	ten_init_fun.frob_norm(), np.sqrt(31899))
        self.assertAlmostEqual(	ten_init_fun.one_norm(),3087)

        ten_init_ones = xe.Tensor.ones([2,3,2])
        self.assertAlmostEqual(ten_init_ones.frob_norm(), np.sqrt(12))

        ten_init_random = xe.Tensor.random([2,3,2])
        self.assertGreater(ten_init_random.frob_norm(), 0)

        ten_init_id = xe.Tensor.identity([2,3,2,3])
        self.assertAlmostEqual(ten_init_id.frob_norm(), np.sqrt(6))

        ten_init_kron = xe.Tensor.kronecker([2,3,2,3])
        self.assertAlmostEqual(ten_init_kron[1,1,1,1], 1)
        self.assertAlmostEqual(ten_init_kron[1,1,1,0], 0)

        ten_init_dirac = xe.Tensor.dirac([2,3,2,3],[1,1,1,1])
        self.assertAlmostEqual(ten_init_dirac[1,1,1,1], 1)
        self.assertAlmostEqual(ten_init_dirac[1,1,1,0], 0)

    def test_reinterpret_dimensions(self):
        T = xe.Tensor.random([2,3,4,5])
        T_new = T.dense_copy()
        T.reinterpret_dimensions([6,20])
        self.assertAlmostEqual(T[4,6], T_new[1,1,1,1])
        
                              






