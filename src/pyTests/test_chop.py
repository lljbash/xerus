import unittest
import numpy as np
import xerus as xe


def build_TestChop(seed, num_tests, max_order=20, max_dimension=100, max_rank=30):
    random = np.random.RandomState(seed)
    orders = random.randint(low=1, high=max_order+1, size=num_tests)
    dimensions = [random.randint(low=1, high=max_dimension, size=order) for order in orders]
    ranks = [random.randint(low=1, high=max_rank, size=(order-1)) for order in orders]

    def test(dimensions, ranks):
        def test_chop(self):
            A = xe.TTTensor.random(dimensions.tolist(), ranks.tolist())
            L,l,e,r,R = xe.indices(5)

            mirrored = lambda pos: A.degree()-1-pos

            for corePosition in range(A.degree()):
                Al,Ar = A.chop(corePosition)
                Ac = A.get_component(corePosition)
                res = xe.TensorNetwork()
                res(L^corePosition,e,R^mirrored(corePosition)) << Al(L&1,l) * Ac(l,e,r) * Ar(r,R&1)
                # norm(A - res)**2 == norm(A)**2 - 2*inner(A,res) + norm(res)**2
                nA = xe.frob_norm(A)
                nres = xe.frob_norm(res)
                inner = xe.Tensor()
                inner() << A(e&0) * res(e&0)
                self.assertLessEqual(nA**2 - 2*inner[0] + nres**2, 5e-5)
        return test_chop

    name = lambda d,r: "test_chop_{}_{}".format(".".join(d.astype(str)), ".".join(r.astype(str)))
    odir = {name(d,r): test(d,r) for d,r in zip(dimensions, ranks)}

    return type("TestChop", (unittest.TestCase,), odir)
TestChop = build_TestChop(0, 100, 4)
