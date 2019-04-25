import unittest
import numpy as np
import xerus as xe
import pickle


def generate_random_tttensors(num_tests, max_order, max_dimension, max_rank, random):
	orders = random.randint(low=1, high=max_order+1, size=num_tests)
	dimensions = [random.randint(low=1, high=max_dimension, size=order) for order in orders]
	ranks = [random.randint(low=1, high=max_rank, size=(order-1)) for order in orders]
	for dim, rk in zip(dimensions, ranks):
		yield xe.TTTensor.random(dim.tolist(), rk.tolist())


def generate_random_tensors(num_tests, max_order=4, max_dimension=10, random=None):
	orders = random.randint(low=1, high=max_order+1, size=num_tests)
	dimensions = [random.randint(low=1, high=max_dimension, size=order) for order in orders]
	for dim in dimensions:
		yield xe.Tensor.random(dim.tolist())


def __test_tensor(A):
	name = "test_pickle_" + "-".join(map(str, A.dimensions))

	def test_pickle(self):
		bytes = pickle.dumps(A)
		Au = pickle.loads(bytes)
		self.assertEqual(Au.dimensions, A.dimensions)
		self.assertLessEqual(xe.frob_norm(A-Au), 1e-16)

	return name, test_pickle


# def test_tttensor(A):
#     name_d = "-".join(map(str, A.dimensions))
#     name_r = "-".join(map(str, A.ranks()))
#     name = "test_pickle_{}_{}".format(name_d, name_r)


def build_TestPickleTensor(seed, num_tests):
	random = np.random.RandomState(seed)
	odir = dict(__test_tensor(t) for t in generate_random_tensors(num_tests, random=random))
	return type("TestPickleTensor", (unittest.TestCase,), odir)

TestPickleTensor = build_TestPickleTensor(0, 20)
