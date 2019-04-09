import numpy as np

def easy(xs):
	marginal = lambda x: np.sin(np.pi * x)
	xs = np.asarray(xs)
	if not xs.ndim == 2:
		raise AssertionError
	ret = np.prod(marginal(xs), axis=0).reshape(-1, 1)
	if not ret.shape == (xs.shape[1], 1):
		raise AssertionError
	return ret


def hard(xs):
	marginal = lambda x: abs(np.sin(np.pi * x))
	xs = np.asarray(xs)
	if not xs.ndim == 2:
		raise AssertionError
	ret = np.prod(marginal(xs), axis=0).reshape(-1, 1)
	if not ret.shape == (xs.shape[1], 1):
		raise AssertionError
	return ret
