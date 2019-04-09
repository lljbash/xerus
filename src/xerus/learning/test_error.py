import numpy as np
import xerus as xe
import measuredTensor as mT
from time import process_time
from contextlib import contextmanager

@contextmanager
def timer(event=None):
	try:
		duration = process_time()
		yield
	finally:
		duration = process_time() - duration
		if event is None:
			print("Elapsed time: {0}s".format(duration))
		else:
			print("Elapsed time: {0}s ({1})".format(duration, event))


order = 10
dimension = 10
rank = 20
#TODO: numMeasurements == 10 raises an exception
numMeasurements = 100000
first_cmp_meas = [10, 10]  # [10]

result = xe.TTTensor.random([dimension]*order, [rank]*(order-1))

print("Generating", numMeasurements, "random measures...")
rk1ms = list()
for i in range(numMeasurements):
	rk1m = list()
	rk1m.append(xe.Tensor.random(first_cmp_meas))
	for j in range(order):
		rk1m.append(xe.Tensor.random([dimension]))
	rk1ms.append(rk1m);
refs = (10*np.random.rand(numMeasurements)).astype(int).tolist()

print("Initializing MeasuredTensor...")
with timer():
	mtt = mT.RankOneMeasuredTTTensor(result, rk1ms);

all_samples = list(range(mtt.numMeasurements))

print("Computing error...")
with timer("error_fast"):
	err_1 = mT.error_fast(mtt, refs, all_samples);
with timer("error"):
	err_2 = mT.error(mtt, refs, all_samples);
assert abs(err_1-err_2)/abs(err_1) < 1e-12
