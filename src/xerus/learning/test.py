import xerus as xe
import measuredTensor as mT
from time import process_time
from contextlib import contextmanager

@contextmanager
def timer():
	try:
		duration = process_time()
		yield
	finally:
		duration = process_time() - duration
		print("Elapsed time: {0}s".format(duration))

order = 10
dimension = 10
rank = 20
#TODO: numMeasurements == 10 raises an exception
numMeasurements = 100000
first_cmp_meas = [10, 10]  # [10]

result = xe.TTTensor.random([dimension]*order, [rank]*(order-1))
assert xe.frob_norm(result-xe.deserialize(xe.serialize(result)))/xe.frob_norm(result) < 1e-14, xe.frob_norm(result-xe.deserialize(xe.serialize(result)))/xe.frob_norm(result)

print("Generating", numMeasurements, "random measures...")
rk1ms = list()
for i in range(numMeasurements):
	rk1m = list()
	rk1m.append(xe.Tensor.random(first_cmp_meas))
	for j in range(order):
		rk1m.append(xe.Tensor.random([dimension]))
	rk1ms.append(rk1m);

#TODO: check if the generic mean function is slower than a more specific mean method of RankOneMeasuredTTTensor
print("Initializing MeasuredTensor...")
with timer():
	mtt = mT.RankOneMeasuredTTTensor(result, rk1ms);

print("numComponents:", mtt.numComponents)
print("numMeasurements:", mtt.numMeasurements)
print("corePosition:", mtt.corePosition)
print("Moving core...")
with timer():
	mtt.corePosition = 1
print("corePosition:", mtt.corePosition)
assert xe.frob_norm(mtt.measuredTensor-result)/xe.frob_norm(result) < 1e-14, xe.frob_norm(mtt.measuredTensor-result)/xe.frob_norm(result)

print("Computing mean (slow)...")
print("Batch size:", mtt.numMeasurements)
batch = list(range(mtt.numMeasurements))
mean = xe.Tensor([10]);
#TODO: time using a c++-function for reduction (i.e. a function that takes mtt) vs using a python function (i.e. xerus::mean)
with timer():
	mean = mT.mean(lambda idx: mtt.value(idx), batch, mean);
print("Mean value:", mean)

print("Computing mean (fast)...")
print("Batch size:", mtt.numMeasurements)
batch = list(range(mtt.numMeasurements))
#TODO: time using a c++-function for reduction (i.e. a function that takes mtt) vs using a python function (i.e. xerus::mean)
with timer():
	mean = mT.mean_value(mtt, batch);
print("Mean value:", mean)

print("Exiting")
