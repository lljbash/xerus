# coding: utf-8
import numpy as np
import xerus as xe
from fmap import FMap
import measuredTensor as mT
from time import process_time
from contextlib import contextmanager
from itertools import chain, cycle

try:
	from tools.colors import red, yellow, green, blue, bold
except ImportError:
	from colorama import Fore, Style
	red = lambda s: Fore.RED + s + Fore.RESET
	blue = lambda s: Fore.BLUE + s + Fore.RESET
	bold = lambda s: Style.BRIGHT + s + Style.RESET_ALL

import myLogging as logging
logger = logging.getLogger(__name__)
logger.todo("abstrahiere den solver und teste ihn an einfachen problemen")

LAMBDA = 1e-6  #TODO: WEIGHT_DECAY
MIN_LOCAL_DOFS = 1000  #TODO: plot convergence rates for different MIN_LOCAL_DOFS

def h(s, marker="=", width=80):
	s = s.strip()
	l = (width - len(s) - 2)//2
	r = width - len(s) - l
	return marker*l + " " + s + " " + marker*r


def SsASD(meas, refs, sweeper, offset=None):
	# stochastic in sweep
	from scipy.optimize import minimize_scalar
	logger.debug("Stochastic sweep Alternating Steepest Descent")
	assert offset is not None

	def fnc(core, batch):
		meas.core = core
		return mT.loss(meas, refs, batch, offset)

	def grad(core, batch):
		meas.core = core
		return mT.d_loss(meas, refs, batch, offset)

	# def sgd(fnc, x0, args, grad, numSteps):
	#     ret = x0
	#     for it in range(numSteps):
	#         dx = grad(ret, *args)
	#         s = 1/(it+1)
	#         ret = ret-s*dx
	#     return ret

	def gd(fnc, x0, args, grad, numSteps):
		def stepsize(core, grad):
			stepsize_fnc = lambda s: fnc(core - s*grad, *args)
			res = minimize_scalar(stepsize_fnc, options={'maxiter': 10})  # maxiter == 10 since 2**(-10) ~= 1e-3 (bijection error...)
			s = res.x
			if stepsize_fnc(s) > stepsize_fnc(0): s = 0
			return s
		ret = x0
		for it in range(numSteps):
			dx = grad(x0, *args)
			s = stepsize(ret, dx)
			ret = ret-s*dx
		return ret

	def numSteps_and_batch():
		dofs = global_dofs(meas.measuredTensor)
		# batchSize = min(
		#         max(MIN_LOCAL_DOFS, int(np.ceil(2*dofs*np.log(dofs)))),
		#         meas.numMeasurements
		#     )
		# batch = np.random.choice(meas.numMeasurements, batchSize, replace=False).tolist()
		# numSteps = min(int(np.ceil(60000/batchSize)), 40)
		batch = np.arange(mtt.numMeasurements).tolist()
		numSteps = 1
		return numSteps, batch

	sweepLength = 2*(meas.numComponents-1)
	for itr, pos in enumerate(sweeper(meas)):
		if itr % sweepLength == 0:
			assert pos == 0
			numSteps, batch = numSteps_and_batch()
			logger.debug("Perform new sweep with batch size {}".format(len(batch)))
		meas.corePosition = pos
		logger.debug("Old local residuum estimate: {0:.2e}".format(fnc(meas.core, batch)))
		# meas.core = sgd(fnc, meas.core, (batch,), grad, numSteps)
		meas.core = gd(fnc, meas.core, (batch,), grad, numSteps)
		logger.debug("New local residuum estimate: {0:.2e}".format(fnc(meas.core, batch)))


logger.todo("distinguish local/global terminators?")
def ScASD(meas, refs, sweeper):
	# stochastic in components
	logger.debug("Stochastic component Alternating Steepest Descent")

	def fnc(core, batch):
		meas.core = core
		return mT.loss(meas, refs, batch)

	def grad(core, batch):
		meas.core = core
		return mT.d_loss(meas, refs, batch)

	def sgd(fnc, x0, args, grad, numSteps):
		ret = x0
		for it in range(numSteps):
			dx = grad(ret, *args)
			s = 1/(it+1)
			ret = ret-s*dx
		return ret

	def numSteps_and_batch():
		dofs = meas.core.size
		batchSize = min(
				max(MIN_LOCAL_DOFS, int(np.ceil(2*dofs*np.log(dofs)))),
				meas.numMeasurements
			)
		batch = np.random.choice(meas.numMeasurements, batchSize, replace=False).tolist()
		numSteps = min(int(np.ceil(60000/batchSize)), 40)
		return numSteps, batch

	for pos in sweeper(meas):
		meas.corePosition = pos
		numSteps, batch = numSteps_and_batch()
		logger.debug("Perform {} local SGD steps with batch size {}".format(numSteps, len(batch)))
		logger.debug("Old local residuum estimate: {0:.2e}".format(fnc(meas.core, batch)))
		meas.core = sgd(fnc, meas.core, (batch,), grad, numSteps)
		logger.debug("New local residuum estimate: {0:.2e}".format(fnc(meas.core, batch)))


# - teste den error des gespeicherten tensor bevor du ihn überschreibst! (nicht nur einen decrease im error des optimierten tensors)
# - verwende als decrease max(1/(1-1/K)*decrease_in_loss, decrease_in_error)  # betrachte da nochmal die macdonald-bound
# - warum waren im SsASD-diff-sweep die lokalen losses 0.224, der loss bei der berechnung des sweeps aber 2.3?
#   => weil der sweeper den offset nicht verwendet
# - warum ist der initial loss der retraction 10.1 und nicht einer der beiden werte oben?
#   => weil result+diff verrwendet wird (diff ist der random init-tensor) und nicht result+mtt_diff.measuredTensor


class custom_sweeper(object):
	def __init__(self, minRelDecrease, minSweeps, maxSweeps, initCorePosition=0, offset=None):
		self.minRelDecrease = minRelDecrease
		self.minSweeps = minSweeps
		self.maxSweeps = maxSweeps
		self.initCorePosition = initCorePosition

		self.offset = offset
		# all_samples = np.arange(len(ys)).tolist()
		if offset is None:
			self.error = lambda mtt: mT.error(mtt, ys, all_samples)
			self.loss = lambda mtt: mT.loss(mtt, ys, all_samples)
		else:
			self.error = lambda mtt: mT.error(mtt, ys, all_samples, offset)
			self.loss = lambda mtt: mT.loss(mtt, ys, all_samples, offset)

		self.errors = [np.inf]
		self.losses = [np.inf]

	def __call__(self, mtt):
		def corePositions(numComponents, initCorePosition):
			sweepUp = range(numComponents-1)
			sweepDown = range(numComponents-1, 0, -1)
			sweep = chain(sweepUp, sweepDown)
			corePositions = cycle(sweep)
			for _ in zip(range(initCorePosition), corePositions): pass  # drop the first corePositions to start at `initCorePosition`
			return corePositions

		sweepLength = 2*(mtt.numComponents-1)
		for itr, corePos in enumerate(corePositions(mtt.numComponents, self.initCorePosition)):
			if itr % sweepLength == 0 and self.terminate(mtt):
				break
			yield corePos

	@property
	def storedError(self):
		try:
			return self.__storedError
		except AttributeError:
			stored_tt = xe.load_from_file(file_name)
			if stored_tt is None:
				self.__storedError = np.inf
				return self.__storedError
			stored_mtt = mT.RankOneMeasuredTTTensor(stored_tt, xs)
			# all_samples = np.arange(len(ys)).tolist()
			self.__storedError = mT.error(stored_mtt, ys, all_samples)
			return self.__storedError

	@storedError.setter
	def storedError(self, val):
		assert self.storedError >= val
		self.__storedError = val

	def terminate(self, mtt):
		sweep = len(self.errors)-1
		logger.info("Sweep: {0}".format(sweep))

		logger.todo("use cross-validation to ensure the test-set-error decreases?")
		# all_samples = np.arange(mtt.numMeasurements).tolist()
		# err = mT.error(mtt, ys, all_samples)
		err = self.error(mtt)
		if err <= self.errors[-1] and err <= self.storedError:
			logger.info("Update stored tensor. Old error: {0:.2f}%. New error: {1:.2f}%.".format(100*self.storedError, 100*err))
			if self.offset is None:
				xe.save_to_file(mtt.measuredTensor, file_name, xe.FileFormat.BINARY)
			else:
				xe.save_to_file(mtt.measuredTensor+self.offset.measuredTensor, file_name, xe.FileFormat.BINARY)
			self.storedError = err
		# loss = mT.loss(mtt, ys, all_samples)
		loss = self.loss(mtt)
		# fctr = -1/np.log(1-1/10)  # fctr * loss is an estimate for err
		# but this is irrelevant for relative decreases
		rel_dec = lambda old, new: (old-new)/new
		decrease = max(rel_dec(self.losses[-1], loss),
					   rel_dec(self.errors[-1], err))

		logger.info("Current error: {0:.2f}%".format(100*err))
		logger.info("Current loss: {0:.2e}".format(loss))
		logger.info("Relative decrease: {0:.2e} >= {1:.2e}".format(decrease, self.minRelDecrease))

		self.errors.append(err)
		self.losses.append(loss)
		sweep += 1
		return sweep > self.minSweeps and (sweep > self.maxSweeps or 0 <= decrease < self.minRelDecrease)
		# a negative decrease is an increase


def add_mode(mtt, terminator):
	logger.info("Add mode")
	result = mtt.measuredTensor
	logger.todo("measuredTensor.tensor_shape")
	logger.todo("measuredTensor.measure_shape")
	logger.todo("measuredTensor.measuredTensor = new_tensor")

	if mtt.numComponents < len(xs[0]):
		dim = xs[0][result.degree()].size
		ret = xe.TTTensor.random(result.dimensions + [dim], result.ranks() + [1])
		d = result.degree()
		for i in range(d):
			ret.set_component(i, result.get_component(i))
		ret.set_component(d, xe.Tensor.dirac([1,dim,1], [0,0,0]))
		ret.move_core(d, keepRank=True)
		mtt_new = mT.RankOneMeasuredTTTensor(ret, xs)

		all_samples = np.arange(len(ys)).tolist()
		mtt_loss = mT.loss(mtt, ys, all_samples)
		mtt_new_loss = mT.loss(mtt_new, ys, all_samples)
		assert (mtt_loss - mtt_new_loss)/mtt_loss < 1e-12, "{} != {}".format(mtt_loss, mtt_new_loss)
	else:
		logger.warning("Modes depleted.")
		mtt_new = mT.RankOneMeasuredTTTensor(result, xs)

	ScASD(mtt_new, ys, terminator)

	return mtt_new
	# if terminator.errors[-1] <= terminator.errors[1]:
	#     return mtt_new
	# else:
	#     return mtt


def add_rank(mtt, diff_terminator, terminator):
	logger.info("Add rank")

	result = mtt.measuredTensor
	#TODO: implement optimal reconstruction
	diff = xe.TTTensor.random(result.dimensions, [1]*(result.degree()-1))
	logger.info("Initializing update")
	mtt_diff = mT.RankOneMeasuredTTTensor(diff, xs)
	SsASD(mtt_diff, ys, diff_terminator, offset=mtt)
	diff = mtt_diff.measuredTensor
	logger.info("Perform retraction")
	logger.debug("Relative update norm: {0:.2e}".format(xe.frob_norm(diff)/xe.frob_norm(result)))
	# c = 0.1*xe.frob_norm(result)/xe.frob_norm(diff)
	# diff = c*diff
	mtt_new = mT.RankOneMeasuredTTTensor(result+diff, xs)
	ScASD(mtt_new, ys, terminator)

	return mtt_new
	# if terminator.errors[-1] <= terminator.errors[1]:
	#     return mtt_new
	# else:
	#     return mtt


# fMap_name = "HaarScattering/PCA80/BasisTransform:Poly3"
# fMap_name = "CompleteHaarScattering/PCA20/BasisTransform:Poly4"
# fMap_name = "HaarSplitting/PCA80/BasisTransform:Poly4"
fMap_name = "HaarSplitting/PCA20/BasisTransform:Poly5"
# fMap_name = "HaarSplitting/PCA20/Normalization/BasisTransform:Fourier5"
data = FMap.from_name("MNIST_train/{fMap}".format(fMap=fMap_name))
xs = [[xe.Tensor.identity([10,10])] + [xe.Tensor.from_ndarray(xi) for xi in x]  for x in data.features]
ys = np.array(data.labels).tolist()
# xs, ys = xs[20:], ys[20:]
all_samples = np.arange(len(ys)).tolist()

for x in xs:
	for x_cmp in x[1:]:
		assert x_cmp[0] == 1, x_cmp[0]  # needed in add_mode

file_name = "__cache__/MNIST_train/{fMap}/learning.xrs".format(fMap=fMap_name)

logger.debug("Loading parameters: {}".format(file_name))
result = xe.load_from_file(file_name)
if result is None:
	logger.debug("Could not load file: {}".format(file_name))
	logger.info("Creaing new tensor")
	result = xe.TTTensor.random([10], [])
	e = lambda i: np.eye(1,10,i)[0]
	OLS_estimator = np.mean([e(y) for y in ys], axis=0)
	result.set_component(0, xe.Tensor.from_ndarray(OLS_estimator[None,:,None]))
	mtt = mT.RankOneMeasuredTTTensor(result, xs)
	mtt = add_mode(mtt, custom_sweeper(0.05, 5, 40, initCorePosition=1))
else:
	mtt = mT.RankOneMeasuredTTTensor(result, xs)


local_dofs = lambda result: max(result.get_component(pos).size for pos in range(result.degree()))
global_dofs = lambda result: xe.TTTensor.degrees_of_freedom(result.dimensions, result.ranks())
def singular_values(tttensor):
	ret = []
	tttensor.move_core(tttensor.degree()-1)
	for pos in range(tttensor.degree()-1):
		tttensor.move_core(pos, keepRank=True)
		cmp = tttensor.get_component(pos).to_ndarray()
		cmp.shape = -1, cmp.shape[-1]
		u,s,v = np.linalg.svd(cmp)
		ret.append(s)
	tttensor.move_core(0, keepRank=True)
	assert len(ret) == len(tttensor.ranks())
	return ret
def lowest_singular_values(tttensor):
	return [s[-1] for s in singular_values(tttensor)]

def reduce_dofs(mtt, terminator_factory=None):
	result = mtt.measuredTensor
	gdofs = global_dofs(result)
	if gdofs*np.log(gdofs) <= 60000:
		return mtt

	logger.info("Truncating SVs")

	min_tt = xe.load_from_file(file_name)
	min_mtt = mT.RankOneMeasuredTTTensor(min_tt, xs)
	min_loss = mT.loss(min_mtt, ys, all_samples)

	logger.debug("Initial global DoFs: {0}".format(gdofs))
	logger.debug("Initial loss: {0:.2e}".format(mT.loss(mtt, ys, all_samples)))

	# round as long as the loss is lower than 1.1*min_loss
	# while loss < 1.1*min_loss:
	# while gdofs*np.log(gdofs) > 60000:
	while gdofs*np.log(gdofs) > 0.9*60000:  # give 10% tolerance since otherwise this if-clause will be triggered in every iteration (which seems to be too often)
		idx = np.argmin(lowest_singular_values(result))
		logger.debug("Truncation index: {}".format(idx))
		rks = result.ranks()
		rks[idx] -= 1
		if rks[idx] == 0: break
		result.round(rks)
		mtt = mT.RankOneMeasuredTTTensor(result, xs)

		ScASD(mtt, ys, custom_sweeper(0.0005, 30, 100))  #TODO: terminator_factory
		result = mtt.measuredTensor
		gdofs = global_dofs(result)

		logger.debug("Global DoFs: {0}".format(gdofs))
		logger.debug("Loss: {0:.2e}".format(mT.loss(mtt, ys, all_samples)))

	return mtt

def print_status(mtt, iter):
	res = mT.loss(mtt, ys, all_samples)
	err = mT.error(mtt, ys, all_samples)
	result = mtt.measuredTensor
	norm = xe.frob_norm(result)
	logger.info("="*80)
	logger.info("Iteration:      {}".format(iter))
	logger.info("Loss:           {0:.2e}".format(res))
	logger.info("Error:          {0:.2f}%".format(100*err))
	logger.info("Dimensions:     {}".format(result.dimensions))
	logger.info("Ranks:          {}".format(result.ranks()))
	lsvs = lowest_singular_values(result/xe.frob_norm(result))
	logger.info("Relevances:     [{}]".format(", ".join("%.2e"%sv for sv in lsvs)))
	logger.info("Local DoFs:     {}".format(local_dofs(result)))
	logger.info("Global DoFs:    {}".format(global_dofs(result)))
	logger.info("="*80)

iter = 1
while iter < 200:
	print_status(mtt, iter)

	#TODO: use block-dmrg (instead of block-als) --> more stable
	#TODO: choose a fixed cutoff by solving a convex optimization problem (minimize the schatten-1-norm with convexity constraints)
	mtt = reduce_dofs(mtt)

	mtt1 = add_mode(mtt, custom_sweeper(0.005, 10, 40, initCorePosition=result.degree()+1))
	mtt2 = add_rank(mtt, custom_sweeper(0.05, 5, 40, offset=mtt), custom_sweeper(0.005, 10, 40))
	err1 = mT.error(mtt1, ys, all_samples)
	err2 = mT.error(mtt2, ys, all_samples)
	if err1 < err2:
		logger.info("Adding mode to solution")
		mtt = mtt1
	else:
		logger.info("Increasing rank of solution")
		mtt = mtt2

	iter += 1
