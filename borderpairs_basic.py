import numpy as np
from sklearn.linear_model import Perceptron

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class BorderPairsMethodClassifier(BaseEstimator):
	def __init__(self) -> None:
		super().__init__()
		self.bp_model = None

	def get_params(self, deep=True):
		return super().get_params(deep)

	def set_params(self, **params):
		return super().set_params(**params)

	def fit(self, X: np.ndarray, y: np.ndarray):
		'''
		'''
		self._validate_data(X, y)

		categories = np.unique(y)
		if categories.shape[0] != 2:
			pass

		pts_1 = X[y == 1, :]
		pts_2 = X[y == 0, :]

		self.bp_model = _bp_multi_layer(pts_1, pts_2)

	def predict(self, X):
		'''

		'''
		if self.bp_model is None:
			raise NotFittedError("BorderPairsMethod object not yet fitted")

		y = _bp_predict(X, self.bp_model)
		return y


def _find_border_pairs(x_1, x_2):
	'''
	x_1 - np.ndarray, NxF
	x_2 - np.ndarray, MxF
	
	Find border pairs - data points from x_1 and x_2 that are on the
	border between their respective regions.

	returns:
		bp_ind - np.ndarray, Px2
		
		Array of P pairs of indices of border pairs, first column index 
		for x_1 second column index for x_2. For example, we get
		the fourth border pair f_1 and f_2 as:
		f_1 = x_1[bp_ind[3, 0], :] 
		f_2 = x_2[bp_ind[3, 1], :] 
	'''
	N, F = x_1.shape
	M, _ = x_2.shape

	border_pairs = []

	# calculate all the distances between points of:
	# class 1 and 2
	distances_12 = ((x_1.reshape(N, 1, F)-x_2.reshape(1, M, F))**2).sum(2)**0.5
	# class 1 and 1
	distances_11 = ((x_1.reshape(N, 1, F)-x_1.reshape(1, N, F))**2).sum(2)**0.5
	# class 2 and 2
	distances_22 = ((x_2.reshape(M, 1, F)-x_2.reshape(1, M, F))**2).sum(2)**0.5

	for n in range(N):
		for m in range(M):
			# select point n from class 1 and point m from class 2
			d = distances_12[n, m]
			neighbors = True
			# check if any other point of class 1 that is closer to n
			# is also closer to point m
			for k in range(N):
				if k == n:
					continue
				if (distances_11[n, k] < d) and (distances_12[k, m] < d):
					neighbors = False
					break
			if not neighbors:
				continue
			# check the reverse for points of class 2
			for k in range(M):
				if k == m:
					continue
				if (distances_12[n, k] < d) and (distances_22[m, k] < d):
					neighbors = False
					break

			# if they are clossest, add them to list of border pairs
			if neighbors:
				border_pairs.append((n, m))

	return np.array(border_pairs)


def _group_border_pairs(x_1: np.ndarray, x_2: np.ndarray, bp_list: np.ndarray, skip_grouping: bool = False) -> list[Perceptron]:
	'''
	x_1 - np.ndarray, NxF
		Data samples of class 1.
	x_2 - np.ndarray, MxF
		Data samples of class 2.
	bp_list - np.ndarray, Bx2
		List of B border pair indices.
	skip_grouping - bool
		If set to true, each border pair gets it's own linear divider.

	Find a linear divider for each border and attempt to fit such a 
	divider to multiple border pairs. Border pairs are thus grouped 
	together by common linear dividers.
	
	returns:
		dividers - list of linear dividers, sklearn.linear_model.Perceptron objects
	'''
	P = bp_list.shape[0]
	pair_processed = np.zeros(P, bool)
	perceptrons = []
	# check each border pair
	for p in range(P):
		if pair_processed[p]:
			continue
		pair_processed[p] = True

		n, m = bp_list[p]
		pm = [x_1[n, :], x_2[m, :]]
		tm = [1, 0]
		# train a linear clasifier for the selected pair
		p_model = Perceptron(warm_start=True, tol=None)

		# lets manually fit the divisor between two points
		w = x_1[n, :]-x_2[m, :]
		w /= (w**2).sum()**0.5
		w.shape = 1, w.shape[0]
		b = -np.array((w@x_1[n, :]+w@x_2[m, :])/2)
		p_model.fit(pm, tm, coef_init=w, intercept_init=b)

		perceptrons.append(p_model)
		if skip_grouping:
			continue

		# check all other border pairs if any can also be properly separated by this clasifier
		for r in range(p+1, P):
			if pair_processed[r]:
				continue

			# add the pair to training data
			i, j = bp_list[r]
			pm += [x_1[i, :], x_2[j, :]]
			tm += [1, 0]
			# remember the current model parameters
			coef_prev, intercept_prev = p_model.coef_.copy(), p_model.intercept_.copy()
			# retrain the classifier
			p_model.fit(pm, tm)
			# check if training was successful
			if p_model.score(pm, tm) == 1.0:
				pair_processed[r] = True
			else:
				# if training failed
				# reset the model parameters
				# and remove the pair from the training data
				p_model.coef_ = coef_prev
				p_model.intercept_ = intercept_prev
				pm = pm[:-2]
				tm = tm[:-2]

	return perceptrons


def _homogenize(x_1: np.ndarray, x_2: np.ndarray, perceptrons: list[Perceptron]) -> list[Perceptron]:
	'''
	x_1 - np.ndarray, NxF
		Data samples of class 1.
	x_2 - np.ndarray, MxF
		Data samples of class 2.
	perceptrons - list[sklearn.linearmodel.Perceptron]
		List of linear dividers.
		
	Check if linear dividers create homogenious regions of the two classes.
	If not, add identify additional border pairs and dividers to create homogenous regions.

	returns:
		dividers - list of linear dividers, sklearn.linear_model.Perceptron objects. Together
			these provide a completely homogenous division of samples into the two classes.
	
	'''
	while True:
		N = x_1.shape[0]
		M = x_2.shape[0]
		F = len(perceptrons)
		# get encodings for data
		x_1_encoding = np.array([p.predict(x_1) for p in perceptrons]).T
		x_2_encoding = np.array([p.predict(x_2) for p in perceptrons]).T

		# find a matching encodings of samples in different classes
		matching = np.all(
                    x_1_encoding.reshape(
                    	N, 1, F) == x_2_encoding.reshape(1, M, F),
                    axis=2)
		i_1, i_2 = np.where(matching)

		# if there were no matching encodings, the division is homogenous
		if i_1.shape[0] == 0:
			break

		# find all the samples with this encoding
		encoding = x_1_encoding[i_1[0], :]
		x_1_sub = x_1[np.all(x_1_encoding == encoding, axis=1), :]
		x_2_sub = x_2[np.all(x_2_encoding == encoding, axis=1), :]

		# for the samples with the same encoding, find borders pairs
		# and their divisors
		bp_sub = _find_border_pairs(x_1_sub, x_2_sub)
		perceptrons_sub = _group_border_pairs(x_1_sub, x_2_sub, bp_sub)
		perceptrons += perceptrons_sub

	return perceptrons


def _bp_multi_layer(pts_1: np.ndarray, pts_2: np.ndarray) -> list[list[Perceptron]]:
	'''
	pts_1 - np.ndarray, NxF
		Data samples of class 1.
	pts_2 - np.ndarray, MxF
		Data samples of class 2.
		
	Use the border pairs method to recursively build up a multilayer perceptron
	that will separate samples of x_1 and x_2.
	
	returns:
		dividers - list of lists of linear dividers, sklearn.linear_model.Perceptron objects.
					These comprise the layers/levels of a MLP.
	
	'''
	# to start we use the features as sample encodings on the first layer
	encoding_1 = pts_1
	encoding_2 = pts_2
	layers = []
	while True:
		# find a homogenous division on the n-th layer
		border_pairs = _find_border_pairs(encoding_1, encoding_2)
		perceptrons = _group_border_pairs(
			encoding_1, encoding_2, border_pairs, skip_grouping=False)
		perceptrons = _homogenize(encoding_1, encoding_2, perceptrons)

		layers.append(perceptrons)
		# if this layer required only 1 perceptron, the MLP is complete
		if len(perceptrons) == 1:
			break

		# calculate the encodings of the training data for this level
		encoding_1 = np.array([p.predict(encoding_1)
		                      for p in perceptrons], dtype=np.float32).T
		encoding_2 = np.array([p.predict(encoding_2)
		                      for p in perceptrons], dtype=np.float32).T

		# reduce to samples with unique encodings
		encoding_1 = np.unique(encoding_1, axis=0)
		encoding_2 = np.unique(encoding_2, axis=0)

	return layers


def _bp_predict(pts: np.ndarray, bp_layers: list[list[Perceptron]]) -> np.ndarray:
	'''
	pts - np.ndarray, NxF
		Data samples of class 1.
	bp_layers - list of lists of linear dividers, sklearn.linear_model.Perceptron objects.
		
	Use the border pairs method to recursively build up a multilayer perceptron
	that will separate samples of x_1 and x_2.
	
	returns:
		encoding - np.ndarray, Nx1
	'''
	encoding = pts
	for l in bp_layers:
		encoding = np.array([p.predict(encoding) for p in l]).T
	return encoding
