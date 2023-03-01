# adpated from S Raschka ML book 2022
# follows adalineGD except for the Fit function

import numpy as np 
from numpy.random import seed

class AdalineSGD(object):

	def __init__(self, eta = 0.01, n_iter = 10, shuffle= True, 
	random_state = None):

		self.eta = eta
		self.n_iter = n_iter
		self.w_initialization = False
		self.shuffle = shuffle
		self.random_state = random_state

	def fit(self, X, y):

		self._initialize_weights(X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):

			if self.shuffle:
				X, y = self._shuffle(X, y)

			cost = []
			for xi, target in zip(X, y):
				cost.append(self._update_weights(xi, target))
			avg_cost = np.mean(cost)

			# try avg_cost precision error
			#avg_cost = sum(cost) / len(y)

			self.cost_.append(avg_cost)

		return self

	def partial_fit(self, X, y):

		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		if y.ravel().shape[0] > 1:
			for xi, target in zip(X, y):
				self._update_weights(xi, target)
		else:
			self._update_weights(X, y)

		return self

	def _shuffle(self, X, y):

		# try random seed function
		# r = self.rgen.permutation(len(y))
		r = np.random.permutation(len(y))
		return X[r], y[r]

	def _initialize_weights(self, m):

		self.rgen = np.random.RandomState(self.random_state)
		self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
		self.b_ = np.float_(0.)
		self.w_initialized = True

		# original test
		# self.w_ = np.zeros(1 + m)
		# self.w_initialized = True

	def _update_weights(self, xi, target):

		output = self.net_input(xi)
		error = (target - output)
		self.w_ += self.eta * 2.0 * xi * (error)
		self.b_ += self.eta * 2.0 * error
		cost = error**2
		# check-expect weight change 
		# self.w_[1:] += self.eta * xi.dot(error)
		# self.w_[0] += self.eta * error
		# cost = 0.5 * (error ** 2)
		return cost

	def net_input(self, X):
		return np.dot(X, self.w_) + self.b_

		# check-expect array values
		# return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		return self.net_input(X)

	def predict(self, X):
		return np.where(self.activation(X) >= 0.5, 1, 0)

