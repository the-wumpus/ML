# adpated from S Raschka ML book 2022
import numpy as np 


# follows perceptron except for the Fit function
class AdalineGD(object):

	def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self, X, y):
		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
		self.b_ = np.float_(0.)

		# try shape
		# self.w_ = np.zeros(1 + X.shape[1])
		self.cost_ = []


		for i in range(self.n_iter):
			output = self.net_input(X)
			errors = (y - output)
			
			self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
			self.b_ += self.eta * 2.0 * errors.mean()
			
			# try error 
			# self.w_[1:] += self.eta * X.T.dot(errors)
			# self.w_[0] += self.eta * errors.sum()
			cost = (errors**2).sum().mean()
			self.cost_.append(cost)

		return self

	def net_input(self, X):
		return np.dot(X, self.w_) + self.b_

	def activation(self, X):
		return X

	def predict(self, X):
		return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
		# try return
		# return np.where(self.activation(X) >= 0.0, 1, -1)