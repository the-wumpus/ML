import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data
y = iris.target

class AdalineSGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def activation(self, X):
        return X
    
    def net_input(self, X):
        return np.dot(X, self.weights_[1:]) + self.weights_[0]
    def fit(self, X, y):
        """Fit training data."""

        self.weights_ = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=X.shape[1]+1)
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)

            if errors.ndim == 1:
                self.weights_[1:] += self.eta * X.T.dot(errors)
                self.weights_[0] += self.eta * errors.sum()
            else:
                for j in range(y.shape[1]):
                    errors = (y[:, j] - output[:, j])
                    self.weights_[1:, j] += self.eta * X.T.dot(errors)
                    self.weights_[0, j] += self.eta * errors.sum()

            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self


for i in range(3):
    y_binary = np.where(y == i, 1, -1)
    adalinesgd = AdalineSGD(n_iter=10)
    adalinesgd.fit(X, y_binary)
    plt.plot(range(1, len(adalinesgd.cost_) + 1), adalinesgd.cost_, marker='o')

plt.title('AdalineSGD OVR on Iris dataset')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

