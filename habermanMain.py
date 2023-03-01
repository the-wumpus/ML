import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from perceptron import Perceptron
from adalineGD import AdalineGD
from adalineSGD import AdalineSGD
import snnPlot


# get the habermans data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data',header = None)

# calculate dataset size [rows,columns]
rows = len(df.axes[0])
columns = len(df.axes[1])

# rename columns for clarity
df.columns = ['Age', 'Year Surgery', 'Axillary Nodes', 'Class']

# check-expect 306 x 3 matrix
print("Number of Rows: " + str(rows))
print("Number of Columns: " + str(columns))

# Plot 306 samples of the data
y = df.iloc[0:, [0, 3]]

y = df.iloc[0:306, 3].values
y = np.where(y == 1 , -1 , 1)

# age and axillary nodes
X = df.iloc[0:, [0, 2]].values

plt.scatter(X[:225, 0], X[:225, 1], color = 'r', marker = 'o', label = 'Survived')
plt.scatter(X[225:306, 0], X[225:306, 1], color = 'b', marker = 'x', label = 'Deceased')
plt.xlabel('Age')
plt.ylabel('Axillary Nodes')
plt.legend(loc = 'upper left')
plt.show()

# get the perceptron model
model1 = Perceptron(eta = 0.1, n_iter = 10)

# train the model
model1.fit(X, y)

# plot the Perceptron training error
plt.plot(range(1, len(model1.errors_) + 1), model1.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.show()

# Plot decision regions
snnPlot.plot_decision_regions(X, y, classifier = model1)
plt.title('Perceptron')
plt.xlabel('Age')
plt.ylabel('Axillary nodes')
plt.legend(loc = 'upper left')
plt.show()


# Plot Adaline Data 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# get the AdalineGD model
# Learning rate eta (between 0.0 and 1.0)
model2 = AdalineGD(n_iter = 20, eta = 0.5)

# Train the model
model2.fit(X_std, y)

# Plot the decision boundary
snnPlot.plot_decision_regions(X_std, y, classifier = model2)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Age')
plt.ylabel('Axillary nodes')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(model2.cost_) + 1), model2.cost_, marker = 'o', color = 'r')
plt.xlabel('Batch')
plt.ylabel('Mean squared error')
plt.show()

# get the AdalineSGD model
# Learning rate eta (between 0.0 and 1.0)
model3 = AdalineSGD(n_iter = 15, eta = 0.01, random_state = 1)

# Train the model
model3.fit(X_std, y)

# Plot the decision boundary
# snnPlot.plot_decision_regions(X_std, y, classifier = model3)
snnPlot.plot_decision_regions(X, y, classifier = model3)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Age')
plt.ylabel('Axillary nodes')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(model3.cost_) + 1), model3.cost_, marker = 'x', color = 'b')
plt.title('Adaline - StochasticGradient Descent')
plt.xlabel('Batch')
plt.ylabel('Average loss')
plt.show()