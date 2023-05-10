from sklearn.datasets import make_classification
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate dataset with 6 features and 2 classes
X, y = make_classification(n_samples=500, n_features=6, n_informative=6,
                           n_redundant=0, n_clusters_per_class=1, class_sep=0.8)

# Set 4 features to either 0 or 1
# X[:, :4] = np.random.randint(2, size=(1000, 4))
X[:, :4] = np.random.Generator.integers(1, size=500)
# Set 5th feature to an integer between 5 and 44
X[:, 4] = np.random.Generator.integers(5, high=45, size=500)

# Set 6th feature to a real number between 98.4 and 102.1
X[:, 5] = np.random.uniform(98.4, 102.1, size=500)

# Create a pandas DataFrame from the dataset
df = pd.DataFrame(X)
df['Class'] = y

# Export the DataFrame to a CSV file
df.to_csv('test.csv', index=False)