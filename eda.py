import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RANSACRegressor
# from mlxtend.plotting import scatterplotmatrix, heatmap


columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area','Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t',usecols=columns)
df = df.dropna(axis=0)
df.isnull().sum()

# translate the Central Air column to 0 and 1
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

# scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)
# plt.tight_layout()
# plt.show()

#cm = np.corrcoef(df.values.T)
#hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
#plt.tight_layout()
#plt.show()

ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=0.95, residual_threshold=None, random_state=123)
ransac.fit(X,y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3,10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', edgecolor='white', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='red', edgecolor='white', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Living area above ground in sqft')
plt.ylabel('Sale Price in USD')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()








