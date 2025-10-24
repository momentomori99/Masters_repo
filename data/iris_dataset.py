from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import brian2 as b2


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)


r_min = 7 * b2.Hz
r_max = 50 * b2.Hz
# Convert normalized values to rates in Hz
r = r_min + X_normalized * (r_max - r_min)


# Optionally save normalized data if needed
np.save('iris_X_rates.npy', r)
np.save('iris_y.npy', y)
