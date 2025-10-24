import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Load output and coerce into a (n_samples, 4000) array
output = np.load('data/X_output.npy', allow_pickle=True)
y = np.load('data/iris_y.npy', allow_pickle=True)
X = np.array(output)
# Shuffle all values in X globally




from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# --- Standardize, then PCA, then logistic regression ---
scaler = StandardScaler()
pca    = PCA(n_components=50)  # choose number of PCs empirically
clf    = LogisticRegression(max_iter=500)

model = make_pipeline(scaler, pca, clf)

scores = cross_val_score(model, X, y, cv=5)
print("Cross-val accuracy:", scores.mean())



