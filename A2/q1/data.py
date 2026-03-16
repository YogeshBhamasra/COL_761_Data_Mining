import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits

data = load_digits()  # Change to load_iris() or load_wine() for different datasets
X = data.data

np.save("digits.npy", X)

from sklearn.datasets import make_blobs
# import numpy as np

X, _ = make_blobs(
    n_samples=600,
    centers=5,
    n_features=5,
    cluster_std=0.2,
    random_state=42
)

np.save("blobs5.npy", X)