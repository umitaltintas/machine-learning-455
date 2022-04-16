# Write the function pca(X) that takes an 𝑛 × 𝑛 matrix and returns mean, weights and vectors.
# The mean is the mean of the columns of X. The principle components of X are in vectors. The
# corresponding eigenvalues are in weights. You should use only a function performing SVD and
# nothing else from any Python libraries.



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler



def pca(X):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    S = np.dot(X_centered.T, X_centered)
    U, S, V = np.linalg.svd(S)
    return mean, S, V


# load mnist data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Using only a portion of the data (e.g., about 1000 imagesrandomly chosen from the training set) perform PCA and train a classifier.
#Using the MNIST data, do a series of PCA-based reductions on the data.

standardized_data = StandardScaler().fit_transform(x_train)
print(standardized_data.shape)


def pca_reduction(X, n_components):
    mean, S, V = pca(X)
    X_reduced = np.dot(X - mean, V[:n_components])
    return X_reduced
