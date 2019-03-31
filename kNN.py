import numpy as np
from kernels import Spectrum_kernel

def kNN_predict(X, X_train, Y_train, kernel="spectrum", kmer=10, k=15):
    '''Implementation of k-nearest neighbors prediction using kernel.
    '''
    pred = np.zeros(len(X)).astype(int)
    if kernel == "spectrum":
        for i in range(len(X)):
            spectrum_similarity = []
            for j in range(len(X_train)):
                spectrum_similarity.append(Spectrum_kernel(X[i], X_train[j], kmer))
            k_nearest_neighbors = Y_train[np.argpartition(spectrum_similarity, k)[0:k]]
            if sum(k_nearest_neighbors) > k // 2:
                pred[i] = 1
            else:
                pred[i] = 0
    return pred