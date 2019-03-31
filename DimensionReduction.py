import numpy as np
from kernels import *

def PCA(X, dim=50):
    '''Implementation of the standard PCA
    '''
    mean = np.mean(X, axis=0)
    centered_X = X - mean.reshape((1, -1))
    cov_mat = np.dot(centered_X.T, centered_X)

    eig_values, eig_vectors = np.linalg.eigh(cov_mat)
    eig_values = eig_values.reshape((-1, 1))
    
    ordered_indexes = np.argsort(np.linalg.norm(eig_values, axis=1))[::-1]
    total = np.sum(np.linalg.norm(eig_values, axis=1))
    principal = 0
    for i, idx in enumerate(ordered_indexes):
        principal += np.linalg.norm(eig_values[idx])
        if principal / total >= 0.85:
            dim = i+1
            break
    print(dim)
    
    ordered_eig_vectors = eig_vectors[:, ordered_indexes]

    transformed = np.dot(centered_X, ordered_eig_vectors[:, 0:dim])
    return transformed, mean, ordered_eig_vectors[:, 0:dim]


def kernelPCA(X, kernel='RBF', sigma=1.0, k=12, d=2, m=5):
    '''Implementation of Kernel PCA
    '''
    n = X.shape[0]
    K = np.zeros((n, n))
    if kernel == 'RBF':
        for i in range(n):
            for j in range(n):
                K[i, j] = RBF_kernel(X[i], X[j], sigma)
    elif kernel == 'spectrum':
        for i in range(n):
            for j in range(n):
                K[i, j] = Spectrum_kernel(X[i], X[j], k)
    elif kernel == 'poly':
        for i in range(n):
            for j in range(n):
                K[i, j] = poly_kernel(X[i], X[j], d)
        
    ones = (1./n) * np.ones((n, n))
    cov_mat = K - ones.dot(K) - K.dot(ones) + ones.dot(K).dot(ones)

    eig_values, eig_vectors = np.linalg.eigh(cov_mat)
    eig_values = eig_values.reshape((-1, 1))
    
    ordered_indexes = np.argsort(np.linalg.norm(eig_values, axis=1))[::-1]
    total = np.sum(np.linalg.norm(eig_values, axis=1))
    principal = 0
    for i, idx in enumerate(ordered_indexes):
        principal += np.linalg.norm(eig_values[idx])
        if principal / total >= 0.85:
            dim = i+1
            break
    print(dim)
    
    ordered_eig_values = eig_values[ordered_indexes]
    print(ordered_eig_values.shape)
    ordered_eig_vectors = eig_vectors[:, ordered_indexes]
    alpha = ordered_eig_vectors[:, 0:dim] / np.sqrt(np.linalg.norm(ordered_eig_values[0:dim], axis=1))
    transformed = np.dot(cov_mat, alpha)
    return transformed
