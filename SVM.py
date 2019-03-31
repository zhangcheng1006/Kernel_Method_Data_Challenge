'''Implementation of SVM algorithm.
'''

import numpy as np
from cvxopt import matrix, solvers
from kernels import *
import os
import pandas as pd
from LAkernel import *

def compute_kernel_matrix(X_train, X_test, kernel=None, d=2, k=13, m=5, p=13, mu=0.3, sigma=0.15, lamda=1.0):
    '''Compute the kernel matrix for training and testing according to different kernels given as argument.
    '''
    n = len(X_train)
    m = len(X_test)
    K_train = np.zeros((n, n))
    K_test = np.zeros((n, m))
    if kernel is None:
        for i in range(n):
            for j in range(n):
                K_train[i, j] = np.dot(X_train[i].reshape((1, -1)), X_train[j].reshape((-1, 1)))
            for j in range(m):
                K_test[i, j] = np.dot(X_train[i].reshape((1, -1)), X_test[j].reshape((-1, 1)))
    elif kernel == 'RBF':
        for i in range(n):
            for j in range(n):
                K_train[i, j] = RBF_kernel(X_train[i], X_train[j], sigma)
            for j in range(m):
                K_test[i, j] = RBF_kernel(X_train[i], X_test[j], sigma)
    elif kernel == 'spectrum':
        for i in range(n):
            for j in range(n):
                K_train[i, j] = Spectrum_kernel(X_train[i], X_train[j], k)
            for j in range(m):
                K_test[i, j] = Spectrum_kernel(X_train[i], X_test[j], k)
    elif kernel == 'poly':
        for i in range(n):
            for j in range(n):
                K_train[i, j] = poly_kernel(X_train[i], X_train[j], d)
            for j in range(m):
                K_test[i, j] = poly_kernel(X_train[i], X_test[j], d)
    elif kernel == 'sub':
        for i in range(n):
            for j in range(n):
                K_train[i, j] = substring_kernel(X_train[i], X_train[j], k, m)
            for j in range(m):
                K_test[i, j] = substring_kernel(X_train[i], X_test[j], k, m)
    elif kernel == 'gap':
        for i in range(n):
            for j in range(n):
                K_train[i, j] = gapped_kernel(X_train[i], X_train[j], k)
            for j in range(m):
                K_test[i, j] = gapped_kernel(X_train[i], X_test[j], k)
    elif kernel == 'di_mismatch':
        all_kmers = get_all_kmers(X_train, k)
        print(len(all_kmers))
        features = di_mismatch(all_kmers, X_train, k, m)
        test_features = di_mismatch(all_kmers, X_test, k, m)
        K_train = np.dot(features, features.T)
        K_test = np.dot(features, test_features.T)
    elif kernel == 'LA':
        for i in range(n):
            print(i)
            K_train[i, i] = LAkernel(X_train[i], X_train[i])
            for j in range(i+1, n):
                K_train[i, j] = LAkernel(X_train[i], X_train[j])
                K_train[j, i] = K_train[i, j]
            for j in range(m):
                K_test[i, j] = LAkernel(X_train[i], X_test[j])

    return K_train, K_test


def SVM_train(K, y, lamda=0.01):
    '''Train the SVM classifier given a kernel matrix and labels
    '''
    n = K.shape[0]
    print(K[0:10, 0:10])
    P = matrix(2 * K, tc='d')
    y = y.astype(np.double)
    q = matrix(- 2 * y, tc='d')
    G = matrix(np.concatenate((np.diagflat(y), np.diagflat(-y)), axis=0), tc='d')
    h = matrix(np.concatenate((np.ones(n) / (2 * lamda * n), np.zeros(n))), tc='d')

    sol = solvers.qp(P, q, G, h)
    print(sol['x'][0:10])
    return sol['x']



def SVM_predict(K, alpha):
    '''Prediction by SVM classifier
    '''
    pred = np.dot(K.T, np.array(alpha).reshape((-1, 1))) >= 0
    pred = pred.astype(int)
    return pred
