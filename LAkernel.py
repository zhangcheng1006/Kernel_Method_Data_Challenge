'''Implementation of Local Alignment kernel
'''

import numpy as np

letter2id = {'A':0, 'T':1, 'G':2, 'C':3}

S = (np.array([[-1, -3, -3, -3],
              [-3, -1, -3, -3],
              [-3, -3, -1, -3],
              [-3, -3, -3, -3]]))

def DPcompute(x, y, beta, e, d):
    n = len(x)
    m = len(y)
    M = np.zeros((n+1, m+1))
    X = np.zeros((n+1, m+1))
    Y = np.zeros((n+1, m+1))
    X2 = np.zeros((n+1, m+1))
    Y2 = np.zeros((n+1, m+1))

    for i in range(1, n+1):
        for j in range(1, m+1):
            s = S[letter2id[x[i-1]], letter2id[y[j-1]]]
            M[i, j] = np.exp(beta * s) * (1 + X[i-1, j-1] + Y[i-1, j-1] + M[i-1, j-1])
            X[i, j] = np.exp(beta * d) * M[i-1, j] + np.exp(beta * e) * X[i-1, j]
            Y[i, j] = np.exp(beta * d) * (M[i, j-1] + X[i, j-1]) + np.exp(beta * e) * Y[i, j-1]
            X2[i, j] = M[i-1, j] + X2[i-1, j]
            Y2[i, j] = M[i, j-1] + X2[i, j-1] + Y2[i, j-1]
            
    return X2[n, m], Y2[n, m], M[n, m]

def LAkernel(x, y, beta=0.2, e=11, d=1):
    x2, y2, m = DPcompute(x, y, beta, e, d)
    k = 1 + x2 + y2 + m
    log_k = np.log(k) / beta
    return log_k

def spectral_translation(K):
    n = K.shape[0]
    eigvalues, eigvectors = np.linalg.eig(K)
    smallest = np.min(eigvalues)
    if smallest < 0:
        K = K - smallest * np.identity(n)
    return K

def empirical_kernel_map(K):
    return np.dot(K.T, K)


