import numpy as np
from cvxopt import matrix, solvers
from kernels import RBF_kernel

def SVM_train(X, y, kernel=None, sigma=0.15, lamda=1.0):
    n, d = X.shape
    K = np.zeros((n, n))
    if kernel is None:
        for i in range(n):
            for j in range(n):
                K[i, j] = np.dot(X[i].reshape((1, -1)), X[j].reshape((-1, 1)))
    else:
        for i in range(n):
            for j in range(n):
                K[i, j] = RBF_kernel(X[i], X[j], sigma)

    P = matrix(2 * K, tc='d')
    q = matrix(- 2 * y, tc='d')
    G = matrix(np.concatenate((np.diag(y), np.diag(-y)), axis=0), tc='d')
    h = matrix(np.concatenate((np.ones(n) / (2 * lamda * n), np.zeros(n))), tc='d')

    sol = solvers.qp(P, q, G, h)
    # print(sol['x'])
    return sol['x']

def SVM_predict(X, X_train, alpha, kernel=None, sigma=0.15):
    real_alpha = []
    indexes = []
    for i, ele in enumerate(alpha):
        if ele != 0:
            indexes.append(i)
            real_alpha.append(ele)
    
    n = len(real_alpha)
    m = X.shape[0]
    K = np.zeros((n, m))
    if kernel is None:
        for i in range(n):
            for j in range(m):
                idx = indexes[i]
                K[i, j] = np.dot(X_train[idx].reshape((1, -1)), X[j].reshape((-1, 1)))

    else:
        for i in range(n):
            for j in range(m):
                idx = indexes[i]
                K[i, j] = RBF_kernel(X_train[idx], X[j], sigma)

    pred = np.dot(K.T, np.array(real_alpha).reshape((-1, 1))) >= 0
    pred = pred.astype(int)
    print(pred)
    return pred
