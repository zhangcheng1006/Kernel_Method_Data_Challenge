import numpy as np
from kernels import RBF_kernel

def sigmoid(alpha):
    """Calculate the sigmoid of alpha.
    """
    return 1 / (1 + np.exp(-alpha))

def KLR_train(X, y, kernel=None, sigma=0.15, lamda=0, iter_max=1000, tol=1e-5):
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
    
    alpha_0 = np.random.rand(n, 1)
    m = np.dot(K, alpha_0)
    P = np.diagflat(- sigmoid(- m * y.reshape((-1, 1))))
    W = np.diagflat(sigmoid(m) * sigmoid(-m))
    z = m - np.linalg.solve(W, np.dot(P, y.reshape((-1, 1))))
    W_sqrt = W**0.5
    alpha_1 = np.dot(W_sqrt, np.linalg.solve(np.dot(W_sqrt, np.dot(K, W_sqrt)) + n * lamda * np.identity(n), np.dot(W_sqrt, z)))

    n_iter = 0
    while np.linalg.norm(alpha_1 - alpha_0) > tol and n_iter < iter_max:
        alpha_0 = alpha_1
        m = np.dot(K, alpha_0)
        P = np.diagflat(- sigmoid(- m * y.reshape((-1, 1))))
        W = np.diagflat(sigmoid(m) * sigmoid(-m))
        z = m - np.linalg.solve(W, np.dot(P, y.reshape((-1, 1))))
        W_sqrt = W**0.5
        alpha_1 = np.dot(W_sqrt, np.linalg.solve(np.dot(W_sqrt, np.dot(K, W_sqrt)) + n * lamda * np.identity(n), np.dot(W_sqrt, z)))

        n_iter += 1

    print(alpha_1)
    return alpha_1

def KLR_predict(X, X_train, alpha, kernel=None, sigma=0.15):
    n = X_train.shape[0]
    m = X.shape[0]
    K = np.zeros((n, m))
    if kernel is None:
        for i in range(n):
            for j in range(m):
                K[i, j] = np.dot(X_train[i].reshape((1, -1)), X[j].reshape((-1, 1)))

    else:
        for i in range(n):
            for j in range(m):
                K[i, j] = RBF_kernel(X_train[i], X[j], sigma)

    pred = np.dot(K.T, alpha) >= 0
    pred = pred.astype(int)
    # print(pred)
    return pred

