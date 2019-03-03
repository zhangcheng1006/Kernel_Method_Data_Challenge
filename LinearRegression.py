import numpy as np

def Linear_train(X, Y):
    """Calculate w, b and sigma of linear regression.
    w = (X'X)^{-1}X'Y
    sigma = ||Y-(Xw+b)||/n
    """
    n = len(Y)
    X_tmp = np.hstack((X, np.ones((n, 1))))
    X_tmp_T = X_tmp.T
    w = np.dot(np.linalg.inv(np.dot(X_tmp_T, X_tmp)), np.dot(X_tmp_T, Y.T))
    sigma = np.mean((Y-np.dot(X_tmp, w.T))**2, axis=0)

    return w[:-1], w[-1], sigma

def Linear_predict(X, w, b):
    """Predict the labels of X. w and b are returned by Linear_train().
    """
    pred = np.dot(X, w.T) + b >= 0.5
    pred = pred.astype(int)
    return pred
