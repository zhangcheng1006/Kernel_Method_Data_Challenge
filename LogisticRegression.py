import numpy

def sigmoid(alpha):
    """Calculate the sigmoid of alpha.
    """
    return 1 / (1 + np.exp(-alpha))

def first_order_deriv(X, Y, sigma):
    """Calculate the first order derivation of the log-likelihood.
    """
    return np.dot((Y-sigma), X)

def second_order_deriv(X, sigma):
    """Calculate the second order derivation of the log-likelihood.
    """
    sigma_ = sigma.reshape((-1, 1))
    return -np.dot(X.T, X*sigma_*(1-sigma_))

def Logistic_train(X, Y, iter_max=1000, tol=1e-5):
    """Train the logistic regression model by Newton method.
    Parameters:
        iter_max : integer, default 1000
            the maximum number of iterations in Newton method
        tol : float, default 1e-5
            the criterion of convergence
    """
    n, d = X.shape
    X_tmp = np.hstack((X, np.ones((n, 1))))
    X_tmp_T = X_tmp.T
    # initialize w with zeros or random values
    w = np.zeros(d+1)
    # w = np.random.rand(d+1)
    sigma = sigmoid(np.dot(w, X_tmp_T))
    deriv1 = first_order_deriv(X_tmp, Y, sigma)
    deriv2 = second_order_deriv(X_tmp, sigma)
    w_new = w - np.dot(np.linalg.pinv(deriv2), deriv1)
    iter_count = 0
    while iter_count<iter_max and (np.abs(w-w_new)>tol).any():
        iter_count += 1
#        if iter_count % 1 == 0:
#            print("iteration {}: w = {}".format(iter_count, w_new))
        w = w_new.copy()
        sigma = sigmoid(np.dot(w, X_tmp_T))
        deriv1 = first_order_deriv(X_tmp, Y, sigma)
        deriv2 = second_order_deriv(X_tmp, sigma)
        w_new = w - np.dot(np.linalg.pinv(deriv2), deriv1)
    return w[:-1], w[-1]

def Logistic_predict(X, w, b):
    """Predict the labels of X. w and b are returned by Logistic_train().
    """
    pred = np.dot(X, w.T) + b >= 0
    pred = pred.astype(int)
    return pred

