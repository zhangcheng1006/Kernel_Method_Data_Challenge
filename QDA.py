import numpy as np

def QDA_train(X, Y):
    """Calculate estimators of \pi, \mu_0, \mu_1, \Sigma_0 and \Sigma_1 of QDA
    """
    n = len(Y)
    n1 = sum(Y)
    n0 = n - n1
    pie = Y.sum(axis=0) / n
    mu0 = X[Y==0, :].mean(axis=0)
    mu1 = X[Y==1, :].mean(axis=0) 
    X_tmp = X.copy()
    X_tmp[Y==0, :] -= mu0
    X_tmp[Y==1, :] -= mu1
    sigma0 = np.dot(X_tmp[Y==0, :].T, X_tmp[Y==0, :]) / n0
    sigma1 = np.dot(X_tmp[Y==1, :].T, X_tmp[Y==1, :]) / n1
    return pie, mu0, mu1, sigma0, sigma1

def QDA_predict(X, pie, mu0, mu1, sigma0, sigma1):
    """Predict the labels of X. pie, mu0, mu1, sigma0 and sigma1 are 
    returned by QDA_train().
    """
    X1 = X - mu1
    X0 = X - mu0
    # probability of label 1
    prob1 = pie * (1 / np.sqrt(np.linalg.det(sigma1))) * \
            np.exp(- (X1.T * np.dot(np.linalg.inv(sigma1), 
                                    X1.T)).sum(axis=0) / 2)
    # probability of label 0
    prob0 = (1-pie) * (1 / np.sqrt(np.linalg.det(sigma0))) * \
            np.exp(- (X0.T * np.dot(np.linalg.inv(sigma0), 
                                    X0.T)).sum(axis=0) / 2)
    # predicting by comparing prob1 and prob0
    pred = prob1 >= prob0
    pred = pred.astype(int)
    return pred

