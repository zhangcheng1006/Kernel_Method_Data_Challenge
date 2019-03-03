import numpy as np

def RBF_kernel(x, y, sigma=1.0):
    return np.exp(- np.linalg.norm(x - y)**2 / (2 * sigma**2))