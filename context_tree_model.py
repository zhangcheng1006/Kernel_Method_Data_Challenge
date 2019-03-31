"""This module implements the context-tree model according to this artical:
The context-tree kernel for strings (http://marcocuturi.net/Papers/cuturi05context.pdf)
"""
import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_function
from cvxopt import matrix, solvers
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def all_contexts(alphabet, D=3):
    """Generates all combinations of size D for the elements in alphabet.
    Parameters:
    -----------
        alphabet: a list of elements, [A, T, C, G] in our case
        D: integer, default 3. The size of context
    
    Returns:
    --------
        all_conts: a list of all context of size D of alphabet
    """
    assert D > 1
    all_conts = alphabet[::]
    for _ in range(D-1):
        all_conts_new = []
        for cont in all_conts:
            for l in alphabet:
                all_conts_new.append(l+cont)
        all_conts = all_conts_new
    return all_conts

def count_occurrences(sequences, CONTEXTS, ALPHABET, D=3):
    """Counts the occurences of contexts and (context, letter)
    Parameters:
    -----------
        sequences: a list of sequences of train
        CONTEXTS: a list of contexts of size D
        ALPHABET: a list of letters in the sequences
        D: integer, default 3, the length of context
    
    Return:
    -------
        counts_rho: array of size [num_sequences, num_contexts], the count of occurences of contexts in the sequences
        counts_theta: array of size [num_sequences, num_contexts, num_alphabet], the count of occurences of (context, letter)
    """
    counts_rho = np.zeros((len(sequences), len(CONTEXTS)), dtype=int)
    counts_theta = np.zeros((len(sequences), len(CONTEXTS), len(ALPHABET)), dtype=int)
    c2id = {c: idx for idx, c in enumerate(CONTEXTS)}
    l2id = {l: idx for idx, l in enumerate(ALPHABET)}
    for s_id, seq in enumerate(sequences):
        for i in range(len(seq)-D):
            c = seq[i:i+D]
            l = seq[i+D]
            c_id, l_id = c2id[c], l2id[l]
            counts_rho[s_id, c_id] += 1
            counts_theta[s_id, c_id, l_id] += 1
    return counts_rho, counts_theta

def compute_avg_weight(rho_x, rho_y, theta_x, theta_y):
    """Computes the average weights of occurences of contexts and (context, letter) on 2 sequences
    Parameters:
    -----------
        rho_x, rho_y: size (num_contexts, )
        theta_x, theta_y: size (num_contexts, d)

    Returns:
    --------
        avg_weight: size (num_contexts, d)
    """
    Nx = np.sum(rho_x)
    Ny = np.sum(rho_y)
    avg_weight = rho_x.reshape((-1, 1)) * theta_x / Nx + rho_y.reshape((-1, 1)) * theta_y / Ny
    return avg_weight

def compute_G_beta(alpha, beta):
    """Computes the G function in the article
    Parameters:
    -----------
        alpha: size (num_contexts, d)
        beta: size (d, )
    
    Return:
    -------
        G: size (num_contexts, )
    """
    alpha_beta = alpha + beta
    beta_sum = np.sum(beta)
    a_b_sum = np.sum(alpha_beta, axis=1)
    G = np.full((alpha.shape[0],), gamma_function(beta_sum)/gamma_function(a_b_sum))
    for i in range(len(beta)):
        G *= gamma_function(alpha_beta[:, i]) / gamma_function(beta[i])
    assert G.shape == (alpha.shape[0],)
    return G

def compute_K(gamma, betas, sigma, avg_weight):
    """Computes the K value for each node in the context tree
    Parameters:
    -----------
        gamma: (n, )
        betas: (n, d)
        sigma: float
        avg_weight: (num_contexts, d)
    Return:
    -------
        rslt: (num_contexts, )
    """
    betas = np.array(betas)
    rslt = np.zeros(avg_weight.shape[0], dtype=float)
    for gam, beta in zip(gamma, betas):
        alpha = sigma * avg_weight
        G = compute_G_beta(alpha, beta)
        rslt += gam * G
    non_occurred = np.where(np.sum(avg_weight, axis=1)==0)[0]
    rslt[non_occurred] = 1
    return rslt

def compute_kernel_value(rho_X, rho_Y, theta_X, theta_Y, epsilon, gamma, betas, sigma):
    """Computes the kernel value between two sequences
    Paramters:
    ----------
        rho_X, rho_Y: array of size (num_contexts, )
        theta_X, theta_Y: array of size (num_contexts, num_alphabet)
        epsilin, gamma, betas, sigma: hyperparameters
    Return:
    -------
        The kernel value between X and Y
    """
    d = theta_X.shape[-1]
    # size (num_contexts, d)
    avg_weight = compute_avg_weight(rho_X, rho_Y, theta_X, theta_Y)
    # size (num_contexts, )
    K = compute_K(gamma, betas, sigma, avg_weight)
    Gamma = K
    while len(Gamma) > 1:
        layer_size = len(Gamma)
        assert layer_size % d == 0
        avg_weight_new = np.zeros((layer_size//d, d), dtype=float)
        for i in range(layer_size//d):
            avg_weight_new[i, :] = np.sum(avg_weight[i*d:(i+1)*d, :], axis=0)
        K_new = compute_K(gamma, betas, sigma, avg_weight_new) # size (layer_size//d, )
        Gamma_new = (1 - epsilon) * K_new
        for i in range(layer_size//d):
            Gamma_new[i] += epsilon * np.prod(Gamma[i*d:(i+1)*d])
        zero_idx = avg_weight_new.sum(axis=1) == 0
        Gamma_new[zero_idx] = 1

        avg_weight = avg_weight_new
        Gamma = Gamma_new
    return Gamma[0]

def normalize_kernel_matrix(K):
    """Normalizes a Kernel Matrix
    """
    new_K = np.zeros(K.shape, dtype=float)
    n = K.shape[0]
    for i in range(n):
        new_K[i, i] = 1
        for j in range(i+1, n):
            new_K[i, j] = K[i, j] / np.sqrt(K[i, i] * K[j, j])
            new_K[j, i] = new_K[i, j]
    return new_K

def compute_kernel_matrix(train_X, counts_rho, counts_theta, CONTEXTS, ALPHABET, D=3):
    """Computes the kernel matrix of training dataset
    """
    n_train = len(train_X)
    K = np.zeros((n_train, n_train), dtype=float)
    for i in range(n_train):
        if i % 100 == 0:
            logging.info("train sample No.{}/{}".format(i+1, n_train))
        K[i, i] = compute_kernel_value(counts_rho[i], counts_rho[i], counts_theta[i], counts_theta[i], epsilon, gamma, betas, sigma)
        for j in range(i+1, n_train):
            K[i, j] = compute_kernel_value(counts_rho[i], counts_rho[j], counts_theta[i], counts_theta[j], epsilon, gamma, betas, sigma)
            K[j, i] = K[i, j]
    return K

def train_context_tree(K, train_Y, lamda):
    """Computes the SVM coefficients
    Parameters:
    -----------
        K: the normalized train Kernel matrix
        train_Y: the labels of training samples, with labels -1 or 1
        lamda: the regularization factor
    Return:
    -------
        supp_idx: the indices of support vectors
        supp_alpha: the coefficients of support vectors
    """
    n_train = K.shape[0]
    P = matrix(K, tc='d')
    q = matrix(-train_Y, tc='d')
    G = matrix(np.concatenate((np.diag(train_Y), np.diag(-train_Y)), axis=0), tc='d')
    h = matrix(np.concatenate((np.ones(n_train) / (2 * lamda * n_train), np.zeros(n_train))), tc='d')
    sol = solvers.qp(P, q, G, h)
    alpha = sol['x']

    supp_alpha = []
    supp_idx = []
    for i, ele in enumerate(alpha):
        if ele != 0:
            supp_idx.append(i)
            supp_alpha.append(ele)
    supp_alpha = np.array(supp_alpha).reshape((-1, 1))
    return supp_idx, supp_alpha

def predict_context_tree(test_X, K_train, supp_idx, supp_alpha, counts_rho_train, counts_theta_train, CONTEXTS, ALPHABET, epsilon, gamma, betas, sigma, D=3):
    """Predicts the labels of test samples
    Parameters:
    -----------
        test_X: the test sequences
        K_train: the unnormalized kernel matrix of training samples
        supp_idx, supp_alpha: support vector indices and coefficients
        counts_rho_train, counts_theta_train: counts of occurrence of contexts on training set
    Return:
    -------
        pred_test: the predicted labels of test samples
    """
    n_test = len(test_X)
    n_support = len(supp_idx)
    K_test = np.zeros((n_support, n_test), dtype=float)

    counts_rho_test, counts_theta_test = count_occurrences(test_X, CONTEXTS, ALPHABET, D=D)
    for j in range(n_test):
        if j % 100 == 0:
            logging.info("test sample No.{}/{}".format(j+1, n_test))
        self_kernel_value = compute_kernel_value(counts_rho_test[j], counts_rho_test[j], counts_theta_test[j], counts_theta_test[j], epsilon, gamma, betas, sigma)
        for i in range(n_support):
            supp_i = supp_idx[i]
            K_test[i, j] = compute_kernel_value(counts_rho_train[supp_i], counts_rho_test[j], counts_theta_train[supp_i], counts_theta_test[j], epsilon, gamma, betas, sigma)
            K_test[i, j] /= np.sqrt(self_kernel_value * K_train[supp_i, supp_i])

    pred_test = np.dot(K_test.T, supp_alpha) >= 0
    pred_test = pred_test.astype(int).flatten()
    pred_test[pred_test==0] = -1
    return pred_test

###############################################################################
############### the training and prediction script ############################
###############################################################################

# define the hyper parameters
# the hyperparameters of Context-Tree model is suggested by the articl of Cuturi
# tunning them may give better performance
ALPHABET = ['A', 'C', 'G', 'T']
D = 3
CONTEXTS = all_contexts(ALPHABET, D=D)

logging.info("All contexts: {}".format(CONTEXTS))

epsilon = 1. / len(ALPHABET)
num_dirichelet = 1
gamma = [1 / num_dirichelet for _ in range(num_dirichelet)]
betas = np.full((num_dirichelet, len(ALPHABET)), 0.5)
sigma = 2

# the regularization parameter to tune
lamdas = np.logspace(-7, -1, 7)

############## Cross Validation to tune the regularization parameter ##########
benchmark = pd.DataFrame(columns=['train 0', 'val 0', 'train 1', 'val 1', 'train 2', 'val 2'], index=lamdas)
benchmark.index.name = 'lambda'
for lamda_idx, lamda in enumerate(lamdas):
    logging.info("\n\nlamda = {}".format(lamda))
    for dataset_id in [0, 1, 2]:
        logging.info("Data set No.{}".format(dataset_id))
        train_X = pd.read_csv('./data/Xtr{}.csv'.format(dataset_id), header=0, index_col=0).values.flatten()
        train_Y = pd.read_csv('./data/Ytr{}.csv'.format(dataset_id), header=0, index_col=0)['Bound']
        train_Y[train_Y==0] = -1
        logging.info("Loaded {} train samples".format(len(train_X)))
        assert len(train_X) == len(train_Y)

        # train-validation-split
        n_samples = len(train_X)
        filter_idx = np.full((n_samples,), True)
        choose_idx = np.random.choice(range(n_samples), size=int(0.3*n_samples), replace=False)
        filter_idx[choose_idx] = False
        val_X = train_X[~filter_idx]
        val_Y = train_Y[~filter_idx]
        train_X = train_X[filter_idx]
        train_Y = train_Y[filter_idx]
        logging.info("Split train samples into {} validation samples and {} training samples".format(len(val_X), len(train_X)))
        
        # compute the train Kernel matrix
        counts_rho, counts_theta = count_occurrences(train_X, CONTEXTS, ALPHABET, D)
        K = compute_kernel_matrix(train_X, counts_rho, counts_theta, CONTEXTS, ALPHABET, D=D)  
        # normalize the matrix 
        K_norm = normalize_kernel_matrix(K)
        # solve the SVM optimization problem
        supp_idx, supp_alpha = train_context_tree(K_norm, train_Y, lamda)
        # compute the accuracy on training set
        K_train = K_norm[:, supp_idx]
        pred_train = np.dot(K_train, supp_alpha) >= 0
        pred_train = pred_train.astype(int).flatten()
        pred_train[pred_train==0] = -1
        train_accuracy = np.mean(pred_train == train_Y)
        logging.info("Obtain training accuracy: {}".format(train_accuracy))
        benchmark.iloc[lamda_idx, 2*dataset_id] = train_accuracy
        # predict the validation set labels
        logging.info("Computing on validation set")
        pred_val = predict_context_tree(val_X, K, supp_idx, supp_alpha, counts_rho, counts_theta, CONTEXTS, ALPHABET, epsilon, gamma, betas, sigma, D)
        # compute the accuracy on validation set
        val_accuracy = np.mean(pred_val == val_Y)
        logging.info("Obtain validation accuracy: {}".format(val_accuracy))
        benchmark.iloc[lamda_idx, 2*dataset_id+1] = val_accuracy

# benchmark.to_csv('benchmark.csv')

# choose the best lamda obtained by cross validation
best_lamda_index = benchmark.mean(axis=1).idxmax()
best_lamda = lamdas[int(best_lamda_index)]

logging.info("Choosing best lamda: {}".format(best_lamda))
lamda = best_lamda

############## Training on full training set and Prediction ###################
# a dataframe to store the predictions
result = pd.DataFrame(columns=['Bound'])

for dataset_id in [0, 1, 2]:
    logging.info("Data set No.{}".format(dataset_id))
    train_X = pd.read_csv('./data/Xtr{}.csv'.format(dataset_id), header=0, index_col=0).values.flatten()
    test_X = pd.read_csv('./data/Xte{}.csv'.format(dataset_id), header=0, index_col=0).values.flatten()
    train_Y = pd.read_csv('./data/Ytr{}.csv'.format(dataset_id), header=0, index_col=0)['Bound']
    train_Y[train_Y==0] = -1
    logging.info("Loaded {} train samples, {} test samples".format(len(train_X), len(test_X)))
    
    # compute the train kernel matrix
    logging.info("Computing kernel matrix on training set")
    counts_rho, counts_theta = count_occurrences(train_X, CONTEXTS, ALPHABET, D)
    K = compute_kernel_matrix(train_X, counts_rho, counts_theta, CONTEXTS, ALPHABET, D)
    K_norm = normalize_kernel_matrix(K)
    # solve the SVM optimization problem
    supp_idx, supp_alpha = train_context_tree(K_norm, train_Y, lamda)
    # predict on test dataset
    logging.info("Computing on test set")
    pred_test = predict_context_tree(test_X, K, supp_idx, supp_alpha, counts_rho, counts_theta, CONTEXTS, ALPHABET, epsilon, gamma, betas, sigma, D)
    pred_test = pd.DataFrame(pred_test, columns=['Bound'])
    pred_test.index.name = 'Id'
    # append predictions to result
    result = pd.concat((result, pred_test), axis=0, ignore_index=True)

# write to a local file Yte.csv
result.index.name = 'Id'
result.to_csv("Yte.csv")
