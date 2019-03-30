from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_function
from cvxopt import matrix, solvers
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def all_contexts(alphabet, D=3):
    assert D > 1
    all_conts = alphabet[::]
    for d in range(D-1):
        all_conts_new = []
        for cont in all_conts:
            for l in alphabet:
                all_conts_new.append(l+cont)
        all_conts = all_conts_new
    return all_conts

# print(all_contexts(['0', '1']))

def count_occurrences(sequences, CONTEXTS, ALPHABET, D=3):
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
    """
    rho_x, rho_y: size (num_contexts, )
    theta_x, theta_y: size (num_contexts, d)
    return: avg_weight: size (num_contexts, d)
    """
    Nx = np.sum(rho_x)
    Ny = np.sum(rho_y)
    avg_weight = rho_x.reshape((-1, 1)) * theta_x / Nx + rho_y.reshape((-1, 1)) * theta_y / Ny
    return avg_weight

def compute_G_beta(alpha, beta):
    """
    alpha: size (num_contexts, d)
    beta: size (d, )
    G: size (num_contexts, )
    """
    alpha_beta = alpha + beta
    beta_sum = np.sum(beta)
    a_b_sum = np.sum(alpha_beta, axis=1)
    G = gamma_function(beta_sum) * np.prod(gamma_function(alpha_beta), axis=1) / (np.prod(gamma_function(beta)) * gamma_function(a_b_sum))
    return G

def compute_K(gamma, betas, sigma, avg_weight):
    """
    gamma: (n, )
    betas: (n, d)
    sigma: float
    avg_weight: (num_contexts, d)
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
            avg_weight_new[i, :] = np.sum(avg_weight[i:i+d, :], axis=0)
        K_new = compute_K(gamma, betas, sigma, avg_weight_new) # size (layer_size//d, )
        Gamma_new = (1 - epsilon) * K_new
        for i in range(layer_size//d):
            Gamma_new[i] += epsilon * np.prod(Gamma[i:i+d])
        zero_idx = avg_weight_new.sum(axis=1) == 0
        Gamma_new[zero_idx] = 1

        avg_weight = avg_weight_new
        Gamma = Gamma_new
    return Gamma[0]

ALPHABET = ['A', 'C', 'G', 'T']
D = 3
CONTEXTS = all_contexts(ALPHABET, D=D)
logging.info("All contexts: {}".format(CONTEXTS))

epsilon = 1. / len(ALPHABET)
num_dirichelet = 1
gamma = [1 / num_dirichelet for _ in range(num_dirichelet)]
betas = np.full((num_dirichelet, len(ALPHABET)), 0.5)
sigma = 2
lamdas = [0.1, 1, 10]

benchmark = pd.DataFrame(columns=['train 0', 'val 0', 'train 1', 'val 1', 'train 2', 'val 2'], index=lamdas)
benchmark.index.name = 'lambda'
for lamda_idx, lamda in enumerate(lamdas):
    logging.info("lamda = {}".format(lamda))
    for dataset_id in [0, 1, 2]:
        logging.info("Data set No.{}".format(dataset_id))
        train_X = pd.read_csv('./data/Xtr{}.csv'.format(dataset_id), header=0, index_col=0).values.flatten()[:100]
        test_X = pd.read_csv('./data/Xte{}.csv'.format(dataset_id), header=0, index_col=0).values.flatten()[:100]
        train_Y = pd.read_csv('./data/Ytr{}.csv'.format(dataset_id), header=0, index_col=0)['Bound'][:100]
        # train_Y[train_Y==0] = -1
        logging.info("Loaded {} train samples, {} test samples".format(len(train_X), len(test_X)))
        assert len(train_X) == len(train_Y)

        n_samples = len(train_X)
        filter_idx = np.full((n_samples,), True)
        choose_idx = np.random.choice(range(n_samples), size=int(0.3*n_samples), replace=False)
        filter_idx[choose_idx] = False
        val_X = train_X[~filter_idx]
        val_Y = train_Y[~filter_idx]
        train_X = train_X[filter_idx]
        train_Y = train_Y[filter_idx]
        # cut_pos = int(0.3*n_samples)
        # val_X = train_X[:cut_pos]
        # val_Y = train_Y[:cut_pos]
        # train_X = train_X[cut_pos:]
        # train_Y = train_Y[cut_pos:]
        logging.info("Split train samples into {} validation samples and {} training samples".format(len(val_X), len(train_X)))
        
        n_train = len(train_X)

        logging.info("Computing kernel matrix on training set")
        K = np.zeros((n_train, n_train), dtype=float)
        counts_rho, counts_theta = count_occurrences(train_X, CONTEXTS, ALPHABET, D=D)

        for i in range(n_train):
            if i % 100 == 0:
                logging.info("sample No.{}/{}".format(i, n_train))
            K[i, i] = compute_kernel_value(counts_rho[i], counts_rho[i], counts_theta[i], counts_theta[i], epsilon, gamma, betas, sigma)
            for j in range(i+1, n_train):
                K[i, j] = compute_kernel_value(counts_rho[i], counts_rho[j], counts_theta[i], counts_theta[j], epsilon, gamma, betas, sigma)
                K[j, i] = K[i, j]
        
        P = matrix(K, tc='d')
        q = matrix(-train_Y, tc='d')
        G = matrix(np.concatenate((np.diag(train_Y), np.diag(-train_Y)), axis=0), tc='d')
        h = matrix(np.concatenate((np.ones(n_train) / (2 * lamda * n_train), np.zeros(n_train))), tc='d')

        sol = solvers.qp(P, q, G, h)
        # print(sol['x'])
        alpha = sol['x']

        supp_alpha = []
        supp_idx = []
        for i, ele in enumerate(alpha):
            if ele != 0:
                supp_idx.append(i)
                supp_alpha.append(ele)
        supp_alpha = np.array(supp_alpha).reshape((-1, 1))
        logging.info("Computing on validation set")
        n_support = len(supp_alpha)

        K_train = K[:, supp_idx]
        pred_train = np.dot(K_train, supp_alpha) >= 0
        pred_train = pred_train.astype(int).flatten()

        train_accuracy = np.mean(pred_train == train_Y)
        logging.info("Obtain training accuracy: {}".format(train_accuracy))
        benchmark.iloc[lamda_idx, 2*dataset_id] = train_accuracy

        n_val = len(val_X)
        K_val = np.zeros((n_support, n_val), dtype=float)

        counts_rho_val, counts_theta_val = count_occurrences(val_X, CONTEXTS, ALPHABET, D=D)
        for i in range(n_support):
            for j in range(n_val):
                supp_i = supp_idx[i]
                K_val[i, j] = compute_kernel_value(counts_rho[supp_i], counts_rho_val[j], counts_theta[supp_i], counts_theta_val[j], epsilon, gamma, betas, sigma)

        pred_val = np.dot(K_val.T, supp_alpha) >= 0
        pred_val = pred_val.astype(int).flatten()

        val_accuracy = np.mean(pred_val == val_Y)
        logging.info("Obtain validation accuracy: {}".format(val_accuracy))
        benchmark.iloc[lamda_idx, 2*dataset_id+1] = val_accuracy

        logging.info("Computing on test set")
        n_test= len(test_X)
        K_test = np.zeros((n_support, n_test), dtype=float)
        counts_rho_test, counts_theta_test = count_occurrences(test_X, CONTEXTS, ALPHABET, D=D)
        for i in range(n_support):
            for j in range(n_test):
                supp_i = supp_idx[i]
                K_test[i, j] = compute_kernel_value(counts_rho[supp_i], counts_rho_test[j], counts_theta[supp_i], counts_theta_test[j], epsilon, gamma, betas, sigma)

        pred_test = np.dot(K_test.T, supp_alpha) >= 0
        pred_test = pred_test.astype(int)

        pred_test = pd.DataFrame(pred_test, columns=['Bound'])
        pred_test.index.name = 'Id'
        pred_test.to_csv("pred_te{}_lamda_{}.csv".format(dataset_id, lamda))

benchmark.to_csv('benchmark.csv')

best_lamda_index = benchmark.mean(axis=1).idxmax()
best_lamda = lamdas[int(best_lamda_index)]

logging.info("Choosing best lamda: {}".format(best_lamda))
lamda = best_lamda
for i in range(3):
    logging.info("Data set No.{}".format(i))
    train_X = pd.read_csv('./data/Xtr{}.csv'.format(i), header=0, index_col=0).values.flatten()[:100]
    test_X = pd.read_csv('./data/Xte{}.csv'.format(i), header=0, index_col=0).values.flatten()[:100]
    train_Y = pd.read_csv('./data/Ytr{}.csv'.format(i), header=0, index_col=0)['Bound'][:100]
    # train_Y[train_Y==0] = -1
    logging.info("Loaded {} train samples, {} test samples".format(len(train_X), len(test_X)))
    
    n_train = len(train_X)

    logging.info("Computing kernel matrix on training set")
    K = np.zeros((n_train, n_train), dtype=float)
    counts_rho, counts_theta = count_occurrences(train_X, CONTEXTS, ALPHABET, D=D)

    for i in range(n_train):
        if i % 100 == 0:
            logging.info("sample No.{}/{}".format(i, n_train))
        K[i, i] = compute_kernel_value(counts_rho[i], counts_rho[i], counts_theta[i], counts_theta[i], epsilon, gamma, betas, sigma)
        for j in range(i+1, n_train):
            K[i, j] = compute_kernel_value(counts_rho[i], counts_rho[j], counts_theta[i], counts_theta[j], epsilon, gamma, betas, sigma)
            K[j, i] = K[i, j]
    
    P = matrix(K, tc='d')
    q = matrix(-train_Y, tc='d')
    # G = matrix(np.diag(np.concatenate((np.array(train_Y), -np.array(train_Y)))), tc='d')
    G = matrix(np.concatenate((np.diag(train_Y), np.diag(-train_Y)), axis=0), tc='d')
    h = matrix(np.concatenate((np.ones(n_train) / (2 * lamda * n_train), np.zeros(n_train))), tc='d')

    sol = solvers.qp(P, q, G, h)
    # print(sol['x'])
    alpha = sol['x']

    supp_alpha = []
    supp_idx = []
    for i, ele in enumerate(alpha):
        if ele != 0:
            supp_idx.append(i)
            supp_alpha.append(ele)
    n_support = len(supp_alpha)
    supp_alpha = np.array(supp_alpha).reshape((-1, 1))

    logging.info("Computing on test set")
    n_test= len(test_X)
    K_test = np.zeros((n_support, n_test), dtype=float)
    counts_rho_test, counts_theta_test = count_occurrences(test_X, CONTEXTS, ALPHABET, D=D)
    for i in range(n_support):
        for j in range(n_test):
            supp_i = supp_idx[i]
            K_test[i, j] = compute_kernel_value(counts_rho[supp_i], counts_rho_test[j], counts_theta[supp_i], counts_theta_test[j], epsilon, gamma, betas, sigma)

    pred_test = np.dot(K_test.T, supp_alpha) >= 0
    pred_test = pred_test.astype(int)

    pred_test = pd.DataFrame(pred_test, columns=['Bound'])
    pred_test.index.name = 'Id'
    pred_test.to_csv("pred_total_te{}_lamda_{}.csv".format(i, lamda))
