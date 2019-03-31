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
    G = np.full((alpha.shape[0],), gamma_function(beta_sum)/gamma_function(a_b_sum))
    for i in range(len(beta)):
        G *= gamma_function(alpha_beta[:, i]) / gamma_function(beta[i])
    assert G.shape == (alpha.shape[0],)
    # G = gamma_function(beta_sum) * np.prod(gamma_function(alpha_beta), axis=1) / (np.prod(gamma_function(beta)) * gamma_function(a_b_sum))
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
    new_K = np.zeros(K.shape, dtype=float)
    n = K.shape[0]
    for i in range(n):
        new_K[i, i] = 1
        for j in range(i+1, n):
            new_K[i, j] = K[i, j] / np.sqrt(K[i, i] * K[j, j])
            new_K[j, i] = new_K[i, j]
    return new_K

ALPHABET = ['A', 'C', 'G', 'T']
D = 3
CONTEXTS = all_contexts(ALPHABET, D=D)
logging.info("All contexts: {}".format(CONTEXTS))

epsilon = 1. / len(ALPHABET)
num_dirichelet = 1
gamma = [1 / num_dirichelet for _ in range(num_dirichelet)]
betas = np.full((num_dirichelet, len(ALPHABET)), 0.5)
sigma = 2
lamdas = [1e-6, 1e-7]

# benchmark = pd.DataFrame(columns=['train 0', 'val 0', 'train 1', 'val 1', 'train 2', 'val 2'], index=lamdas)
# benchmark.index.name = 'lambda'
# for lamda_idx, lamda in enumerate(lamdas):
#     logging.info("\n\nlamda = {}".format(lamda))
#     for dataset_id in [0, 1, 2]:
#         logging.info("Data set No.{}".format(dataset_id))
#         train_X = pd.read_csv('./data/Xtr{}.csv'.format(dataset_id), header=0, index_col=0).values.flatten()
#         train_Y = pd.read_csv('./data/Ytr{}.csv'.format(dataset_id), header=0, index_col=0)['Bound']
#         train_Y[train_Y==0] = -1
#         logging.info("Loaded {} train samples".format(len(train_X)))
#         assert len(train_X) == len(train_Y)

#         n_samples = len(train_X)
#         filter_idx = np.full((n_samples,), True)
#         choose_idx = np.random.choice(range(n_samples), size=int(0.3*n_samples), replace=False)
#         filter_idx[choose_idx] = False
#         val_X = train_X[~filter_idx]
#         val_Y = train_Y[~filter_idx]
#         train_X = train_X[filter_idx]
#         train_Y = train_Y[filter_idx]
#         logging.info("Split train samples into {} validation samples and {} training samples".format(len(val_X), len(train_X)))
        
#         n_train = len(train_X)

#         logging.info("Computing kernel matrix on training set")
#         K = np.zeros((n_train, n_train), dtype=float)
#         counts_rho, counts_theta = count_occurrences(train_X, CONTEXTS, ALPHABET, D=D)

#         for i in range(n_train):
#             if i % 100 == 0:
#                 logging.info("train sample No.{}/{}".format(i+1, n_train))
#             K[i, i] = compute_kernel_value(counts_rho[i], counts_rho[i], counts_theta[i], counts_theta[i], epsilon, gamma, betas, sigma)
#             for j in range(i+1, n_train):
#                 K[i, j] = compute_kernel_value(counts_rho[i], counts_rho[j], counts_theta[i], counts_theta[j], epsilon, gamma, betas, sigma)
#                 K[j, i] = K[i, j]
  
        
#         K_norm = normalize_kernel_matrix(K)
        
        
#         P = matrix(K_norm, tc='d')
#         q = matrix(-train_Y, tc='d')
#         G = matrix(np.concatenate((np.diag(train_Y), np.diag(-train_Y)), axis=0), tc='d')
#         h = matrix(np.concatenate((np.ones(n_train) / (2 * lamda * n_train), np.zeros(n_train))), tc='d')

#         sol = solvers.qp(P, q, G, h)
#         alpha = sol['x']

#         supp_alpha = []
#         supp_idx = []
#         for i, ele in enumerate(alpha):
#             if ele != 0:
#                 supp_idx.append(i)
#                 supp_alpha.append(ele)
#         supp_alpha = np.array(supp_alpha).reshape((-1, 1))
        
#         n_support = len(supp_alpha)

#         K_train = K_norm[:, supp_idx]
#         pred_train = np.dot(K_train, supp_alpha) >= 0
#         pred_train = pred_train.astype(int).flatten()
#         pred_train[pred_train==0] = -1

#         train_accuracy = np.mean(pred_train == train_Y)
#         logging.info("Obtain training accuracy: {}".format(train_accuracy))
#         benchmark.iloc[lamda_idx, 2*dataset_id] = train_accuracy

#         logging.info("Computing on validation set")
#         n_val = len(val_X)
#         K_val = np.zeros((n_support, n_val), dtype=float)

#         counts_rho_val, counts_theta_val = count_occurrences(val_X, CONTEXTS, ALPHABET, D=D)
#         for j in range(n_val):
#             if j % 100 == 0:
#                 logging.info("validation sample No.{}/{}".format(j+1, n_val))
#             self_kernel_value = compute_kernel_value(counts_rho_val[j], counts_rho_val[j], counts_theta_val[j], counts_theta_val[j], epsilon, gamma, betas, sigma)
#             for i in range(n_support):
#                 supp_i = supp_idx[i]
#                 K_val[i, j] = compute_kernel_value(counts_rho[supp_i], counts_rho_val[j], counts_theta[supp_i], counts_theta_val[j], epsilon, gamma, betas, sigma)
#                 K_val[i, j] /= np.sqrt(self_kernel_value * K[supp_i, supp_i])

#         pred_val = np.dot(K_val.T, supp_alpha) >= 0
#         pred_val = pred_val.astype(int).flatten()
#         pred_val[pred_val==0] = -1

#         val_accuracy = np.mean(pred_val == val_Y)
#         logging.info("Obtain validation accuracy: {}".format(val_accuracy))
#         benchmark.iloc[lamda_idx, 2*dataset_id+1] = val_accuracy

# benchmark.to_csv('benchmark.csv')

# best_lamda_index = benchmark.mean(axis=1).idxmax()
# best_lamda = lamdas[int(best_lamda_index)]

# logging.info("Choosing best lamda: {}".format(best_lamda))
# lamda = best_lamda
lamda = 1e-7

result = pd.DataFrame(columns=['Bound'])

for dataset_id in [0, 1, 2]:
    logging.info("Data set No.{}".format(dataset_id))
    train_X = pd.read_csv('./data/Xtr{}.csv'.format(dataset_id), header=0, index_col=0).values.flatten()
    test_X = pd.read_csv('./data/Xte{}.csv'.format(dataset_id), header=0, index_col=0).values.flatten()
    train_Y = pd.read_csv('./data/Ytr{}.csv'.format(dataset_id), header=0, index_col=0)['Bound']
    train_Y[train_Y==0] = -1
    logging.info("Loaded {} train samples, {} test samples".format(len(train_X), len(test_X)))
    
    n_train = len(train_X)

    logging.info("Computing kernel matrix on training set")
    K = np.zeros((n_train, n_train), dtype=float)
    counts_rho, counts_theta = count_occurrences(train_X, CONTEXTS, ALPHABET, D=D)

    for i in range(n_train):
        if i % 100 == 0:
            logging.info("train sample No.{}/{}".format(i+1, n_train))
        K[i, i] = compute_kernel_value(counts_rho[i], counts_rho[i], counts_theta[i], counts_theta[i], epsilon, gamma, betas, sigma)
        for j in range(i+1, n_train):
            K[i, j] = compute_kernel_value(counts_rho[i], counts_rho[j], counts_theta[i], counts_theta[j], epsilon, gamma, betas, sigma)
            K[j, i] = K[i, j]
    
    K_norm = normalize_kernel_matrix(K)
    
    P = matrix(K_norm, tc='d')
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
    n_support = len(supp_alpha)
    supp_alpha = np.array(supp_alpha).reshape((-1, 1))

    logging.info("Computing on test set")
    n_test= len(test_X)
    K_test = np.zeros((n_support, n_test), dtype=float)
    counts_rho_test, counts_theta_test = count_occurrences(test_X, CONTEXTS, ALPHABET, D=D)
    for j in range(n_test):
        if j % 100 == 0:
            logging.info("test sample No.{}/{}".format(j+1, n_test))
        self_kernel_value = compute_kernel_value(counts_rho_test[j], counts_rho_test[j], counts_theta_test[j], counts_theta_test[j], epsilon, gamma, betas, sigma)
        for i in range(n_support):
            supp_i = supp_idx[i]
            K_test[i, j] = compute_kernel_value(counts_rho[supp_i], counts_rho_test[j], counts_theta[supp_i], counts_theta_test[j], epsilon, gamma, betas, sigma)
            K_test[i, j] /= np.sqrt(self_kernel_value * K[supp_i, supp_i])

    pred_test = np.dot(K_test.T, supp_alpha) >= 0
    pred_test = pred_test.astype(int)

    pred_test = pd.DataFrame(pred_test, columns=['Bound'])
    pred_test.index.name = 'Id'
    result = pd.concat((result, pred_test), axis=0, ignore_index=True)
    pred_test.to_csv("pred_total_te{}_lamda_{}.csv".format(dataset_id, lamda))

result.index.name = 'Id'
result.to_csv("submission_lamda_{}.csv".format(lamda))
