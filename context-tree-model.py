from collections import defaultdict
import numpy as np
from scipy.special import gamma as gamma_function

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

def compute_kernel_value(X_id, Y_id, counts_rho, counts_theta, epsilon, gamma, betas, sigma):
    d = counts_theta.shape[-1]
    # size (num_contexts, d)
    avg_weight = compute_avg_weight(counts_rho[X_id], counts_rho[Y_id], counts_theta[X_id], counts_theta[Y_id])
    # size (num_contexts, )
    K = compute_K(gamma, betas, sigma, avg_weight)
    Gamma = K
    while len(Gamma) > 1:
        layer_size = len(Gamma)
        assert layer_size % d == 0
        avg_weight_new = np.zeros((layer_size//d, d), dtype=float)
        # print("avg_weight_new", avg_weight_new.shape)
        for i in range(layer_size//d):
            try:
                avg_weight_new[i, :] = np.sum(avg_weight[i:i+d, :], axis=0)
            except:
                print("layer size", layer_size)
                print(avg_weight_new.shape)
                print(np.sum(avg_weight[i:i+d, :], axis=0).shape)
                exit(2)
        K_new = compute_K(gamma, betas, sigma, avg_weight_new) # size (layer_size//d, )
        Gamma_new = (1 - epsilon) * K_new
        for i in range(layer_size//d):
            Gamma_new[i] += epsilon * np.prod(Gamma[i:i+d])
        avg_weight = avg_weight_new
        Gamma = Gamma_new
    return Gamma[0]

X = [
    "GGAGAATCATTTGAACCCGGGAGGTGGAGGTTGCCGTGAGCTGAGATTGCGCCATTGCACTCCAGCCTGGGCAACAAGAGCAAAACTCTGTCTCACAAAAC",
    "TGCAAATCTGTAAGCATTTCTCAGGCAATGAATTATGTCAACACAATTGCACCATCATTGATGGACTTGGAAATGCAGACAGAACTGAAGAGGAGCGTCTC"
]
y = [0, 1]

ALPHABET = ['A', 'T', 'G', 'C']

n_samples = len(X)
D = 3
epsilon = 0.4
num_dirichelet = 1
gamma = [1 / num_dirichelet for _ in range(num_dirichelet)]
betas = np.full((num_dirichelet, len(ALPHABET)), 0.5)
sigma = 1

K = np.zeros((n_samples, n_samples), dtype=float)

CONTEXTS = all_contexts(ALPHABET, D=D)

counts_rho, counts_theta = count_occurrences(X, CONTEXTS, ALPHABET, D=D)
print(counts_rho)
print(counts_theta)

for i in range(n_samples):
    K[i, i] = compute_kernel_value(i, i, counts_rho, counts_theta, epsilon, gamma, betas, sigma)
    for j in range(i+1, n_samples):
        K[i, j] = compute_kernel_value(i, j, counts_rho, counts_theta, epsilon, gamma, betas, sigma)
        K[j, i] = K[i, j]

print(K)

y = np.array(y)

# P = matrix(2 * K, tc='d')
# q = matrix(- 2 * y, tc='d')
# G = matrix(np.concatenate((np.diag(y), np.diag(-y)), axis=0), tc='d')
# h = matrix(np.concatenate((np.ones(n) / (2 * lamda * n), np.zeros(n))), tc='d')

# sol = solvers.qp(P, q, G, h)
# # print(sol['x'])
# return sol['x']








