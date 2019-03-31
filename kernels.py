import numpy as np

def get_all_kmers(X, kmer):
    '''Get all kmers in training dataset. Utilised in di-mismatch kernel.
    '''
    kmers = set()
    for item in X:
        for i in range(len(item) - kmer):
            sub = item[i:i+kmer]
            kmers.add(sub)
    return list(kmers)

def compute_feature(s, phi, m):
    '''Compute features for di-mismatch kernel.
    '''
    kmer = len(phi)
    for i in range(len(s) - kmer):
        sub = s[i:i+kmer]
        num_matching = 0
        for j in range(kmer - 1):
            if sub[j]+sub[j+1] == phi[j]+phi[j+1]:
                num_matching += 1
    if num_matching < kmer - m - 1:
        num_matching = 0
    return num_matching

def di_mismatch(all_kmers, X, kmer=13, tol=5):
    '''Implementation of di-mismatch kernel
    '''
    d = len(all_kmers)
    n = X.shape[0]
    print(d)
    print(n)
    features = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            kmer = all_kmers[j]
            features[i, j] = compute_feature(X[i], kmer, tol)
    return features
    

def RBF_kernel(x, y, sigma=1.0):
    '''Implementation of Gaussian kernel
    '''
    return np.exp(- np.linalg.norm(x - y)**2 / (2 * sigma**2))

def poly_kernel(x, y, d=2):
    '''Implementation of polynomial kernel
    '''
    return (np.dot(np.reshape(x, (1, -1)), np.reshape(y, (-1, 1))) + 1) ** d

def Spectrum_kernel(x, y, k=10):
    '''Implementation of spectrum kernel
    '''
    l = len(x)
    spectrum_dict = {}
    for i in range(l-k):
        sub_x = x[i:i+k]
        sub_y = y[i:i+k]
        if sub_x in spectrum_dict.keys():
            spectrum_dict[sub_x][0] += 1
        else:
            spectrum_dict[sub_x] = [1, 0]
        if sub_y in spectrum_dict.keys():
            spectrum_dict[sub_y][1] += 1
        else:
            spectrum_dict[sub_y] = [0, 1]
    res = 0
    for key, value in spectrum_dict.items():
        res += value[0] * value[1]
    return res

def gapped_kernel(x, y, k=6):
    '''Implementation of spectrum kernel with gaps
    '''
    l = len(x)
    spectrum_dict = {}
    for i in range(l-k):
        sub_x = [x[i+k-1]]
        sub_y = [y[i+k-1]]
        for j in range(1, k):
            tmp_x = []
            tmp_y = []
            for a in range(len(sub_x)):
                if spectrum_dict.get(sub_x[a]) is not None:
                    spectrum_dict[sub_x[a]][0] += 1
                else:
                    spectrum_dict[sub_x[a]] = [1, 0]
                if spectrum_dict.get(sub_y[a]) is not None:
                    spectrum_dict[sub_y[a]][1] += 1
                else:
                    spectrum_dict[sub_y[a]] = [0, 1]
                
                tmp_x.append(x[i+k-j-1] + sub_x[a])
                tmp_y.append(y[i+k-j-1] + sub_y[a])
                tmp_x.append('N' + sub_x[a])
                tmp_y.append('N' + sub_y[a])
            sub_x = tmp_x
            sub_y = tmp_y

        for a in range(len(sub_x)):
            if spectrum_dict.get(sub_x[a]) is not None:
                spectrum_dict[sub_x[a]][0] += 1
            else:
                spectrum_dict[sub_x[a]] = [1, 0]
            if spectrum_dict.get(sub_y[a]) is not None:
                spectrum_dict[sub_y[a]][1] += 1
            else:
                spectrum_dict[sub_y[a]] = [0, 1]
                
    res = 0
    for value in spectrum_dict.values():
        res += value[0] * value[1]
    return res

def within_gap(x, y, m=5):
    '''Check if two strings are within gap of size m. Utilised in substring kernel
    '''
    l = len(x)
    for i in range(m):
        sub_x = x[i:l-m+i+1]
        for j in range(m):
            sub_y = y[j:l-m+j+1]
            if sub_x == sub_y:
                return True
    return False

def substring_kernel(x, y, kmer=13, m=5):
    '''Implementation of substring kernel
    '''
    l = len(x)
    substring_dict = {}
    for i in range(l-kmer):
        sub_x = x[i:i+kmer]
        sub_y = y[i:i+kmer]

        if substring_dict.get(sub_x) is None:
            substring_dict[sub_x] = [0, 0]
        if substring_dict.get(sub_y) is None:
            substring_dict[sub_y] = [0, 0]
        
        for key in substring_dict.keys():
            if within_gap(key, sub_x):
                substring_dict[key][0] += 1
            if within_gap(key, sub_y):
                substring_dict[key][1] += 1
        
    res = 0
    for key, value in substring_dict.items():
        res += value[0] * value[1]
    return res

