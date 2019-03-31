'''Embedding sequences by 12-mers using fasttext
'''

import numpy as np
import pandas as pd

def embed(infiles, w2v_file, out_files, kmer=12, dim=100):
    with open('./data/' + w2v_file, 'r') as f:
        lines = f.read().splitlines()
    num_vec, dim = lines.pop(0).split(' ')
    num_vec = int(num_vec)
    dim = int(dim)
    assert(len(lines) == num_vec)
    matrix = np.zeros((num_vec+1, dim))
    vocab_dict = {}
    for i in range(num_vec):
        vec = lines.pop(0).split(' ')
        word = vec.pop(0)
        vocab_dict[word] = i
        vec = np.array([np.float32(x) for x in vec])
        matrix[i, :] = vec
    matrix[:-1] = matrix[:-1] / np.linalg.norm(matrix[:-1], axis=1, keepdims=True) # normalize each row

    for i, filename in enumerate(infiles):
        out = open('./ft2vec/' + out_files[i], 'w')
        sequences = pd.read_csv('./data/' + filename, sep=',')['seq'].values
        for seq in sequences:
            embeddings = np.zeros((len(seq)-kmer+1, dim))
            for j in range(len(seq)-kmer+1):
                word = seq[j:j+kmer]
                if word in vocab_dict.keys():
                    embeddings[j] = matrix[vocab_dict[word]]
                else:
                    embeddings[j] = matrix[-1]
            out.write(' '.join([str(x) for x in np.sum(embeddings, axis=0)]) + '\n')
        out.close()

infiles = ['Xtr0.csv', 'Xtr1.csv', 'Xtr2.csv', 'Xte0.csv', 'Xte1.csv', 'Xte2.csv']
w2v_file = 'ft_w2v.vec'
out_files = ['Xtr0_ft2vec.csv', 'Xtr1_ft2vec.csv', 'Xtr2_ft2vec.csv', 'Xte0_ft2vec.csv', 'Xte1_ft2vec.csv', 'Xte2_ft2vec.csv']
embed(infiles, w2v_file, out_files)
