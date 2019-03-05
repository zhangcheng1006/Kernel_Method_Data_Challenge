"""Use dna2vec to embed a sequence/"""
import numpy as np
import pandas as pd

from dna2vec.dna2vec.multi_k_model import MultiKModel

filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
dna2vec = MultiKModel(filepath)
# print(dna2vec.vector('AAA'))

def seq2vec(d2v, seq, epochs=5, low=3, high=5):
    """ Fragment a sequence into non-overlapping k-mers,
    Sum vectors of all k-mers to obtain the embedding of the sequence,
    Repeat several times and average the obtained sequence embeddings.
    """
    assert low >= d2v.k_low
    assert high <= d2v.k_high
    vec = np.zeros((d2v.vec_dim, ))
    for _ in range(epochs):
        anchor = 0
        vec_epoch = np.zeros((d2v.vec_dim,))
        while len(seq) - anchor > (high + low):
            k = np.random.randint(low, high+1)
            vec_epoch += d2v.vector(seq[anchor:(anchor+k)])
            anchor += k
        # remaining length less than a long k-mer, take it as a single last k-mer
        if len(seq) - anchor <= high:
            vec_epoch += d2v.vector(seq[anchor:])
        # remaining length enough for 2 k-mers, split it
        else:
            try:
                k = np.random.randint(low, len(seq)-anchor-low+1)
            except:
                print(anchor, k)
            vec_epoch += d2v.vector(seq[anchor:(anchor+k)])
            vec_epoch += d2v.vector(seq[(anchor+k):])
        vec += vec_epoch
    vec /= epochs
    return vec

data_files = ['Xtr0', 'Xtr1', 'Xtr2', 'Xte0', 'Xte1', 'Xte2']
for data_file in data_files:
    print("vectorizing file {}".format(data_file))
    sequences = pd.read_csv('data/{}.csv'.format(data_file), header=0, index_col=0)
    seq_vec = sequences['seq'].apply(lambda s: seq2vec(dna2vec, s))
    seq_array = np.array(seq_vec.to_list())
    np.savetxt('{}_dna2vec.csv'.format(data_file), seq_array, delimiter=' ', newline='\n', fmt='%1.18e')
