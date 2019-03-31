'''Train word embeddings using fasttext
'''

import numpy as np
import pandas as pd
from fastText import train_unsupervised

def preprocess(in_files, out_files, kmer=12):
    corpus = open('./data/' + out_files[1], 'w', encoding='utf8')
    l = len(in_files)
    words = []
    for i in range(l):
        filename = './data/' + in_files[i]
        sequences = pd.read_csv(filename, sep=',')['seq'].values
        for seq in sequences:
            sent = []
            for j in range(len(seq)-kmer+1):
                word = seq[j:j+kmer]
                words.append(word)
                sent.append(word)
            sent = ' '.join(sent)
            corpus.write(sent + '\n')
    corpus.close()

    with open('./data/' + out_files[0], 'w', encoding='utf8') as f:
        f.write('\n'.join(list(set(words))))

def train_embeddings(vocab, corpus, output_file, epoch=500, dim=100, min_count=1):
    ft = train_unsupervised('./data/' + corpus, epoch=epoch, dim=dim, minCount=min_count)

    with open('./data/' + vocab, 'r', encoding='utf8') as f:
        words = f.read().splitlines()
    with open('./data/' + output_file, 'w', encoding='utf8') as f:
        f.write('%d %d\n' % (len(words), dim))
        for word in words:
            vec = ft.get_word_vector(word)
            out = word + ' ' + ' '.join([str(x) for x in np.around(vec, decimals=4)]) + '\n'
            f.write(out)

in_files = ['Xtr0.csv', 'Xtr1.csv', 'Xtr2.csv']
vocab_file = 'vocab.list'
corpus_file = 'corpus.list'
out_files = [vocab_file, corpus_file]
embedding_file = 'ft_w2v.vec'

preprocess(in_files, out_files)
train_embeddings(vocab_file, corpus_file, embedding_file)
