import pickle
import numpy as np
import bcolz

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'glove.6B.50d.dat', mode='w')

with open(f'glove.6B.50d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if word == "<unk>":
            continue
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

vectors = bcolz.carray((vectors[1:]).reshape((400000, 50)), rootdir=f'glove.6B.50d.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'glove.6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'glove.6B.50_idx.pkl', 'wb'))
