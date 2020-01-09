import os
import pickle
import numpy as np
from gensim.models import KeyedVectors


class Word2VecTransformer:
    def __init__(self):
        self.model_path = 'embedding/word2vec.pkl'
        self.vector_path = 'embedding/word2vec.vec'

        if os.path.isfile(self.model_path):
            print('loading pretrained embeddings from pickle')
            model_file = open(self.model_path, 'rb')
            self.model = pickle.load(model_file)
        else:
            self.model = KeyedVectors.load_word2vec_format(self.vector_path, binary=False, unicode_errors='replace')
            model_file = open(self.model_path, 'wb')
            pickle.dump(self.model, model_file)

        self.vector_size = self.model.vector_size

    def transform(self, x):
        if x in self.model.wv:
            r = self.model.wv[x]
        elif x.lower() in self.model.wv:
            r = self.model.wv[x.lower()]
        else:
            r = self.create_random_vec()
        return r

    def create_random_vec(self):
        out_of_vocab_vector = np.random.rand(1, self.vector_size)[0]
        out_of_vocab_vector = out_of_vocab_vector - np.linalg.norm(out_of_vocab_vector)
        return out_of_vocab_vector
