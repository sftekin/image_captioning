import os
import csv
import pickle
import codecs
import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)  # To produce same vectors for appended tokens


class Embedding(nn.Module):

    def __init__(self, dataset_path, train_on, vector_dim=300, device='cpu'):
        nn.Module.__init__(self)

        self.word2int, self.int2word = self.__create_dicts(dataset_path)
        # self.word2int, self.int2word = self.__preprocess_dicts()
        self.num_vector = len(self.word2int)
        self.vector_dim = vector_dim

        # Init vectors to uniform distribution [-1, 1]
        self.vectors = -2 * torch.rand(self.num_vector, self.vector_dim) + 1
        self.vectors = nn.Parameter(self.vectors, requires_grad=train_on).to(device)

    def load_pre_trained(self, embedding_path, limited=False):
        vocab_load, vectors_load = self.__create_embeddings(embedding_path, limited)

        for idx, word in enumerate(self.word2int.keys()):
            vec_idx = vocab_load[word]
            self.vectors[idx, :] = torch.from_numpy(vectors_load[vec_idx])
            self.vocab_dict[word] = torch.from_numpy(vectors_load[vec_idx])

    def forward(self, one_hot):
        return torch.matmul(one_hot, self.vectors)

    def __getitem__(self, caption):
        """
        :param caption: list(caption_length)
        :return: numpy or tensor(batch_size, caption_count, sentence_length, self.vector_dim)
        """
        return self.vocab_dict[caption]

    def translate(self, captions):
        sentence = ' '.join([self.int2word[int(v)]
                             for v in captions]).replace("x_UNK_", "").replace("x_NULL_", "")
        return sentence

    def __create_embeddings(self, embedding_path, limited):
        dict_path = os.path.join(embedding_path, 'vocab.dict.pkl')
        vector_path = os.path.join(embedding_path, 'vector.npy')
        if os.path.isfile(dict_path) and os.path.isfile(vector_path):
            dict_file = open(dict_path, 'rb')
            vocab_dict = pickle.load(dict_file)
            vectors = np.load(vector_path)
        else:
            if limited:
                embed_txt = os.path.join(embedding_path, 'limited_glove_vectors.txt')
            else:
                embed_txt = os.path.join(embedding_path, 'glove_original.txt')
            vocab_dict = {}
            with codecs.open(embed_txt, 'r', "UTF-8") as f:
                content = f.readlines()
                vocab_size = len(content)
                words = [""] * vocab_size
                vectors = np.zeros((vocab_size, self.vector_dim))
                for idx, line in enumerate(content):
                    vals = line.rstrip().split(' ')
                    words[idx] = vals[0]
                    vocab_dict[vals[0]] = idx  # indices start from 0
                    vec = list(map(float, vals[1:]))
                    try:
                        vectors[idx, :] = vec
                    except IndexError:
                        if vals[0] == '<unk>':  # ignore the <unk> vector
                            pass
                        else:
                            raise Exception('IncompatibleInputs')

            dict_file = open(dict_path, 'wb')
            pickle.dump(vocab_dict, dict_file)
            np.save(vector_path, vectors)

        return vocab_dict, vectors

    def __preprocess_dicts(self):
        # change indexes of x_START_, x_UNK_, x_NULL_, x_END_
        vocab_len = len(self.word2int)
        new_word2int = {}
        new_int2word = {}
        for idx, word in enumerate(self.word2int.keys()):
            if word not in ['x_START_', 'x_UNK_', 'x_NULL_', 'x_END_']:
                new_word2int[word] = idx - 4
                new_int2word[idx - 4] = word

        for idx, token in enumerate(['x_END_', 'x_START_', 'x_UNK_', 'x_NULL_']):
            pos = vocab_len-4 + idx
            new_int2word[pos] = token
            new_word2int[token] = pos

        return new_word2int, new_int2word

    @staticmethod
    def __create_dicts(dataset_path):

        word2int_csv_path = os.path.join(dataset_path, 'word2int.csv')

        data = []
        with open(word2int_csv_path, mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                data.append(row)

        word2int = {}
        for word, value in zip(data[0], data[1]):
            word2int[word] = int(float(value))

        word2int = {k: v for k, v in sorted(word2int.items(), key=lambda item: item[1])}
        int2word = {v: k for k, v in word2int.items()}
        return word2int, int2word

    @property
    def vocab_dict(self):
        return {word: self.vectors[i].clone()
                for i, word in enumerate(self.word2int.keys())}


if __name__ == '__main__':

    glove_name = 'embedding'
    embed = Embedding('../dataset', train_on=False)

    embed.load_pre_trained('', limited=True)
    print(embed['x_START_'])
    # embed.load_pre_trained('', limited=False)
