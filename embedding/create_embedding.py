import os
import pickle
import codecs
import numpy as np

np.random.seed(42)


class Embedding:

    def __init__(self, embedding_path, vector_dim=300):
        self.vector_dim = vector_dim
        self.embedding_path = embedding_path
        self.vocab_dict, self.vectors = self.__create_embeddings()
        self.__append_tokens()

    def create_word2embed(self, caption):
        """
        :param caption: list(caption_length)
        :return: numpy or tensor(batch_size, caption_count, sentence_length, self.vector_dim)
        """
        unk_index = len(self.vectors) - 4

        id_list = list(map(lambda x: self.vocab_dict.get(x, unk_index), caption))
        vectors = list(map(lambda x: self.vectors[x], id_list))

        return np.array(vectors)

    def __create_embeddings(self):
        dict_path = os.path.join(self.embedding_path, 'vocab.dict.pkl')
        vector_path = os.path.join(self.embedding_path, 'vector.npy')
        if os.path.isfile(dict_path) and os.path.isfile(vector_path):
            dict_file = open(dict_path, 'rb')
            vocab_dict = pickle.load(dict_file)
            vectors = np.load(vector_path)
        else:
            embed_txt = os.path.join(self.embedding_path, 'glove_original.txt')
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

    def __append_tokens(self):
        corpus_len = len(self.vectors)

        unk_token = np.random.rand(1, self.vector_dim)
        start_token = np.random.rand(1, self.vector_dim)
        end_token = np.random.rand(1, self.vector_dim)
        null_token = np.zeros_like(unk_token)

        self.vectors = np.concatenate((self.vectors, unk_token,
                                       start_token, end_token,
                                       null_token), axis=0)

        for idx, token in enumerate(['x_UNK_', 'x_START_', 'x_END_', 'x_NULL_']):
            self.vocab_dict[token] = idx + corpus_len


if __name__ == '__main__':
    glove_name = 'embedding'
    embed = Embedding(glove_name)
    example = [[['a', 'couple', 'sitting', 'on', 'the', 'back', 'of', 'a', 'horse', 'drawn', 'carriage'],
               ['a', 'horse', 'and', 'carriage', 'ride', 'in', 'an', 'old', 'town', 'x_UNK_', 'x_UNK_']]]
    vects = embed.create_word2embed(example)
    print(vects.shape)