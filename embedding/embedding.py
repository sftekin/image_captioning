import codecs
import numpy as np


class Embedding:

    def __init__(self, file_name):
        self.vector_dim = 300
        self.file_name = file_name
        self.vocab_dict = dict()
        self.vectors = []
        self.__create_embeddings()

    def create_word2embed(self, batch_captions):
        """
        :param batch_captions: list(batch_size, caption_count, caption_length)
        :return: numpy or tensor(batch_size, caption_count, self.vector_dim)
        """
        batch_size = len(batch_captions)
        batch_vectors = [[]] * batch_size
        for batch_id in range(batch_size):
            vectors = []
            for caption in batch_captions[batch_id]:
                id_list = list(map(lambda x: self.vocab_dict[x], caption))
                vectors.append(list(map(lambda x: self.vectors[x], id_list)))
            batch_vectors[batch_id] = vectors
        return batch_vectors

    def __create_embeddings(self):
        with codecs.open(self.file_name, 'r', "UTF-8") as f:
            content = f.readlines()
            vocab_size = len(content)
            words = [""] * vocab_size
            vectors = np.zeros((vocab_size, self.vector_dim))
            for idx, line in enumerate(content):
                vals = line.rstrip().split(' ')
                words[idx] = vals[0]
                self.vocab_dict[vals[0]] = idx  # indices start from 0
                vec = list(map(float, vals[1:]))
                try:
                    vectors[idx, :] = vec
                except IndexError:
                    if vals[0] == '<unk>':  # ignore the <unk> vector
                        pass
                    else:
                        raise Exception('IncompatibleInputs')

            self.vectors = vectors


if __name__ == '__main__':
    glove_name = 'glove_original.txt'
    embed = Embedding(glove_name)
    example = [[['a', 'couple', 'sitting', 'on', 'the', 'back', 'of', 'a', 'horse', 'drawn', 'carriage'],
               ['a', 'horse', 'and', 'carriage', 'ride', 'in', 'an', 'old', 'town']]]
    vects = embed.create_word2embed(example)
    print(len(vects), len(vects[0]), len(vects[0][0]))
    print(np.array(vects[0][0]).shape)
