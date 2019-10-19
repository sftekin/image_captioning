import h5py
import pandas as pd


class LoadData:

    def __init__(self, dataset_path, caption_path,
                 caption_word_path, im_addr_path):

        self.caption_word_path = pd.read_csv(caption_word_path)
        self.captions = pd.read_csv(caption_path)
        self.im_addr = pd.read_csv(im_addr_path)
        self.word2int, self.int2word = self._create_dicts(dataset_path)

    @staticmethod
    def _create_dicts(filename):
        with h5py.File(filename, 'r') as f:
            word_key = list(f.keys())[-1]
            words_id = list(f[word_key])[0]
            words = f[word_key].dtype.names

        word2int = {}
        int2word = {}
        for idx, word in zip(words_id, words):
            word2int[word] = idx
            int2word[idx] = word

        return word2int, int2word

