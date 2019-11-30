import os
import csv
import pandas as pd

from PIL import Image


class LoadData:

    def __init__(self, dataset_path, images_path):

        self.word2int, self.int2word = self.__create_dicts(dataset_path)
        self.caption_words = self.__create_caption_words(dataset_path)
        self.captions_int = self.__create_captions_int(dataset_path)
        self.image_addr = self.__create_image_addr(dataset_path)
        self.image_paths = self.__create_image_paths(images_path)

    @staticmethod
    def __create_caption_words(dataset_path):
        """
        :param dataset_path: string
        :return: pd.DataFrame
        """
        caption_word_path = os.path.join(dataset_path, 'captions_words.csv')
        caption_words = pd.read_csv(caption_word_path)
        return caption_words

    @staticmethod
    def __create_captions_int(dataset_path):
        """
        :param dataset_path: string
        :return: pd.DataFrame
        """
        caption_path = os.path.join(dataset_path, 'captions.csv')
        captions = pd.read_csv(caption_path)
        return captions

    @staticmethod
    def __create_image_addr(dataset_path):
        """
        :param dataset_path: str
        :return: pd.DataFrame
        """
        im_addr_path = os.path.join(dataset_path, 'imid.csv')
        im_addr = pd.read_csv(im_addr_path)
        return im_addr

    @staticmethod
    def __create_image_paths(images_path):
        """
        :param images_path: string
        :return: list of strings
        """
        image_paths = [os.path.join(images_path, f) for f in sorted(os.listdir(images_path))]
        image_paths.pop(0)  # remove .gitignore file from the list

        # clean the dataset
        for im_path in image_paths:
            try:
                image = Image.open(im_path)
                image.verify()
            except OSError:
                os.remove(im_path)
                image_paths.remove(im_path)

        return image_paths

    @staticmethod
    def __create_dicts(dataset_path):

        word2int_csv_path = os.path.join(dataset_path, 'word2int.csv')

        data = []
        with open(word2int_csv_path, mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                data.append(row)

        word2int = {}
        int2word = {}
        for word, value in zip(data[0], data[1]):
            word2int[word] = value
            int2word[value] = word
        return word2int, int2word
