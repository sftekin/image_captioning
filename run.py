import pickle

from config import model_params, batch_params, train_params
from load_data import LoadData
from batch_generator import BatchGenerator
from models.caption_model import CaptionLSTM
from train_helper import calc_class_weights
from train_helper import sample
from train import train
from test import test


def main(mode):
    print('Loading data...')
    data = LoadData(dataset_path='dataset',
                    images_path='dataset/images/')

    print('Creating Batch Generator...')
    batch_creator = BatchGenerator(data_dict=data.data_dict,
                                   captions_int=data.captions_int,
                                   image_addr=data.image_addr,
                                   **batch_params)

    if mode == 'train':
        print('Creating Models...')
        caption_model = CaptionLSTM(model_params=model_params,
                                    int2word=data.int2word)

        print('Starting training...')
        class_weights = calc_class_weights(data.captions_int.values)
        train(caption_model, batch_creator, class_weights, **train_params)

    elif mode == 'sample':
        print('Loading model...')
        model_file = open('vgg_lstm.pkl', 'rb')
        model = pickle.load(model_file)
        print('Creating sample..')
        sample(model, batch_creator, top_k=10, seq_len=16, show_image=True)

    elif mode == 'test':
        print('Loading model')
        model_file = open('vgg_lstm.pkl', 'rb')
        model = pickle.load(model_file)
        print('Testing model...')
        test(model, batch_creator, top_k=10, seq_len=16)


if __name__ == '__main__':
    run_mode = 'train'
    main(run_mode)
