import config
from data_extractor import get_data
from models.cnn_lstm import CNNLSTM
from batch_generator import BatchGenerator


def main():
    data_parameters = config.DataParams().__dict__
    model_parameters = config.CNNLSTMParams().__dict__
    parameters = model_parameters.copy()
    parameters.update(data_parameters)

    get_data(parameters)

    model = CNNLSTM(parameters)
    batch_gen = BatchGenerator(**parameters)

    for idx, (im, cap) in enumerate(batch_gen.generate('train')):
        model.fit(im, cap)


if __name__ == '__main__':
    main()
