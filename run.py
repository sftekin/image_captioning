import os
import config
from models.cnn_lstm import CNNLSTM

from batch_generator import BatchGenerator


def main():
    data_parameters = config.DataParams().__dict__
    model_parameters = config.CNNLSTMParams().__dict__
    parameters = model_parameters.copy()
    parameters.update(data_parameters)

    model = CNNLSTM(parameters)
    batch_gen = BatchGenerator(parameters["dataset_path"], parameters["image_path"])

    for idx, (im, cap) in enumerate(batch_gen.generate('train')):
        loss = model.fit(im, cap)
        print("\rTraining Loss: " + str(loss), flush=True, end="")


if __name__ == '__main__':
    main()
