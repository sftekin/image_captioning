import os
import config
import pandas as pd
from data_extractor import get_data
from models.cnn_lstm import CNNLSTM
from batch_generator import BatchGenerator

extract_data = False

if __name__ == '__main__':
    data_parameters = config.DataParams().__dict__
    model_parameters = config.CNNLSTMParams().__dict__

    samples = os.listdir("./dataset/images/")
    image_samples = [sample for sample in samples if sample.split(".")[-1] in ["png", "jpg"]]

    code_dictionary = pd.read_csv(data_parameters["data_path"]["code_dict_path"])

    if not image_samples or extract_data:
        file_name = "./dataset/eee443_project_dataset_train.h5"
        get_data(file_name)

    batch_generator = BatchGenerator(data_path=data_parameters["data_path"],
                                     batch_size=data_parameters["batch_size"],
                                     im_size=data_parameters["input_size"],
                                     min_num_captions=data_parameters["min_num_captions"],
                                     sequence_length=data_parameters["sequence_length"],
                                     word_length=data_parameters["word_length"])

    model = CNNLSTM(data_params=data_parameters,
                    params=model_parameters,
                    code_dictionary=code_dictionary)

    for _ in range(0):
        batch_x, batch_y = next(batch_generator)
        model.fit(batch_x, batch_y)

    batch_x, _ = next(batch_generator)
    out = model.predict(batch_x)
    pass
