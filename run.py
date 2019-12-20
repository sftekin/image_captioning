import config
from data_extractor import get_data
from models.vgg_rnn import VggRNN
from models.inception_rnn import InceptionRNN
from batch_generator import BatchGenerator


models = {"vggrnn": {"model": VggRNN,
                     "params": config.VggRNNParams},

          "inceptionrnn": {"model": InceptionRNN,
                           "params": config.InceptionRNNParams}}


extract_data = False


def main():
    data_parameters = config.DataParams().__dict__
    model_parameters = models[data_parameters["model_name"]]["params"]().__dict__
    parameters = model_parameters.copy()
    parameters.update(data_parameters)

    get_data(parameters)

    model = models[parameters["model_name"]]["model"](parameters)
    batch_gen = BatchGenerator(**parameters)

    for e in range(parameters["num_epochs"]):
        for idx, (im, cap) in enumerate(batch_gen.generate('train')):
            model.fit(im, cap)


if __name__ == '__main__':
    main()
