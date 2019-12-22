import random
import config
import matplotlib.pyplot as plt
from data_extractor import get_data
from models.vgg_rnn import VggRNN, VggLSTM
from models.inception_rnn import InceptionRNN, InceptionLSTM
from batch_generator import BatchGenerator


models = {"vggrnn": {"model": VggRNN,
                     "params": config.VggRNNParams},

          "vgglstm": {"model": VggLSTM,
                      "params": config.VggLSTMParams},

          "inceptionrnn": {"model": InceptionRNN,
                           "params": config.InceptionRNNParams},

          "inceptionlstm": {"model": InceptionLSTM,
                            "params": config.InceptionLSTMParams}
          }


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
        print("Epoch num: " + str(e))
        for idx, (im, cap) in enumerate(batch_gen.generate('train')):
            loss = model.fit(im, cap)
            print("\rTraining: " + str(loss) + " [" + "="*idx, end="", flush=True)
        print("]")

        (im, cap) = next(batch_gen.generate("train"))
        generated_caption = model.caption(im)
        random_idx = random.sample(list(range(parameters["batch_size"])), 5)
        for idx in random_idx:
            cap = generated_caption[idx]
            img = im[idx]
            img = (img.permute(1, 2, 0) - img.min())/(img.max() - img.min())
            plt.imshow(img)
            plt.title(cap)
            plt.show()


if __name__ == '__main__':
    main()
