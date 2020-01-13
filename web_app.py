import random
import config
from flask import Flask
import matplotlib.pyplot as plt
from data_extractor import get_data
from models.vgg_rnn import VggRNN, VggLSTM
from models.inception_rnn import InceptionRNN, InceptionLSTM
from batch_generator import BatchGenerator


model = None
batch_gen = None

models = {"vggrnn": {"model": VggRNN,
                     "params": config.VggRNNParams},

          "vgglstm": {"model": VggLSTM,
                      "params": config.VggLSTMParams},

          "inceptionrnn": {"model": InceptionRNN,
                           "params": config.InceptionRNNParams},

          "inceptionlstm": {"model": InceptionLSTM,
                            "params": config.InceptionLSTMParams}
          }


app = Flask(__name__)


@app.route('/train')
def train():
    global model
    global batch_gen

    for e in range(parameters["num_epochs"]):
        print("Epoch num: " + str(e))
        for idx, (im, cap) in enumerate(batch_gen.generate('train')):
            if idx == 20:
                break
            loss = model.fit(im, cap)
            print("\rTraining: " + str(loss) + " [" + "="*idx, end="", flush=True)
        print("]")

    return "Done!"


@app.route("/caption")
def caption():
    (ims, caps) = next(batch_gen.generate("train"))
    generated_captions = model.caption(ims)
    random_idx = random.sample(list(range(parameters["batch_size"])), 1)

    for idx in random_idx:
        cap = generated_captions[idx]
        img = ims[idx]
        img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.tight_layout()
        plt.title(cap)
        plt.show()

    return "Done!"


@app.route("/reset")
def reset():
    global model
    global batch_gen

    model = models[parameters["model_name"]]["model"](parameters)
    batch_gen = BatchGenerator(**parameters)

    return "Reset!"


if __name__ == '__main__':

    data_parameters = config.DataParams().__dict__
    model_parameters = models[data_parameters["model_name"]]["params"]().__dict__
    parameters = model_parameters.copy()
    parameters.update(data_parameters)

    get_data(parameters)

    model = models[parameters["model_name"]]["model"](parameters)
    batch_gen = BatchGenerator(**parameters)

    app.run(host='0.0.0.0')
