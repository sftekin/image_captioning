import torch.nn as nn
from losses import loss_dict
from optimizers import optimizer_dict


class BaseModel(nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.params = params

        self.image_process = None
        self.embedding = None
        self.word_process = None

        self._construct_model()
        self.optimizer = optimizer_dict[params](self.parameters(), **params["optimizer_params"])
        self.criterion = loss_dict[params["loss"]](**params["loss_params"])

    def forward(self, input_):
        """

        :param input_: (b, d, m, n)
        :return:
        """
        image_features = self.image_process(input_)             # (b, f)
        generated_words = self.word_process(image_features)     # (b, l, w)
        return generated_words

    def fit(self, batch_x, batch_y):
        """
        Single step prediction.

        :param batch_x: (b, d, m, n) Input image.
        :param batch_y: (b, l) Caption of the input image.
        :return:
        """
        generated_words = self(batch_x)
        self.optimizer.zero_grad()

        loss = self.criterion(generated_words, batch_y)
        loss.backward()

        self.optimizer.step(loss)
        return loss.item()

    def caption(self, batch_x):
        """

        :param batch_x: (b, d, m, n)
        :return:
        """
        image_features = self.image_process(batch_x)                 # (b, f)
        generated_words = self.word_process.caption(image_features)  # (b, l)
        caption = self.embedding.translate(generated_words)
        return caption
