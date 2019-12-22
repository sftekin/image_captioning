import torch
import torch.nn as nn
from losses import loss_dict
from optimizers import optimizer_dict
from embedding.create_embedding import Embedding


class BaseModel(nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.params = params

        self.image_process = None
        self.word_process = None
        self.embedding = Embedding(params["dataset_path"], device=params["device"])

        self._construct_model()
        self.optimizer = optimizer_dict[params["optimizer_type"]](self.parameters(),
                                                                  **params["optimizer_params"])

        self.criterion = loss_dict[params["criterion_type"]](**params["criterion_params"])

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
        batch_x = batch_x.to(self.params["device"])
        batch_y = batch_y.to(self.params["device"])

        generated_words = self(batch_x)
        self.optimizer.zero_grad()

        loss = 0

        for l in range(self.params["sequence_length"]):
            cur_word = generated_words.narrow(1, l, 1).squeeze(dim=1)
            loss += self.criterion(cur_word, batch_y[:, l])
        loss.backward(retain_graph=True)

        self.optimizer.step()
        return loss.item()

    def caption(self, batch_x):
        """

        :param batch_x: (b, d, m, n)
        :return:
        """
        batch_x = batch_x.to(self.params["device"])
        with torch.no_grad():
            image_features = self.image_process(batch_x)                 # (b, f)
            generated_words = self.word_process.caption(image_features)  # (b, l)
            caption = self.embedding.translate(generated_words)
        return caption
