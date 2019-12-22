from base_model import BaseModel
from image_modules.vgg import VGG16
from word_modules.rnn import rnn_models
from word_modules.lstm import lstm_models


class VggRNN(BaseModel):
    def __init__(self, params):
        super(VggRNN, self).__init__(params)

    def _construct_model(self):
        self.image_process = VGG16(**self.params)
        self.params["feature_dim"] = self.image_process.feature_dim
        self.word_process = rnn_models[self.params["rnn_flow"]](embedding=self.embedding,
                                                                **self.params)


class VggLSTM(BaseModel):
    def __init__(self, params):
        super(VggLSTM, self).__init__(params)

    def _construct_model(self):
        self.image_process = VGG16(**self.params)
        self.params["feature_dim"] = self.image_process.feature_dim
        self.word_process = lstm_models[self.params["rnn_flow"]](embedding=self.embedding,
                                                                 **self.params)
