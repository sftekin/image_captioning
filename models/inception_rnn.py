from base_model import BaseModel
from image_modules.inception import Inceptionv3
from word_modules.rnn import rnn_models
from word_modules.lstm import lstm_models


class InceptionRNN(BaseModel):
    def __init__(self, params):
        super(InceptionRNN, self).__init__(params)

    def _construct_model(self):
        self.image_process = Inceptionv3(**self.params)
        self.params["feature_dim"] = self.image_process.feature_dim
        self.word_process = rnn_models[self.params["rnn_flow"]](embedding=self.embedding,
                                                                **self.params)


class InceptionLSTM(BaseModel):
    def __init__(self, params):
        super(InceptionLSTM, self).__init__(params)

    def _construct_model(self):
        self.image_process = Inceptionv3(**self.params)
        self.params["feature_dim"] = self.image_process.feature_dim
        self.word_process = lstm_models[self.params["rnn_flow"]](embedding=self.embedding,
                                                                 **self.params)
