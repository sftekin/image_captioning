from base_model import BaseModel
from image_modules.inception import Inceptionv3
from word_modules.rnn import RNN
from word_modules.lstm import LSTM


class InceptionRNN(BaseModel):
    def __init__(self, params):
        super(InceptionRNN, self).__init__(params)

    def _construct_model(self):
        self.image_process = Inceptionv3(**self.params)
        self.word_process = RNN(embedding=self.embedding,
                                **self.params)


class InceptionLSTM(BaseModel):
    def __init__(self, params):
        super(InceptionLSTM, self).__init__(params)

    def _construct_model(self):
        self.image_process = Inceptionv3(**self.params)
        self.word_process = LSTM(embedding=self.embedding,
                                 **self.params)
