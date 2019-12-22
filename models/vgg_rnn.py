from base_model import BaseModel
from image_modules.vgg import VGG16
from word_modules.rnn import RNN
from word_modules.lstm import LSTM


class VggRNN(BaseModel):
    def __init__(self, params):
        super(VggRNN, self).__init__(params)

    def _construct_model(self):
        self.image_process = VGG16(**self.params)
        self.word_process = RNN(embedding=self.embedding,
                                **self.params)


class VggLSTM(BaseModel):
    def __init__(self, params):
        super(VggLSTM, self).__init__(params)

    def _construct_model(self):
        self.image_process = VGG16(**self.params)
        self.word_process = LSTM(embedding=self.embedding,
                                 **self.params)
