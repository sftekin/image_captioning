from base_model import BaseModel
from image_modules.vgg import VGG16
from word_modules.rnn import RNN


class VggRNN(BaseModel):
    def __init__(self, params):
        super(VggRNN, self).__init__(params)

    def _construct_model(self):
        self.image_process = VGG16(**self.params)
        self.word_process = RNN(embedding=self.embedding,
                                **self.params)

