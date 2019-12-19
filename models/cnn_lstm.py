from base_model import BaseModel
from image_modules.vgg import VGG16
from word_modules.multi_step_rnn import RNN


class CNNLSTM(BaseModel):
    def __init__(self, params):
        super(CNNLSTM, self).__init__(params)

    def _construct_model(self):
        self.image_process = VGG16(self.params["word_length"],
                                   self.params["pretrained_cnn"])

        self.embedding = self.params["embedding"]

        self.word_process = RNN(embedding=self.embedding,
                                sequence_length=self.params["sequence_length"])

