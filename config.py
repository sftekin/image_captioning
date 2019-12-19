
class Params:
    def __init__(self):
        pass

    def params(self):
        return self.__dict__


class DataParams(Params):
    def __init__(self):
        super(DataParams, self).__init__()

        self.data_path = {"image_path": "./dataset/images/",
                          "caption_path": "./dataset"}

        self.batch_size = 8
        self.sequence_length = 16
        self.word_length = 1004
        self.input_size = (240, 320)
        self.min_num_captions = 3

        self.train_length = []
        self.validation_length = []
        self.test_length = []


class CNNLSTMParams(Params):
    def __init__(self):
        super(CNNLSTMParams, self).__init__()
        self.pretrained_cnn = False

        self.optimizer_type = "ADAM"
        self.optimizer_params = {}

        self.criterion_type = "MSE"
        self.criterion_params = {}
