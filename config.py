
class Params:
    def __init__(self):
        pass

    def params(self):
        return self.__dict__


class DataParams(Params):
    def __init__(self):
        super(DataParams, self).__init__()

        self.image_path = "./dataset/images/"
        self.dataset_path = "./dataset"
        self.url_path = "./dataset/img_url.csv"

        self.batch_size = 64
        self.sequence_length = 16
        self.word_length = 1004
        self.input_size = (224, 224)
        self.min_num_captions = 3

        self.train_length = []
        self.validation_length = []
        self.test_length = []

        self.hidden_size = self.word_length


class CNNLSTMParams(Params):
    def __init__(self):
        super(CNNLSTMParams, self).__init__()
        self.pretrained_cnn = False

        self.num_layers = 1

        self.optimizer_type = "ADAM"
        self.optimizer_params = {"lr": 0.01}

        self.criterion_type = "CE"
        self.criterion_params = {}
