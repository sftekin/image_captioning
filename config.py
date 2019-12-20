
class Params:
    def __init__(self):
        pass

    def params(self):
        return self.__dict__


class DataParams(Params):
    def __init__(self):
        super(DataParams, self).__init__()

        self.model_name = "inceptionrnn"

        self.image_path = "./dataset/images/"
        self.dataset_path = "./dataset"
        self.url_path = "./dataset/img_url.csv"

        self.num_epochs = 10
        self.batch_size = 100
        self.sequence_length = 16
        self.word_length = 1004
        self.input_size = (480, 480)
        self.min_num_captions = 3

        self.train_length = []
        self.validation_length = []
        self.test_length = []

        self.hidden_size = self.word_length


class VggRNNParams(Params):
    def __init__(self):
        super(VggRNNParams, self).__init__()
        self.pretrained_cnn = False
        self.trainable_cnn = False

        self.num_layers = 1

        self.optimizer_type = "ADAM"
        self.optimizer_params = {"lr": 0.001}

        self.criterion_type = "CE"
        self.criterion_params = {}


class InceptionRNNParams(Params):
    def __init__(self):
        super(InceptionRNNParams, self).__init__()
        self.pretrained_cnn = True
        self.trainable_cnn = False

        self.num_layers = 1

        self.optimizer_type = "ADAM"
        self.optimizer_params = {"lr": 0.001}

        self.criterion_type = "CE"
        self.criterion_params = {}
