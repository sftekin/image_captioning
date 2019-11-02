
class Params:
    def __init__(self):
        pass

    def params(self):
        return self.__dict__


class DataParams(Params):
    def __init__(self):
        super(DataParams, self).__init__()

        self.train_length = []
        self.validation_length = []
        self.test_length = []


class CNNLSTMParams(Params):
    def __init__(self):
        super(CNNLSTMParams, self).__init__()

        self.sequence_length = 16
        self.word_length = 128
        self.batch_size = 32
        self.input_size = 12, 17
        self.cnn_params = {"image_channels": 3,
                           "layers": ["conv 0", "relu 0", "conv 1", "relu 1", "conv 2", "relu 2"],
                           "params": [{"out_channels": 10, "kernel_size": 5, "padding": 1}, {},
                                      {"out_channels": 5, "kernel_size": 3, "padding": 0}, {},
                                      {"out_channels": 1, "kernel_size": 3, "padding": 0}, {}]}

        self.rnn_params = {"layers": ["lstm 0", "lstm 1"],
                           "params": [{"hidden_size": 32},
                                      {"hidden_size": 32}]}

        self.state_mlp_params = {"layers": ["linear 0", "relu 0", "linear 1", "sigmoid 1"],
                                 "params": [{"out_features": 32}, {},
                                            {"out_features": self.word_length}, {}]}
