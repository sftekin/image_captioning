
class Params:
    def __init__(self):
        pass

    def params(self):
        return self.__dict__


class DataParams(Params):
    def __init__(self):
        super(DataParams, self).__init__()

        self.data_path = {"image_path": "./dataset/images/",
                          "caption_path": "./dataset/imid.csv",
                          "sentence_path": "./dataset/captions.csv",
                          "code_dict_path": "./dataset/code_dictionary.csv"}

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

        self.embedding_params = {"layers": ["linear 0"],
                                 "params": [{"out_features": 500}]}

        self.cnn_params = {"image_channels": 3,
                           "layers": ["conv 0", "relu 0",
                                      "max_pool 1",
                                      "conv 2", "relu 2",
                                      "max_pool 2",
                                      "conv 4", "relu 4",
                                      "max_pool 1",
                                      "conv 6", "relu 6",
                                      "max_pool 2",
                                      "conv 8", "relu 8",
                                      "max_pool 1",
                                      "conv 10", "relu 10",
                                      "max_pool 2",
                                      "conv 12", "relu 12",
                                      "max_pool 3",
                                      "conv 14", "relu 14",
                                      "max_pool 4",
                                      "conv 16", "relu 16",
                                      "max_pool 5",
                                      "conv 18", "relu 18",
                                      "max_pool 6",
                                      "conv 20", "relu 20",
                                      "max_pool 7",
                                      "conv 22", "relu 22",
                                      "max_pool 8",
                                      "conv 24", "relu 24"],
                           "params": [{"out_channels": 16, "kernel_size": 13, "padding": 0}, {},
                                      {"kernel_size": 13, "stride": 1, "padding": 0},
                                      {"out_channels": 16, "kernel_size": 13, "padding": 0}, {},
                                      {"kernel_size": 13, "stride": 1, "padding": 0},
                                      {"out_channels": 16, "kernel_size": 13, "padding": 0}, {},
                                      {"kernel_size": 13, "stride": 1, "padding": 0},
                                      {"out_channels": 16, "kernel_size": 13, "padding": 0}, {},
                                      {"kernel_size": 13, "stride": 1, "padding": 0},
                                      {"out_channels": 16, "kernel_size": 13, "padding": 0}, {},
                                      {"kernel_size": 13, "stride": 1, "padding": 0},
                                      {"out_channels": 16, "kernel_size": 13, "padding": 0}, {},
                                      {"kernel_size": 13, "stride": 1, "padding": 0},
                                      {"out_channels": 8, "kernel_size": 11, "padding": 0}, {},
                                      {"kernel_size": 11, "stride": 1, "padding": 0},
                                      {"out_channels": 8, "kernel_size": 11, "padding": 0}, {},
                                      {"kernel_size": 11, "stride": 1, "padding": 0},
                                      {"out_channels": 8, "kernel_size": 9, "padding": 0}, {},
                                      {"kernel_size": 9, "stride": 1, "padding": 0},
                                      {"out_channels": 4, "kernel_size": 9, "padding": 0}, {},
                                      {"kernel_size": 9, "stride": 1, "padding": 0},
                                      {"out_channels": 4, "kernel_size": 7, "padding": 0}, {},
                                      {"kernel_size": 7, "stride": 1, "padding": 0},
                                      {"out_channels": 2, "kernel_size": 5, "padding": 0}, {},
                                      {"kernel_size": 5, "stride": 1, "padding": 0},
                                      {"out_channels": 1, "kernel_size": 3, "padding": 0}, {}]}

        self.rnn_params = {"layers": ["lstm 0", "lstm 1"],
                           "params": [{"hidden_size": 32},
                                      {"hidden_size": 32}]}

        self.state_mlp_params = {"layers": ["linear 0", "relu 0"],
                                 "params": [{"out_features": 32}, {}]}
        self.state_mlp_output_activation = "sigmoid"

        self.optimizer_type = "adam"
        self.optimizer_params = {"sgd": {},
                                 "adam": {},
                                 "lr_sch": {}}

        self.criterion_type = "cross_ent"
        self.criterion_params = {"mse": {},
                                 "cross_ent": {}}
