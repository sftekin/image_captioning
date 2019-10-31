import torch
import torch.nn as nn
import torch.optim as optim


class MultiNet(nn.Module):
    def __init__(self, params):
        super(MultiNet, self).__init__()
        self.params = params
        self.batch_size = params["batch_size"]

        self.conv_net = []
        conv_layers = params["cnn"]["layers"]
        conv_params = params["cnn"]["params"]
        conv_params = self.__parameter_completion(conv_layers, conv_params)
        self.__construct_network(conv_layers, conv_params)

        self.rnn_net = []
        self.H = []
        self.C = []
        self.last_state_size = None
        rnn_layers = params["rnn"]["layers"]
        rnn_params = params["rnn"]["params"]
        rnn_params = self.__parameter_completion(rnn_layers, rnn_params)
        self.__construct_network(rnn_layers, rnn_params)

        self.mlp_net = []
        mlp_layers = params["mlp"]["layers"]
        mlp_params = params["mlp"]["params"]
        mlp_params = self.__parameter_completion(mlp_layers, mlp_params)
        self.__construct_network(mlp_layers, mlp_params)

        self.optimizer = None
        self.criterion = None

    def forward(self, _input):
        cnn_output = _input.clone()

        for conv_layer in self.conv_net:
            cnn_output = conv_layer(cnn_output)

        rnn_output = cnn_output.view(self.batch_size, -1)
        for l, rnn_layer in enumerate(self.rnn_net):
            if self.H[l] is None or self.C[l] is None:
                rnn_output = rnn_layer(rnn_output)
            elif self.C[l] is None:
                state = self.H[l].clone()
                rnn_output, state = rnn_layer(rnn_output, state)
                self.H[l] = state.clone()
            else:
                state = self.H[l].clone(), self.C[l].clone()
                rnn_output, state = rnn_layer(rnn_output, state)
                self.H[l], self.C[l] = state[0].clone(), state[1].clone()
        return rnn_output

    def fit(self, x, y):

        y_hat = self.forward(x)


    def __parameter_completion(self, layer_names, layer_params):
        input_channels = self.params["image_channels"]
        input_size = self.batch_size
        in_features = self.last_state_size

        new_params = {}
        for layer_name, layer_param in zip(layer_names, layer_params):
            layer_type = layer_name.split[" "][0]
            if layer_type == "conv":
                layer_param.update({"in_channels": input_channels})
                input_channels = layer_param["out_channels"]

            elif layer_type in ["lstm", "rnn", "gru"]:
                layer_param.update({"input_size": input_size})
                input_size = layer_param["hidden_size"]

            elif layer_type == "mlp":
                layer_param.update({"in_features": in_features})
                in_features = layer_param["out_features"]

            new_params[layer_name] = layer_param
        return new_params

    def __construct_network(self, layer_names, layer_params):

        for layer_name, layer_param in zip(layer_names, layer_params):
            layer_type = layer_name.split[" "][0]
            self.layer_list.append(self.__construct_layer(layer_type, layer_param))

        return

    @staticmethod
    def __construct_layer(layer_type, params):
        layer_types = {"cnn": nn.Conv2d,
                       "lstm": nn.LSTM,
                       "rnn": nn.RNN,
                       "linear": nn.Linear,
                       "max_pool": nn.MaxPool2d,
                       "average_pool": nn.AvgPool2d,
                       "relu": nn.ReLU,
                       "leaky_relu": nn.LeakyReLU,
                       "softmax": nn.Softmax,
                       "sigmoid": nn.Sigmoid,
                       "softplus": nn.Softplus}

        return layer_types[layer_type](**params)

    def __init_states(self, rnn_layers, rnn_params):

        for layer, params in zip(rnn_layers, rnn_params):
            if layer in ["lstm", "gru", "rnn"]:
                self.H.append(torch.zeros(params["num_layers"], self.batch_size, params["hidden_size"]))
                self.last_state_size = params["hidden_size"]
                if layer == "lstm":
                    self.C.append(torch.zeros(params["num_layers"], self.batch_size, params["hidden_size"]))
                else:
                    self.C.append(None)
            else:
                self.H.append(None)
                self.C.append(None)
        return

    def __configure_train(self, optimizer_params, criterion_params):
        optimizer_types = {"adam": optim.Adam,
                           "sgd": optim.SGD,
                           "lr_schedule": optim.lr_scheduler}

        criterion_types = {"mse": nn.MSELoss,
                           "ce": nn.CrossEntropyLoss}

        self.optimizer = optimizer_types[optimizer_params["type"]](optimizer_params["params"])
        self.criterion = criterion_types[criterion_params["type"]](criterion_params["params"])
        return
