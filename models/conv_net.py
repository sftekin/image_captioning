import torch.nn as nn
import torch.optim as optim


class ConvolutionNet(nn.Module):

    def __init__(self, params):
        super(ConvolutionNet, self).__init__()

        self.layer_list = []
        self.params = params
        layer_names = params["layers"]
        layer_params = params["layer_params"]

        layer_params = self.__parameter_completion(layer_names, layer_params)
        self.__construct_network(layer_names, layer_params)

        self.optimizer = None
        self.criterion = None

    def forward(self, _input):

        output = _input.clone()

        for layer in self.layer_list:
            output = layer(output)

        return output

    @staticmethod
    def __construct_layer(layer_type, params):
        layer_types = {"conv": nn.Conv2d,
                       "max_pool": nn.MaxPool2d,
                       "average_pool": nn.AvgPool2d,
                       "relu": nn.ReLU,
                       "leaky_relu": nn.LeakyReLU,
                       "softmax": nn.Softmax,
                       "sigmoid": nn.Sigmoid,
                       "softplus": nn.Softplus}

        return layer_types[layer_type](params)

    def __construct_network(self, layer_names, layer_params):

        for layer_name, layer_param in zip(layer_names, layer_params):
            layer_type = layer_name.split[" "][0]
            self.layer_list.append(self.__construct_layer(layer_type, layer_param))

        return

    def __parameter_completion(self, layer_names, layer_params):
        input_channels = self.params["in_channels"]

        new_params = {}
        for layer_name, layer_param in zip(layer_names, layer_params):
            layer_type = layer_name.split[" "][0]
            if layer_type == "conv":
                layer_param.update({"in_channels": input_channels})
                input_channels = layer_param["out_channels"]

            new_params[layer_name] = layer_param
        return new_params

    def __configure_train(self, optimizer_params, criterion_params):
        optimizer_types = {"adam": optim.Adam,
                           "sgd": optim.SGD,
                           "lr_schedule": optim.lr_scheduler}

        criterion_types = {"mse": nn.MSELoss,
                           "ce": nn.CrossEntropyLoss}

        self.optimizer = optimizer_types[optimizer_params["type"]](optimizer_params["params"])
        self.criterion = criterion_types[criterion_params["type"]](criterion_params["params"])
        return
