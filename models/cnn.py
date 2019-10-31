"""Configurable CNN module implementation."""
import torch.nn as nn


class CNN(nn.Module):
    """
    Fully configurable CNN module. Can stack convolutional layers and activation layers
    to construct a convolutional neural network.
    """
    def __init__(self, params):
        """Convolutional Network initialization."""
        super(CNN, self).__init__()
        self.params = params
        self.batch_size = params["batch_size"]

        self.conv_net = []
        conv_layers = params["cnn"]["layers"]
        conv_params = params["cnn"]["params"]
        conv_params = self.__parameter_completion(conv_layers, conv_params)
        self.__construct_network(conv_layers, conv_params)

    def forward(self, _input):
        """
        Forward procedure that propagates the input image through the layers.
        :param _input: (B, D_i, M_i, N_i) input image as torch tensor.
        :return: (B, D_o, M_o, N_o) output image as torch tensor.
        """
        cnn_output = _input.clone()

        for conv_layer in self.conv_net:
            cnn_output = conv_layer(cnn_output)

        return cnn_output

    def __parameter_completion(self, layer_names, layer_params):
        """
        Fills the missing parameters that were not specified during the initialization
        of the network.
        :param layer_names: Name of the layers that will be stacked.
        :param layer_params: Parameters of the layer corresponding to each level.
        :return: Filled/corrected parameters.
        """
        input_channels = self.params["image_channels"]

        new_params = {}
        for layer_name, layer_param in zip(layer_names, layer_params):
            layer_type = layer_name.split[" "][0]

            if layer_type == "conv":
                layer_param.update({"in_channels": input_channels})
                input_channels = layer_param["out_channels"]

            new_params[layer_name] = layer_param
        return new_params

    def __construct_network(self, layer_names, layer_params):
        """
        Constructs the network by iteratively constructing every layer and stacking them.
        :param layer_names: Name of the layers that will be stacked.
        :param layer_params: Parameters of the layer corresponding to each level.
        :return: None
        """
        for layer_name, layer_param in zip(layer_names, layer_params):
            layer_type = layer_name.split[" "][0]
            self.layer_list.append(self.__construct_layer(layer_type, layer_param))

        return

    @staticmethod
    def __construct_layer(layer_type, params):
        """
        Returns a layer from the layer_types dispatcher.
        :param layer_type: Type of the layer inferred from the layer name.
        :param params: Parameters of the current layer.
        :return: Layer object.
        """
        layer_types = {"cnn": nn.Conv2d,
                       "max_pool": nn.MaxPool2d,
                       "average_pool": nn.AvgPool2d,
                       "relu": nn.ReLU,
                       "leaky_relu": nn.LeakyReLU,
                       "softmax": nn.Softmax,
                       "sigmoid": nn.Sigmoid,
                       "softplus": nn.Softplus}

        return layer_types[layer_type](**params)
