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

        self.layers = []
        self.output_channels = None
        conv_layers = params["layers"]
        conv_params = params["params"]
        conv_params = self.__parameter_completion(conv_layers, conv_params)

        self.output_size_reductions = []
        self.__construct_network(conv_layers, conv_params)

    def forward(self, _input):
        """
        Forward procedure that propagates the input image through the layers.
        :param _input: (B, D_i, M_i, N_i) input image as torch tensor.
        :return: (B, D_o, M_o, N_o) output image as torch tensor.
        """
        cnn_output = _input.clone()

        for conv_layer in self.layers:
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
            layer_type = layer_name.split(" ")[0]

            if layer_type == "conv":
                layer_param.update({"in_channels": input_channels})
                input_channels = layer_param["out_channels"]
                self.output_channels = input_channels

            new_params[layer_name] = layer_param
        return new_params

    def __construct_network(self, layer_names, layer_params):
        """
        Constructs the network by iteratively constructing every layer and stacking them.
        :param layer_names: Name of the layers that will be stacked.
        :param layer_params: Parameters of the layer corresponding to each level.
        :return: None
        """
        for layer_name in layer_names:
            layer_param = layer_params[layer_name]
            layer_type = layer_name.split(" ")[0]
            self.layers.append(self.__construct_layer(layer_type, layer_param))

        return

    def __construct_layer(self, layer_type, params):
        """
        Returns a layer from the layer_types dispatcher.
        :param layer_type: Type of the layer inferred from the layer name.
        :param params: Parameters of the current layer.
        :return: Layer object.
        """
        layer_types = {"conv": nn.Conv2d,
                       "max_pool": nn.MaxPool2d,
                       "average_pool": nn.AvgPool2d,
                       "relu": nn.ReLU,
                       "leaky_relu": nn.LeakyReLU,
                       "softmax": nn.Softmax,
                       "sigmoid": nn.Sigmoid,
                       "softplus": nn.Softplus}
        layer = layer_types[layer_type](**params)
        if layer_type in ["conv", "max_pool", "average_pool"]:
            self.__reduce_output_size(layer)
        return layer

    def __reduce_output_size(self, layer):
        f = layer.kernel_size
        p = layer.padding
        s = layer.stride

        self.output_size_reductions.append(lambda x, y: ((x - f[0] + 2*p[0])/s[0] + 1,
                                                         (y - f[1] + 2*p[1])/s[1] + 1))

    def output_size(self, input_size):
        current_size = input_size

        for reduction in self.output_size_reductions:
            current_size = reduction(*current_size)

        return current_size
