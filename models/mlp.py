"""Configurable MLP module implementation."""
import torch.nn as nn


class MLP(nn.Module):
    """
    Fully configurable MLP module. Can stack linear layers and activation layers
    to construct a MLP network.
    """
    def __init__(self, params):
        """MLP Network initialization."""
        super(MLP, self).__init__()
        self.params = params

        self.layers = []
        self.output_feature_len = None
        mlp_layers = params["layers"]
        mlp_params = params["params"]
        mlp_params = self.__parameter_completion(mlp_layers, mlp_params)
        self.__construct_network(mlp_layers, mlp_params)

    def forward(self, _input):
        """
        Forward procedure that propagates the input vector through the layers.
        :param _input: (B, F) input vector as torch tensor.
        :return: (B, D) output vector as torch tensor.
        """
        layer_output = _input.clone()

        for mlp_layer in self.layers:
            layer_output = mlp_layer(layer_output)

        return layer_output

    def __parameter_completion(self, layer_names, layer_params):
        """
        Fills the missing parameters that were not specified during the initialization
        of the network.
        :param layer_names: Name of the layers that will be stacked.
        :param layer_params: Parameters of the layer corresponding to each level.
        :return: Filled/corrected parameters.
        """
        in_features = self.params["in_features"]

        new_params = {}
        for layer_name, layer_param in zip(layer_names, layer_params):
            layer_type = layer_name.split(" ")[0]
            if layer_type == "linear":
                layer_param.update({"in_features": in_features})
                in_features = layer_param["out_features"]
                self.output_feature_len = layer_param["out_features"]

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

    @staticmethod
    def __construct_layer(layer_type, params):
        """
        Returns a layer from the layer_types dispatcher.
        :param layer_type: Type of the layer inferred from the layer name.
        :param params: Parameters of the current layer.
        :return: Layer object.
        """
        layer_types = {"linear": nn.Linear,
                       "relu": nn.ReLU,
                       "leaky_relu": nn.LeakyReLU,
                       "softmax": nn.Softmax,
                       "sigmoid": nn.Sigmoid,
                       "softplus": nn.Softplus}

        return layer_types[layer_type](**params)
