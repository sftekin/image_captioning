"""Fully configurable RNN network."""
import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    Fully configurable RNN module. Can stack recurrent layers and activation layers
    to construct a RNN network. RNN layer can be GRU, LSTM or simple RNN.
    """
    def __init__(self, params):
        """
        Initialization of the network.
        :param params: Parameter dictionary for the RNN layer.
        """
        super(RNN, self).__init__()
        self.params = params
        self.batch_size = params["batch_size"]
        self.in_features = params["in_features"]

        self.word_length = params["word_length"]
        self.sequence_length = params["sequence_length"]
        self.state_encoder = None

        self.layers = []
        self.output_feature_len = None
        
        self.H = []
        self.C = []
        self.last_state_size = None
        rnn_layers = params["layers"]
        rnn_params = params["params"]
        rnn_params = self.__parameter_completion(rnn_layers, rnn_params)
        self.__construct_network(rnn_layers, rnn_params)
        self.__init_states(rnn_layers, rnn_params)

    def forward(self, input_):
        """
        Forward procedure of the model. Input tensor is passed to the first and for the rest
        of the network, state of the previous layer is passed as input.
        :param input_: Input tensor (starts with the START word in first layer).
        :return: Generated sequence.
        """
        rnn_output = torch.zeros(1, self.batch_size, self.word_length)

        self.__reset_states()
        self.H[0] = input_.clone()
        self.C[0] = torch.zeros_like(self.H[0]) if self.C[0] is not None else None

        generated_sequence = []
        for t in range(self.sequence_length):
            for l, rnn_layer in enumerate(self.layers):
                state = self.H[l].clone(), self.C[l].clone()
                if l is not 0:
                    rnn_output = self.H[l-1]

                if self.H[l] is None or self.C[l] is None:
                    rnn_output = rnn_layer(rnn_output)
                elif self.C[l] is None:
                    rnn_output, state = rnn_layer(rnn_output, state)
                    self.H[l] = state.clone()
                else:
                    rnn_output, state = rnn_layer(rnn_output, state)
                    self.H[l], self.C[l] = state[0].clone(), state[1].clone()

            last_layer_state = self.H[-1].clone()
            rnn_output = self.state_encoder(last_layer_state)
            generated_sequence.append(rnn_output)

        return torch.cat(generated_sequence, dim=0)

    def __reset_states(self):
        """
        Resets the Recurrent layers states to zero vector.
        :return: None
        """
        for l in range(len(self.H)):
            self.H[l] = torch.zeros_like(self.H[l])
            if self.C[l] is not None:
                self.C[l] = torch.zeros_like(self.C[l])
        return

    def __parameter_completion(self, layer_names, layer_params):
        """
        Fills the missing parameters that were not specified during the initialization
        of the network.
        :param layer_names: Name of the layers that will be stacked.
        :param layer_params: Parameters of the layer corresponding to each level.
        :return: Filled/corrected parameters.
        """
        input_size = self.word_length

        new_params = {}
        first_rnn_layer = 0
        for layer_name, layer_param in zip(layer_names, layer_params):
            layer_type = layer_name.split(" ")[0]

            if layer_type in ["lstm", "rnn", "gru"]:
                first_rnn_layer += 1
                if first_rnn_layer is 1:
                    layer_param.update({"hidden_size": self.in_features})
                layer_param.update({"input_size": input_size})
                input_size = layer_param["hidden_size"]

            new_params[layer_name] = layer_param.copy()
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
        layer_types = {"lstm": nn.LSTM,
                       "rnn": nn.RNN,
                       "max_pool": nn.MaxPool2d,
                       "average_pool": nn.AvgPool2d,
                       "relu": nn.ReLU,
                       "leaky_relu": nn.LeakyReLU,
                       "softmax": nn.Softmax,
                       "sigmoid": nn.Sigmoid,
                       "softplus": nn.Softplus}

        return layer_types[layer_type](**params)

    def __init_states(self, rnn_layers, rnn_params):
        """
        Constructs the inherent memory vectors of the recurrent layers.
        :param rnn_layers: Name of the rnn layers.
        :param rnn_params: Parameters for the RNN layers.
        :return: None
        """
        for layer in rnn_layers:
            params = rnn_params[layer]
            layer_type = layer.split(" ")[0]
            if layer_type in ["lstm", "gru", "rnn"]:
                self.H.append(torch.zeros(1, self.batch_size, params["hidden_size"]))
                self.output_feature_len = params["hidden_size"]
                self.last_state_size = params["hidden_size"]
                if layer_type == "lstm":
                    self.C.append(torch.zeros(1, self.batch_size, params["hidden_size"]))
                else:
                    self.C.append(None)
            else:
                self.H.append(None)
                self.C.append(None)
        return

    def set_encoder(self, state_encoder):
        """
        :param state_encoder: Encoder MLP that converts the state of the last layer to
                              an output.
        :return:
        """
        self.state_encoder = state_encoder
