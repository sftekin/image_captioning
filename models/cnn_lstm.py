import torch
import torch.nn as nn
import torch.optim as opt
from models.cnn import CNN
from models.rnn import RNN
from models.mlp import MLP


class CNNLSTM(nn.Module):
    def __init__(self, params):
        super(CNNLSTM, self).__init__()
        self.params = params
        self.input_size = params["input_size"]

        cnn_params = params["cnn_params"]
        cnn_params.update({"batch_size": params["batch_size"]})
        self.conv_net = CNN(params=cnn_params)

        output_channels = self.conv_net.output_channels
        output_size = self.conv_net.output_size(self.input_size)
        visual_feature_len = int(output_channels * output_size[0] * output_size[1])

        rnn_params = params["rnn_params"]
        rnn_params.update({"batch_size": params["batch_size"],
                           "in_features": visual_feature_len,
                           "word_length": params["word_length"],
                           "sequence_length": params["sequence_length"]})
        self.rnn_net = RNN(params=rnn_params)
        rnn_feature_len = self.rnn_net.output_feature_len

        state_encoder_params = params["state_mlp_params"]
        state_encoder_params.update({"batch_size": params["batch_size"],
                                     "in_features": rnn_feature_len})
        state_encoder = MLP(params=state_encoder_params)
        self.rnn_net.set_encoder(state_encoder)

        self.optimizer = None
        self.criterion = None

    def forward(self, input_):
        visual_features = self.conv_net(input_)
        flattened_features = visual_features.view((1, self.params["batch_size"], -1))
        temporal_sequence = self.rnn_net(flattened_features)

        output_words = []
        for decoded_word in temporal_sequence:
            output_words.append(decoded_word.unsqueeze(dim=0))

        return torch.cat(output_words, dim=0)

    def fit(self, x, y):
        y_hat = self(x)

        def closure():
            self.optimizer.zero_grad()
            loss = self.criterion(y_hat, y)
            loss.backward()
            return loss

        self.optimizer.step(closure)

    def __construct_optimizer(self, optimizer_type, criterion_type):
        optimizer_dict = {"sgd": opt.SGD,
                          "adam": opt.Adam,
                          "lr_sch": opt.lr_scheduler}

        criterion_dict = {"mse": nn.MSELoss,
                          "cross_ent": nn.CrossEntropyLoss}

        self.optimizer = optimizer_dict[optimizer_type]()
        self.criterion = criterion_dict[criterion_type]()
