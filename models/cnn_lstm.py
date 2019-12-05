import torch
import torch.nn as nn
from models.cnn import CNN
from models.rnn import RNN
from models.mlp import MLP
from base_model import BaseModel


class CNNLSTM(BaseModel):
    def __init__(self, data_params, params, code_dictionary):
        super(CNNLSTM, self).__init__(data_params, params, code_dictionary)

        embedding_params = params["embedding_params"]
        embedding_params.update({"batch_size": data_params["batch_size"],
                                 "in_features": data_params["word_length"]})
        self.embedding = MLP(params=embedding_params)

        cnn_params = params["cnn_params"]
        cnn_params.update({"batch_size": data_params["batch_size"]})
        self.conv_net = CNN(params=cnn_params)

        output_channels = self.conv_net.output_channels
        output_size = self.conv_net.output_size(self.input_size)
        visual_feature_len = int(output_channels * output_size[0] * output_size[1])

        rnn_params = params["rnn_params"]
        rnn_params.update({"batch_size": self.batch_size,
                           "in_features": visual_feature_len,
                           "word_length": data_params["word_length"],
                           "embed_length": embedding_params["params"][-1]["out_features"],
                           "sequence_length": data_params["sequence_length"]})
        self.rnn_net = RNN(params=rnn_params)
        rnn_feature_len = self.rnn_net.output_feature_len

        state_encoder_params = params["state_mlp_params"]
        state_encoder_params.update({"batch_size": self.batch_size,
                                     "in_features": rnn_feature_len})
        state_encoder_params["layers"].extend(["linear -1",
                                               params["state_mlp_output_activation"] + " -1"])
        state_encoder_params["params"].extend([{"out_features": data_params["word_length"]}, {}])

        state_encoder = MLP(params=state_encoder_params)
        self.rnn_net.set_encoder(state_encoder, self.embedding)

        self.modules = nn.ModuleList([self.conv_net, self.rnn_net,
                                      state_encoder])

        self.construct_optimizer()

    def forward(self, input_):
        for l in range(len(self.rnn_net.H)):
            self.rnn_net.H[l] = self.rnn_net.H[l].detach() if self.rnn_net.H is not None else None
            self.rnn_net.C[l] = self.rnn_net.C[l].detach() if self.rnn_net.C is not None else None

        visual_features = self.conv_net(input_)
        flattened_features = visual_features.view((1, self.batch_size, -1))
        temporal_sequence = self.rnn_net(flattened_features)

        output_words = []
        for decoded_word in temporal_sequence:
            output_words.append(decoded_word.unsqueeze(dim=1))

        return torch.cat(output_words, dim=1)
