import torch
import torch.nn as nn
from word_modules.multi_step_rnn import MultiStepRNN
from word_modules.multi_step_parallel import MultiStepParallel


class LSTM(MultiStepRNN):
    def __init__(self, **kwargs):
        super(LSTM, self).__init__(**kwargs)

        self.model = nn.LSTM(input_size=self.embedding.vector_dim,
                             hidden_size=kwargs["hidden_size"],
                             num_layers=kwargs["num_layers"]).to(self.device)

    def _init_states(self, features):
        self.state = [features.unsqueeze(dim=0),
                      torch.zeros_like(features.unsqueeze(dim=0)).to(self.device)]


class LSTMParallel(MultiStepParallel):
    def __init__(self, **kwargs):
        super(LSTMParallel, self).__init__(**kwargs)
        self.hidden_size = kwargs["hidden_size"]
        self.model = nn.LSTM(input_size=self.embedding.vector_dim,
                             hidden_size=kwargs["hidden_size"],
                             num_layers=kwargs["num_layers"]).to(self.device)

    def _init_states(self, features):
        batch_size = features.shape[0]
        self.state = [torch.zeros(1, batch_size, self.hidden_size).to(self.device),
                      torch.zeros(1, batch_size, self.hidden_size).to(self.device)]


lstm_models = {"RNN": LSTM,
               "parallel": LSTMParallel}
