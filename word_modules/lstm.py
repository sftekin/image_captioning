import torch
import torch.nn as nn
from word_modules.multi_step_rnn import MultiStepRNN


class LSTM(MultiStepRNN):
    def __init__(self, **kwargs):
        super(LSTM, self).__init__(**kwargs)

        self.model = nn.LSTM(input_size=self.embedding.vector_dim,
                             hidden_size=kwargs["hidden_size"],
                             num_layers=kwargs["num_layers"]).to(self.device)

    def _init_states(self, features):
        self.state = [features.unsqueeze(dim=0),
                      torch.zeros_like(features.unsqueeze(dim=0)).to(self.device)]
