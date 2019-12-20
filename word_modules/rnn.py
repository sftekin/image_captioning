import torch.nn as nn
from word_modules.multi_step_rnn import MultiStepRNN


class RNN(MultiStepRNN):
    def __init__(self, **kwargs):
        super(RNN, self).__init__(**kwargs)

        self.model = nn.RNN(input_size=self.embedding.vector_dim,
                            hidden_size=kwargs["hidden_size"],
                            num_layers=kwargs["num_layers"])

    def _init_states(self, features):
        self.state = features.unsqueeze(dim=0)
