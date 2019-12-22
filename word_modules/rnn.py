import torch
import torch.nn as nn
from word_modules.multi_step_rnn import MultiStepRNN
from word_modules.multi_step_parallel import MultiStepParallel


class RNN(MultiStepRNN):
    def __init__(self, **kwargs):
        super(RNN, self).__init__(**kwargs)

        self.model = nn.RNN(input_size=self.embedding.vector_dim,
                            hidden_size=kwargs["hidden_size"],
                            num_layers=kwargs["num_layers"]).to(self.device)

    def _init_states(self, features):
        self.state = features.unsqueeze(dim=0)


class RNNParallel(MultiStepParallel):
    def __init__(self, **kwargs):
        super(RNNParallel, self).__init__(**kwargs)

        self.model = nn.RNN(input_size=self.embedding.vector_dim,
                            hidden_size=kwargs["hidden_size"],
                            num_layers=kwargs["num_layers"]).to(self.device)

    def _init_states(self, features):
        batch_size = features.shape[0]
        self.state = torch.zeros(1, batch_size, self.hidden_size).to(self.device)


rnn_models = {"RNN": RNN,
              "Parallel": RNNParallel}
