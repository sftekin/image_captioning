import torch
import torch.nn as nn
from torch.nn import init
from word_modules.multi_step_rnn import MultiStepRNN
from word_modules.multi_step_parallel import MultiStepParallel


class LSTM(MultiStepRNN):
    def __init__(self, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.num_layers = kwargs["num_layers"]
        self.model = nn.LSTM(input_size=self.embedding.vector_dim,
                             hidden_size=kwargs["hidden_size"],
                             num_layers=kwargs["num_layers"]).to(self.device)

    def _init_states(self, features):
        self.state = [features.unsqueeze(dim=0).repeat(self.num_layers, 1, 1),
                      torch.zeros_like(features.unsqueeze(dim=0)).repeat(self.num_layers, 1, 1).to(self.device)]


class LSTMParallel(MultiStepParallel):
    def __init__(self, **kwargs):
        super(LSTMParallel, self).__init__(**kwargs)
        self.hidden_size = kwargs["hidden_size"]
        self.num_layers = kwargs["num_layers"]
        self.model = nn.LSTM(input_size=self.embedding.vector_dim,
                             hidden_size=kwargs["hidden_size"],
                             num_layers=kwargs["num_layers"]).to(self.device)

    def _init_states(self, features):
        batch_size = features.shape[0]
        self.state = [torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)]


class LSTMCore(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(LSTMCore, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.W = nn.Parameter(torch.rand(hidden_size + input_size,
                                         4 * hidden_size),
                              requires_grad=True).to(self.device)
        self.b = nn.Parameter(torch.rand(1, 4 * hidden_size),
                              requires_grad=True).to(self.device)

    def forward(self, input_, state):
        state = [state[0].detach(), state[1].detach()]
        concat = torch.cat([input_, state[0]], dim=-1)
        gates = torch.add(torch.matmul(concat, self.W), self.b)
        mod_gates = torch.split(gates, self.hidden_size, dim=2)

        f_gate = nn.Softmax(dim=2)(mod_gates[0])
        i_gate = nn.Softmax(dim=2)(mod_gates[1])
        g_gate = nn.Tanh()(mod_gates[2])
        o_gate = nn.Softmax(dim=2)(mod_gates[3])

        m = torch.mul(i_gate, g_gate)
        c = torch.add(torch.mul(state[1], f_gate), m)
        h = torch.mul(nn.Tanh()(c), o_gate)

        return h, [h, c]


lstm_models = {"RNN": LSTM,
               "parallel": LSTMParallel}
