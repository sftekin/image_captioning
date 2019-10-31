from unittest import TestCase
from models.rnn import RNN
from models.mlp import MLP
import torch


class TestRNN(TestCase):
    def setUp(self):
        pass

    def test_init(self):
        state_encoder = MLP(params={"layers": ["linear 1", "relu 1", "linear 2", "relu 2"],
                                    "params": [{"out_features": 64}, {}] * 2,
                                    "in_features": 64})
        RNN(params={"layers": ["lstm 1", "lstm 2"], "params": [{"hidden_size": 64}] * 2, "batch_size": 32,
                    "in_features": 50}, state_encoder=state_encoder)

    def test_forward_dims(self):
        state_encoder = MLP(params={"layers": ["linear 1", "relu 1", "linear 2", "relu 2"],
                                    "params": [{"out_features": 64}, {}, {"out_features": 64}, {}],
                                    "in_features": 64})
        model = RNN(params={"layers": ["lstm 1", "lstm 2"], "params": [{"hidden_size": 64}] * 2, "batch_size": 32,
                            "in_features": 64}, state_encoder=state_encoder)

        input_data = torch.zeros((1, 32, 64))
        generated_sequence_raw = model(input_data)

        self.assertEqual(generated_sequence_raw.shape, (16, 32, 64))
