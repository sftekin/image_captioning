import torch
import config
from models import cnn_lstm
from unittest import TestCase


class TestCnnLstm(TestCase):
    def setUp(self):
        pass

    def test_init(self):
        params = config.CNNLSTMParams()
        params = params.params()

        _ = cnn_lstm.CNNLSTM(params=params)

    def test_forward(self):
        params = config.CNNLSTMParams()
        params = params.params()

        model = cnn_lstm.CNNLSTM(params=params)

        fake_image_batch = torch.randn(size=(params["batch_size"],
                                             3, params["input_size"][0],
                                             params["input_size"][1]))
        generated_sequence = model(fake_image_batch)

        pass
