from unittest import TestCase
from models.cnn import CNN
import torch


class TestCNN(TestCase):
    def setUp(self):
        pass

    def test_init(self):
        CNN(params={"layers": ["conv 1", "relu 1", "conv 2", "relu 2"],
                    "params": [{"out_channels": 16, "kernel_size": 5, "padding": 2}, {},
                               {"out_channels": 8, "kernel_size": 3, "padding": 1}, {}],
                    "batch_size": 32,
                    "image_channels": 3})

    def test_forward_dims(self):
        model = CNN(params={"layers": ["conv 1", "relu 1", "conv 2", "relu 2"],
                            "params": [{"out_channels": 16, "kernel_size": 5, "padding": 2}, {},
                                       {"out_channels": 8, "kernel_size": 3, "padding": 1}, {}],
                            "batch_size": 32,
                            "image_channels": 3})

        input_data = torch.zeros((32, 3, 50, 50))
        feature_vector = model(input_data)

        self.assertEqual(feature_vector.shape, (32, 8, 50, 50))
