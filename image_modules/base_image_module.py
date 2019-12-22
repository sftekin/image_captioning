import torch
import torch.nn as nn


class BaseImageModule(nn.Module):
    def __init__(self, input_size, trainable, device):
        super(BaseImageModule, self).__init__()
        self.device = device
        self.trainable = trainable
        self.model = None
        self.input_size = input_size
        self.feature_dim = None

    def forward(self, images):
        """

        :param images: (b, d, m, n)
        :return: (b, f)
        """
        batch_size = images.shape[0]
        features = self.model(images)
        try:
            return features.view(batch_size, -1)
        except AttributeError:
            return features[0].view(batch_size, -1)

    def _get_feature_dim(self):
        self.model.fc = nn.Sequential().to(self.device)
        dummy_image = torch.zeros(1, 3, *self.input_size).to(self.device)
        with torch.no_grad():
            out = self.model(dummy_image)
            try:
                self.feature_dim = out.shape[1]*out.shape[2]*out.shape[3]
            except AttributeError:
                self.feature_dim = out[0].shape[0] * out[0].shape[1]

    def _set_classifier(self, output_dim):
        if not self.trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        self._get_feature_dim()
        classifier = nn.Linear(in_features=self.feature_dim, out_features=output_dim).to(self.device)
        self.model.classifier = classifier
        self.model.fc = classifier
