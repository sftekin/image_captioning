import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19


class VGG(nn.Module):
    def __init__(self, input_size):
        super(VGG, self).__init__()
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
        return features.view(batch_size, -1)

    def _get_feature_dim(self):
        dummy_image = torch.zeros(1, 3, *self.input_size)
        with torch.no_grad():
            out = self.model.features(dummy_image)
            self.feature_dim = out.shape[1]*out.shape[2]*out.shape[3]

    def _set_classifier(self, output_dim):
        self._get_feature_dim()
        classifier = nn.Linear(in_features=self.feature_dim, out_features=output_dim)
        self.model.classifier = classifier


class VGG16(VGG):
    def __init__(self, output_dim, input_size, pretrained=False):
        super(VGG16, self).__init__(input_size)
        self.model = vgg16(pretrained=pretrained)
        self._set_classifier(output_dim)


class VGG19(VGG):
    def __init__(self, output_dim, input_size, pretrained=False):
        super(VGG19, self).__init__(input_size)
        self.model = vgg19(pretrained=pretrained)
        self._set_classifier(output_dim)
