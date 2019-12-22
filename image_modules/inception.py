from torchvision.models import inception
from image_modules.base_image_module import BaseImageModule


class Inceptionv3(BaseImageModule):
    def __init__(self, **kwargs):
        output_dim = kwargs["word_length"]
        self.device = kwargs["device"]
        super(Inceptionv3, self).__init__(kwargs["input_size"], kwargs.get("trainable_cnn", False), self.device)
        self.model = inception.inception_v3(pretrained=kwargs.get("pretrained_cnn", True)).to(self.device)
        self.model.to(self.device)
        self._set_classifier(output_dim)
