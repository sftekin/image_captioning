from torchvision.models import vgg16, vgg19
from image_modules.base_image_module import BaseImageModule


class VGG16(BaseImageModule):
    def __init__(self, **kwargs):
        output_dim = kwargs["word_length"]
        super(VGG16, self).__init__(kwargs["input_size"], kwargs.get("trainable_cnn", True))
        self.model = vgg16(pretrained=kwargs.get("pretrained_cnn", False))
        self._set_classifier(output_dim)


class VGG19(BaseImageModule):
    def __init__(self, **kwargs):
        output_dim = kwargs["word_length"]
        super(VGG19, self).__init__(kwargs["input_size"], kwargs.get("trainable_cnn", True))
        self.model = vgg19(pretrained=kwargs.get("pretrained_cnn", False))
        self._set_classifier(output_dim)
