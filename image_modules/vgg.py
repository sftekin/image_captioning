from torchvision.models import vgg16, vgg19
from image_modules.base_image_module import BaseImageModule


class VGG16(BaseImageModule):
    def __init__(self, **kwargs):
        output_dim = kwargs["word_length"]
        self.rnn_flow = kwargs["rnn_flow"]
        super(VGG16, self).__init__(kwargs["input_size"],
                                    kwargs.get("trainable_cnn", True),
                                    device=kwargs["device"])
        self.model = vgg16(pretrained=kwargs.get("pretrained_cnn", False)).to(kwargs["device"])
        self._set_classifier(output_dim)


class VGG19(BaseImageModule):
    def __init__(self, **kwargs):
        output_dim = kwargs["word_length"]
        self.rnn_flow = kwargs["rnn_flow"]
        super(VGG19, self).__init__(kwargs["input_size"],
                                    kwargs.get("trainable_cnn", True),
                                    device=kwargs["device"])
        self.model = vgg19(pretrained=kwargs.get("pretrained_cnn", False)).to(kwargs["device"])
        self._set_classifier(output_dim)
