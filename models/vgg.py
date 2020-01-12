import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, model_params):
        super(VGG16, self).__init__()
        self.transfer_learn = model_params.get('transfer_learn', True)
        self.pre_train = model_params.get('pre_train', True)

        # init model
        self.vgg = models.vgg16(pretrained=self.pre_train)

        modules = list(self.vgg.children())[:-2]
        self.vgg = nn.Sequential(*modules)

        self.initialize_model()

    def forward(self, image):
        return self.vgg(image)

    def initialize_model(self):
        # close the parameters for training
        if self.transfer_learn:
            for param in self.vgg.parameters():
                param.requires_grad = False

    def fine_tune(self):
        for c in list(self.vgg.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True
