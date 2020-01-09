import torch
import torch.nn as nn
import torchvision.models as models

from torch.autograd import Variable
from models.embed import Embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGG16(nn.Module):
    def __init__(self, model_params):
        super(VGG16, self).__init__()
        self.transfer_learn = model_params.get('transfer_learn', True)
        self.output_dim = model_params.get('n_hidden', 512)
        self.pre_train = model_params.get('pre_train', True)

        # init model
        self.model = models.vgg16(pretrained=self.pre_train)
        self.__initialize_model()

    def forward(self, image):
        return self.model(image)

    def __initialize_model(self):
        # close the parameters for training
        if self.transfer_learn:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.classifier[6].in_features

        # new trainable parameters are added to end of network
        self.model.classifier[6] = nn.Linear(num_features, self.output_dim)
        return self.model


class CaptionLSTM(nn.Module):
    def __init__(self, model_params, int2word):
        super(CaptionLSTM, self).__init__()
        self.drop_prob = model_params.get('drop_prob', 0.3)
        self.n_layers = model_params.get('n_layers', 2)
        self.n_hidden = model_params.get('n_hidden', 512)

        self.embed_layer = Embedding(int2word)
        self.embed_dim = self.embed_layer.embed_dim
        self.vocab_dim = self.embed_layer.vocab_size

        self.conv_model = VGG16(model_params)
        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            dropout=self.drop_prob,
                            batch_first=True)
        self.drop_out = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(self.n_hidden, self.vocab_dim)

    def forward(self, image, x_cap, hidden):
        """
        :param image:
        :param x_cap: b, seq_len
        :param hidden: tuple((num_layers, b, n_hidden), (num_layers, b, n_hidden))
        :return:
        """
        image_vec = self.conv_model(image)
        h, c = image_vec, image_vec

        # expand it for each layer of image
        h = h.expand(hidden[0].shape).contiguous()
        c = c.expand(hidden[1].shape).contiguous()

        embed = self.embed_layer(x_cap).float()
        r_output, hidden = self.lstm(embed, (h, c))

        out = self.drop_out(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (Variable(torch.rand(self.n_layers, batch_size, self.n_hidden)).to(device),
                  Variable(torch.rand(self.n_layers, batch_size, self.n_hidden)).to(device))
        return hidden


