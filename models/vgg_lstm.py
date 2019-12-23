import torch
import torch.nn as nn
import torchvision.models as models

from torch.autograd import Variable


class VGG16(nn.Module):
    def __init__(self, model_params):
        super(VGG16, self).__init__()
        self.transfer_learn = model_params.get('transfer_learn', True)
        self.output_dim = model_params.get('output_dim', 512)
        self.pre_train = model_params.get('pre_train', True)

        # init model
        self.model = models.vgg16(pretrained=self.pre_train)
        self.__initialize_model()

    def __initialize_model(self):

        # close the parameters for training
        if self.transfer_learning:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.classifier[6].in_features

        # new trainable parameters are added to end of network
        self.model.classifier[6] = nn.Linear(num_features, self.output_dim)


class CaptionLSTM(nn.Module):
    def __init__(self, model_params, data_params, **kwargs):
        super(CaptionLSTM, self).__init__()
        self.drop_prob = model_params.get('drop_prob', 0.3)
        self.n_layers = model_params.get('n_layers', 2)
        self.n_hidden = model_params.get('n_hidden', 512)

        self.embed_dim = data_params.get('embed_dim', 300)
        self.vocab_dim = data_params.get('vocab_dim', 1004)

        self.embed_layer = nn.Embedding(self.vocab_dim, self.embed_dim)
        self.embed_layer.weight.data.uniform_(-1, 1)

        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            dropout=self.drop_prob,
                            batch_first=True)

        self.drop_out = nn.Dropout(self.drop_prob)

        self.fc = nn.Linear(self.n_hidden, self.vocab_dim)

        self.device = kwargs.get('device', 'cpu')

    def forward(self, x, hidden):
        """
        :param x: b, seq_len
        :param hidden: tuple((num_layers, b, n_hidden), (num_layers, b, n_hidden))
        :return:
        """
        embed = self.embed_layer(x)
        r_output, hidden = self.lstm(embed, hidden)

        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden)).to(self.device),
                  Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden)).to(self.device))
        return hidden


def train(net, data, epochs=10, batch_size=10, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    pass


if __name__ == '__main__':
    model_param = {
        'drop_prob': 0.3,
        'n_layers': 1,
        'n_hidden': 512,
        'embed_dim': 300,
        'vocab_dim': 1004,
    }
    a = CaptionLSTM({}, {})
    print(a.init_hidden(32))


