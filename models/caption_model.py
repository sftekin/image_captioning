import torch
import torch.nn as nn

from models.embed import Embedding
from models.attention import Attention
from models.vgg import VGG16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CaptionLSTM(nn.Module):
    def __init__(self, model_params, int2word):
        super(CaptionLSTM, self).__init__()
        self.drop_prob = model_params.get('drop_prob', 0.3)
        self.n_layers = model_params.get('n_layers', 2)
        self.lstm_dim = model_params.get('lstm_dim', 512)
        self.conv_dim = model_params.get('conv_dim', 512)
        self.att_dim = model_params.get('att_dim', 512)

        self.embed_layer = Embedding(int2word)
        self.embed_dim = self.embed_layer.embed_dim
        self.vocab_dim = self.embed_layer.vocab_size

        self.conv_model = VGG16(model_params)
        self.attention = Attention(self.conv_dim, self.lstm_dim, self.att_dim)
        self.lstm = nn.LSTM(input_size=self.embed_dim + self.lstm_dim,
                            hidden_size=self.lstm_dim,
                            num_layers=self.n_layers,
                            dropout=self.drop_prob,
                            batch_first=True)
        self.drop_out = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(self.lstm_dim, self.vocab_dim)

        self.lin_h = nn.Linear(self.conv_dim, self.lstm_dim)
        self.lin_c = nn.Linear(self.conv_dim, self.lstm_dim)

    def forward(self, image, x_cap, sentence_len=17, hidden=None):
        """
        :param image:
        :param x_cap: b, seq_len
        :param sentence_len: int
        :param hidden: tuple((num_layers, b, n_hidden), (num_layers, b, n_hidden))
        :return:
        """
        batch_size = x_cap.shape[0]

        image_vec = self.conv_model(image)
        image_vec = image_vec.view(batch_size, -1, self.conv_dim)

        if hidden is None:
            h, c = self.init_hidden(image_vec)
            # expand it for each layer of image
            h = h.expand((self.n_layers, batch_size, self.lstm_dim)).contiguous()
            c = c.expand((self.n_layers, batch_size, self.lstm_dim)).contiguous()
        else:
            h, c = hidden

        sentence = []
        for word_idx in range(sentence_len):
            weigted_conv_output = self.attention(image_vec, h[0, :])
            embed = self.embed_layer(x_cap[:, word_idx]).float()
            lstm_in = torch.cat([embed, weigted_conv_output], dim=1).unsqueeze(1)
            r_out, (h, c) = self.lstm(lstm_in, (h, c))

            out = self.fc(self.drop_out(r_out))
            sentence.append(out)

        output = torch.cat(sentence, dim=1)
        output = output.contiguous().view(-1, self.vocab_dim)
        return output, (h, c)

    def init_hidden(self, image_vec):
        image_vec = image_vec.mean(dim=1)
        h = self.lin_h(image_vec)
        c = self.lin_c(image_vec)
        return h, c
