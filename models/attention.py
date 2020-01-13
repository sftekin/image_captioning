import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, conv_dim, lstm_dim, att_dim):
        super(Attention, self).__init__()
        self.conv_att = nn.Linear(conv_dim, att_dim)
        self.lstm_att = nn.Linear(lstm_dim, att_dim)
        self.full_att = nn.Linear(att_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, conv_out, lstm_hidden):
        """
        Forward propagation.
        :param conv_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param lstm_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.conv_att(conv_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.lstm_att(lstm_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (conv_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding
