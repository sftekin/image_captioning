import torch
import torch.nn as nn


class MultiStepRNN(nn.Module):
    def __init__(self, **kwargs):
        super(MultiStepRNN, self).__init__()
        self.device = kwargs["device"]
        self.embedding = kwargs["embedding"]
        self.encoder = nn.Linear(kwargs["word_length"], kwargs["word_length"]).to(self.device)
        self.input_size = kwargs["word_length"]

        self.state = None
        self.model = None

        self.sequence_length = kwargs["sequence_length"]

    def forward(self, features):
        """

        :param features: (b, f)
        :return: (b, l, w)
        """
        self._init_states(features)
        batch_size = features.shape[0]
        embedded_input = self.embedding["x_START_"].unsqueeze(dim=0).repeat(batch_size, 1).unsqueeze(0)

        words = []
        for l in range(self.sequence_length):
            out, self.state = self.model(embedded_input, self.state)
            out = self.encoder(out)
            out = nn.Softmax(dim=2)(out)
            embedded_input = self.embedding(out)
            words.append(out.squeeze(dim=0))

        words = torch.stack(words, dim=1)
        return words

    def caption(self, features):
        """

        :param features: (b, f)
        :return: (b, l, w)
        """
        self._init_states(features)
        batch_size = features.shape[0]

        embedded_input = self.embedding["x_START_"].unsqueeze(dim=0).repeat(batch_size, 1).unsqueeze(0)

        sentence = []
        for l in range(self.sequence_length):
            out, self.state = self.model(embedded_input, self.state)
            out = self.encoder(out)
            out = nn.Softmax(dim=2)(out)
            embedded_input = self.embedding(out)
            word = torch.argmax(out, dim=2)
            sentence.append(word.squeeze(dim=0))

        words = torch.stack(sentence, dim=1)
        return words

