import torch
import torch.nn as nn


class MultiStepRNN(nn.Module):
    def __init__(self, **kwargs):
        super(MultiStepRNN, self).__init__()
        self.device = kwargs["device"]
        self.embedding = kwargs["embedding"]
        self.word_flow = kwargs["word_flow"]
        self.encoder = nn.Sequential(nn.Linear(kwargs["word_length"],
                                               kwargs["word_length"]).to(self.device),
                                     nn.Softmax(dim=2))
        self.input_size = kwargs["word_length"]

        self.state = None
        self.model = None

        self.sequence_length = kwargs["sequence_length"]

    def forward(self, features, batch_y=None):
        """

        :param features: (b, f)
        :param batch_y: (b, l)
        :return: (b, l, w)
        """
        if self.word_flow:
            return self.word_forward(features, batch_y)

        self._init_states(features)
        batch_size = features.shape[0]
        embedded_input = self.embedding["x_START_"].unsqueeze(dim=0).repeat(batch_size, 1).unsqueeze(0)

        words = []
        for l in range(self.sequence_length):
            out, self.state = self.model(embedded_input, self.state)
            word_vec = self.encoder(out)
            embedded_input = self.embedding(word_vec)
            words.append(word_vec.squeeze(dim=0))

        words = torch.stack(words, dim=1)
        return words

    def word_forward(self, features, batch_y):
        """

        :param features: (b, f)
        :param batch_y: (b, l, w)
        :return: (b, l, w)
        """
        self._init_states(features)
        batch_size = features.shape[0]
        embedded_input = self.embedding["x_START_"].unsqueeze(dim=0).repeat(batch_size, 1).unsqueeze(0)

        words = []
        for l in range(self.sequence_length):
            out, self.state = self.model(embedded_input, self.state)
            word_vec = self.encoder(out)
            if batch_y is not None:
                new_words = batch_y[:, l + 1][0]
            else:
                new_words = torch.argmax(word_vec, dim=2)[0]
            word_strs = [self.embedding.int2word[int(w)] for w in new_words]
            embedded_input = torch.stack([self.embedding[word_name].unsqueeze(dim=0)
                                          for word_name in word_strs], dim=1)
            words.append(word_vec.squeeze(dim=0))

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
            embedded_input = self.embedding(out)
            word = torch.argmax(out, dim=2)
            sentence.append(word.squeeze(dim=0))

        words = torch.stack(sentence, dim=1)
        return words

