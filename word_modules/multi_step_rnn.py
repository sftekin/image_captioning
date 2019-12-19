import torch
import torch.nn as nn


class MultiStepRNN(nn.Module):
    def __init__(self, embedding, sequence_length):
        super(MultiStepRNN, self).__init__()
        self.embedding = embedding

        self.state = None
        self.model = None

        self.sequence_length = sequence_length

    def forward(self, features):
        """

        :param features: (b, f)
        :return: (b, l, w)
        """
        self.__init_states(features)

        first_word = self.embedding["x_START_"]
        embedded_input = self.embedding(first_word)

        words = []
        for l in range(self.sequence_length):
            out, self.state = self.model(embedded_input, self.state)
            word = self.embedding(out)
            words.append(word)

        words = torch.stack(words, dim=1)
        return words

    def caption(self, features):
        """

        :param features: (b, f)
        :return: (b, l, w)
        """
        self.__init_states(features)

        first_word = self.embedding["x_START_"]
        embedded_input = self.embedding(first_word)

        sentence = []
        for l in range(self.sequence_length):
            out, self.state = self.model(embedded_input, self.state)
            word = self.embedding(out).max(1)
            sentence.append(word)

        words = torch.stack(sentence, dim=1)
        return words


class RNN(MultiStepRNN):
    def __init__(self, **kwargs):
        super(RNN, self).__init__(kwargs["embedding"], kwargs["sequence_length"])

        self.model = nn.RNN(input_size=kwargs["input_size"],
                            hidden_size=kwargs["hidden_size"],
                            num_layers=kwargs["num_layers"])

    def __init_states(self, features):
        self.state = features
