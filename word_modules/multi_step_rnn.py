import torch
import torch.nn as nn


class MultiStepRNN(nn.Module):
    def __init__(self, **kwargs):
        super(MultiStepRNN, self).__init__()
        self.embedding = kwargs["embedding"]
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
            self.state = self.state.detach() if not isinstance(self.state, list) else [self.state[0].detach(),
                                                                                       self.state[1].detach()]
            out, self.state = self.model(embedded_input, self.state)
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
        self.__init_states(features)

        first_word = self.embedding["x_START_"]
        embedded_input = self.embedding(first_word)

        sentence = []
        for l in range(self.sequence_length):
            out, self.state = self.model(embedded_input, self.state)
            out = nn.Softmax()(out)
            embedded_input = self.embedding(out)
            word = out.max(1)
            sentence.append(word.squeeze(dim=0))

        words = torch.stack(sentence, dim=1)
        return words

