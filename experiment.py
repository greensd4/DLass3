import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

BATCH = 100


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        # self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        #tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return lstm_out,self.hidden



    def getDataset(data_file, is_train = True, separator = " "):
        """
        get data as windows saved in data loader
        :param data_file: path to file
        :param is_train: is train routine
        :param separator: seprator in files
        :return: data loader
        """
        print "Getting data from: ", data_file
        sentences = utils.read_data(data_file, is_train=True, seperator=separator)
        if is_train:
            utils.initialize_indexes()
        windows, tags = utils.create_windows(sentences)
        windows, tags = np.asarray(windows, np.float32), np.asarray(tags, np.int32)
        windows, tags = torch.from_numpy(windows), torch.from_numpy(tags)
        windows, tags = windows.type(torch.LongTensor), tags.type(torch.LongTensor)
        dataset = torch.utils.data.TensorDataset(windows, tags)
        if is_train:
            return DataLoader(dataset, batch_size=BATCH, shuffle=True)
        return DataLoader(dataset, batch_size=1, shuffle=True)


class utils:

    C2I = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
           "7": 7, "8": 8, "9": 9, "a": 10, "b": 11, "c": 12, "d": 13}



