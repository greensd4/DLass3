import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import utils as ut
import torch.functional as F
import torch.optim as optim

STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}


# Globals
EMBEDDING_ROW_LENGTH = 50
WINDOWS_SIZE = 5

HID = 100
PT_HID = 120
BATCH = 1024
EPOCHS = 3
LR = 0.01
VOCAV_SIZE = len(ut.C2I)
TAGS_SIZE = len(ut.T2I)
inputfile = "pos_neg_train"


class LSTMmodule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMmodule, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(torch.tanh(tag_space), dim=1)
        return tag_scores

def get_dataset(data_file, is_train=True):
    """
    get data as windows saved in data loader
    :param data_file: path to file
    :param is_train: is train routine
    :param separator: seprator in files
    :return: data loader
    """
    print "Getting data from: ", data_file
    words = ut.read_data(data_file, is_train=True)
    if is_train:
        sequences, tags = ut.createWordVec(words)
        tags = np.asarray(tags, np.int32)
        tags = torch.from_numpy(tags)
        tags = tags.type(torch.LongTensor)
    else:
        sequences = ut.createWordVec(words)

    sequences = np.asarray([np.asarray(sec) for sec in sequences])
    # sequences = np.asarray(sequences, np.float32)

    sequences = [torch.from_numpy(sec) for sec in sequences]

    sequences = [sec.type(torch.LongTensor) for sec in sequences]

    if is_train:
        data_set = torch.utils.data.TensorDataset(sequences)
        return DataLoader(data_set, batch_size=BATCH, shuffle=True)
    data_set = torch.utils.data.TensorDataset(sequences)
    return DataLoader(data_set, batch_size=1, shuffle=True)


def trainer(model, data_set, loss_func, optimizer):

    for ITER in range(EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
        for sequence, tag in data_set:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix)
            # targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sequence)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_func(tag_scores, tag)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        tag_scores = model(data_set[0])
        print(tag_scores)

if __name__ == '__main__':
    module = LSTMmodule(EMBEDDING_ROW_LENGTH, HID, VOCAV_SIZE, TAGS_SIZE)
    data = get_dataset(inputfile)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(module.parameters(), lr=0.1)
    module.trainer(module, data, loss_function, optimizer)
