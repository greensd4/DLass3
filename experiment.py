import sys
import time
import random
import dynet as dy
import numpy as np
import utils as ut
from utils import T2I, I2T, C2I, I2C

STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}

# Globals
LAYERS = 1
EMBEDDED = 50
IN_DIM = 150
HID_DIM = 100
TAGS_SIZE = 2
EPOCHS = 3
VSIZE = len(C2I)


class LSTM(object):
    def __init__(self,layers, em_dim, in_dim, hid_dim, out_dim, vsize, model):
        self.__RNN = dy.LSTMBuilder(layers, em_dim, in_dim, model)
        self.__E = model.add_lookup_parameters((vsize, em_dim))
        self.__W1 = model.add_parameters((hid_dim, in_dim))
        self.__b1 = model.add_parameters(hid_dim)
        self.__W2 = model.add_parameters((out_dim, hid_dim))
        self.__b2 = model.add_parameters(out_dim)

    def __call__(self, sequence):
        dy.renew_cg()
        W1 = dy.parameter(self.__W1)
        W2 = dy.parameter(self.__W2)
        b1 = dy.parameter(self.__b1)
        b2 = dy.parameter(self.__b2)
        lstm = self.__RNN.initial_state()
        embedded = [dy.lookup(self.__E, s) for s in sequence]
        out = lstm.transduce(embedded)
        x = out[-1]
        h = dy.tanh(W1*x + b1)
        return dy.softmax(W2*h + b2)

    def get_loss(self, sequence, expected):
        probs = self(sequence)
        return -dy.log(dy.pick(probs, expected))

    def predict(self, sequence):
        probs = self(sequence)
        return np.argmax(probs.npvalue())


class Trainer(object):
    def __init__(self):
        self.m = dy.Model()
        self.trainer = dy.AdamTrainer(self.m)
        self.acceptor = LSTM(layers=LAYERS, in_dim=IN_DIM, hid_dim=HID_DIM,
                             out_dim=TAGS_SIZE, em_dim=EMBEDDED, vsize=VSIZE, model=self.m)

    def train(self, train_data, dev_data):
        start_time = time.time()
        for epoch in range(EPOCHS):
            sum_loss = 0.0
            random.shuffle(train_data)
            print "Epoch number ", epoch, " started!"
            for word, tag in train_data:
                vector = self.word_to_vec(word)
                loss = self.acceptor.get_loss(vector, T2I[tag])
                sum_loss += loss.value()
                loss.backward()
                self.trainer.update()
            avg_loss = sum_loss/len(train_data)
            accuracy = self.get_accuracy(dev_data)
            print "train #{}: loss is {}, accuracy is {}%".format(epoch, avg_loss, accuracy)

        end_time = time.time()
        total_time = end_time - start_time
        print "total time: " + str(total_time)

    def routine(self, data, is_train=False):
        sum_loss = 0.0
        for word, tag in data:
            dy.renew_cg()
            word_as_vector = self.word_to_vec(word)
            predictions = self.acceptor(word_as_vector)
            loss = dy.pickneglogsoftmax(predictions, T2I[tag])
            sum_loss += loss.npvalue()
            loss.backward()
            if is_train:
                self.trainer.update()
        return sum_loss

    def get_accuracy(self, data):
        good = 0
        random.shuffle(data)
        for word,tag in data:
            sequence = self.word_to_vec(word)
            pred = self.acceptor.predict(sequence)
            if pred == T2I[tag]:
                good += 1
        return 100*(float(good) / float(len(data)))

    def word_to_vec(self, word):
        return [C2I[c] for c in word]


def main(argv):
    train_data = ut.read_data(argv[0], is_train=True)
    test_data = ut.read_data(argv[1], is_train=True)
    trainer = Trainer()
    trainer.train(train_data, test_data)


if __name__ == '__main__':
    main(sys.argv[1:])
