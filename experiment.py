import sys
import time
from random import random
import dynet as dy
import numpy as np
import utils as ut
from utils import T2I, I2T, C2I, I2C

STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}

# Globals
IN_DIM = 100
HID_DIM = 100
TAGS_SIZE = 2
EPOCHS = 6
VSIZE = len(C2I)


class LSTM(object):
    def __init__(self, in_dim, lstm_dim, out_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self.W = model.add_parameters((out_dim, lstm_dim))

    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        prm = self.W.expr()
        sn= lstm.transduce(sequence)
        result = prm * sn[-1]
        return result


class Trainer(object):
    def __init__(self):
        self.m = dy.Model()
        self.trainer = dy.AdamTrainer(self.m)
        self.E = self.m.add_lookup_parameters((VSIZE, IN_DIM))
        self.acceptor = LSTM(IN_DIM, HID_DIM, TAGS_SIZE, self.m)

    def train(self, train_data, test_data):
        sum_loss = 0.0
        start_time = time.time()
        for epoch in range(EPOCHS):
            print "Epoch number ", epoch, " started!"
            random.shuffle(train_data)
            sum_loss += self.routine(train_data, is_train=True)
            print "train #{}: loss is {}, accuracy is {}".format(epoch, sum_loss/len(train_data), self.get_accuracy(train_data))

            test_loss = self.routine(test_data)
            print "test: " + "loss is: " + str(float(test_loss) / len(test_data)) + " accuracy is: " + str(self.get_accuracy(test_data))
            sum_loss = 0.0

        end_time = time.time()
        total_time = end_time - start_time
        print "total time: " + str(total_time)

    def routine(self, data, is_train=False):
        sum_loss = 0.0
        for word, tag in data:
            dy.renew_cg()
            word_as_vector = self.word_to_vec(word)
            predictions = self.acceptor(word_as_vector)
            loss = dy.pickneglogsoftmax(predictions, tag)
            sum_loss += loss.npvalue()
            loss.backward()
            if is_train:
                self.trainer.update()
        return sum_loss

    def predict(self, word):
        dy.renew_cg()
        vec = self.word_to_vec(word)
        preds = dy.softmax(self.acceptor(vec))
        vals = preds.npvalue()
        return np.argmax(vals)

    def get_accuracy(self, data):
        good = 0
        for word,tag in data:
            pred = self.predict(word)
            if tag == pred:
                good += 1
        return float(good) / float(len(data))

    def word_to_vec(self, word):
        return [self.E[C2I[c]] for c in word]


def main(argv):
    train_data = ut.read_data(argv[0], is_train=True)
    test_data = ut.read_data(argv[1], is_train=True)
    trainer = Trainer()
    trainer.train(train_data, test_data)


if __name__ == '__main__':
    main(sys.argv[1:])