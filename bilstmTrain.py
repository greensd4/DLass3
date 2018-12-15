import os
import random
from itertools import izip
from math import floor
from optparse import OptionParser
from time import time

import dynet as dy
import json
from bi_lstm_models import WordEmbeddingDoubleBiLSTM as A
from bi_lstm_models import CharLevelDoubleBiLSTM as B
from bi_lstm_models import SubWordEmbeddingDoubleBiLSTM as C
from bi_lstm_models import CharAndEmbeddedDoubleBiLSTM as D
from bi_lstm_models import word_to_suffix, word_to_prefix
from bi_lstm_models import save_nn_and_data

UNK = "UUUNKKK"
SEPARATOR = " "
UNK_INDEX = -1
WORDS, TAGS , CHARS = set(), set(), set()
T2I, I2T, W2I, I2W = dict(), dict(), dict(), dict()
I2C, C2I = dict(), dict()
S2I, I2S, P2I, I2P = dict(), dict(), dict(), dict()
PARAMS = dict()
BATCH = 500
EPOCHS = 5
IGNORED = ""
option_parser = OptionParser()
option_parser.add_option("-t", "--type", dest="type", help="choose POS/NER tagging (pos/ner) - default is pos tagging",
                         default="pos")
option_parser.add_option("-d", "--dev", help="dev file name", dest="dev", default="dev")


def main():
    options, args = option_parser.parse_args()
    nn_type, ftrain, fmodel = args
    if options.dev == 'dev':
        fdev = os.path.join(options.type, options.dev)
    else:
        fdev = options.dev
    initialize_globals(options.type)
    train_data, dev_data = read_train_and_dev(ftrain, fdev)
    neural_network = initialize_neural_network(nn_type)
    trainer = dy.AdamTrainer(neural_network.model)
    train(neural_network, trainer, train_data, dev_data, IGNORED)
    save_information(fmodel, neural_network)


def save_information(fmodel, neural_network):
    save_nn_and_data(fmodel, neural_network, PARAMS, neural_network.model, I2T, P2I, S2I, I2W, I2C, UNK_INDEX)


def train(neural_network, trainer, train_data, dev_data, ignored_tag):
    total_words = reduce(lambda x, y: x + len(y), train_data, 0.0)
    for epoch in range(EPOCHS):
        total_loss = 0.0
        start_time = time()
        i = 1
        acc = 0
        num_of_words_till_now = 0
        for sentence, tags in train_data:
            sentence = [W2I[w] for w in sentence]
            tags = [T2I[t] for t in tags]
            loss = dy.esum(neural_network.get_loss(sentence, tags))
            total_loss += loss.value()
            loss.backward()
            trainer.update()
            num_of_words_till_now += len(sentence)
            if i % BATCH is 0:
                acc = accuracy(neural_network, dev_data, ignored_tag) * 100
                avg_loss = total_loss/total_words
                print "BATCH: {} , ACC {:11f}, AVG LOSS {}".format(i/BATCH, acc, avg_loss)
            i += 1
        end_time = time()-start_time

        print "Epoch: {}, Total Loss: {:12f}, Time: {:9f}s, ACC: {:11f}".format(epoch+1, total_loss, end_time, acc)


def accuracy(nn, dev, ignored):
    correct = 0
    total = 0.0
    for sentence, tags in dev:
        sentence = [get_word_index(w) for w in sentence]
        tags = [T2I[t] for t in tags]
        preds = nn.predict(sentence)
        for p,t in zip(preds, tags):
            if I2T[t] == ignored:
                continue
            elif t == p:
                correct += 1
            total +=1
    return 100 * float(correct)/total


def get_word_index(w):
    try:
        return W2I[w]
    except KeyError:
        return W2I[UNK]


def split_data_to_batch(data):
    batches = []
    start = 0
    it = int(floor(len(data)/ BATCH))
    for i in range(it):
        batches.append(data[start:start+BATCH])
        start += BATCH
    batches.append(data[start:len(data)-1])
    return batches


def initialize_nn_params(nn_type):
    global PARAMS
    if nn_type in ['b', 'd']:
        initialize_char_dictionaries()
        PARAMS["cvsize"] = len(CHARS)
    elif nn_type is 'c':
        initialize_pref_suff()


def initialize_neural_network(nn_type):
    model = dy.Model()
    initialize_nn_params(nn_type)
    layers, em_dim, in_dim, lstm_dim, tags_size, vsize, cvsize = PARAMS["layers"],PARAMS["em_dim"],PARAMS["in_dim"],\
                                                                 PARAMS["lstm_dim"], PARAMS["tags_size"],\
                                                                 PARAMS["vsize"], PARAMS["cvsize"]
    if nn_type is 'a':
        return A(layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model)
    elif nn_type is 'b':
        return B(layers, em_dim, in_dim, lstm_dim, tags_size, cvsize, model)
    elif nn_type is 'c':
        return C(layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model, P2I, S2I)
    else:
        return D(layers, em_dim, in_dim, lstm_dim, tags_size, vsize, cvsize, model)


def init_fix():
    prefixes = []
    suffix = []
    for word in WORDS:
        pref = word_to_prefix(word)
        suff = word_to_suffix(word)
        prefixes.append(pref)
        suffix.append(suff)
    return set(prefixes), set(suffix)


def initialize_pref_suff():
    pref, suff = init_fix()
    global I2P, I2S, S2I, P2I
    I2P, P2I = index_set(pref)
    I2S, S2I = index_set(suff)


def initialize_char_dictionaries():
    global CHARS, I2C, C2I
    for word in WORDS:
        for c in word:
            CHARS.add(c)
    I2C, C2I = index_set(CHARS)
    global PARAMS
    PARAMS["cvsize"] = len(CHARS)


def initialize_globals(tag_type):
    global SEPARATOR, PARAMS, IGNORED
    if tag_type.lower() == "ner":
        SEPARATOR = "\t"
        IGNORED = "O"
    else:
        SEPARATOR = " "
    config_fd = open("config.json","r")
    PARAMS = json.load(config_fd)
    config_fd.close()


def index_set(set_to_dict):
    i2s, s2i = {}, {}
    for i, item in enumerate(set_to_dict):
        i2s[i] = item
        s2i[item] = i
    return i2s, s2i


def initialize_words_tags():
    global PARAMS
    global T2I, I2T, W2I, I2W
    I2T, T2I = index_set(TAGS)
    I2W, W2I = index_set(WORDS)
    PARAMS["vsize"] = len(WORDS)
    PARAMS["tags_size"] = len(TAGS)


def read_train_and_dev(ftrain, fdev):
    train_data = read_data(ftrain)
    dev_data = read_data(fdev, is_train=False)
    initialize_words_tags()
    return train_data, dev_data


def read_data(fname, tagged_data=True, is_train=True):
    """
    This function reads the data into list of sentences.
    if the data is tagged data and each word is in new line,
    each sentence will be list of words with their tags, as tuples.
    :param fname: file path
    :param tagged_data: if the data is tagged
    :param is_train: if the data is read for train
    :return: data , list of sentences
    """
    data = []
    sentence = []
    tags = []
    # For not tagged data
    print "Reading data from:", fname, " tagged data?", tagged_data, " is train?", is_train
    global TAGS, WORDS
    for line in file(fname):
        try:
            word, label = line.strip().split(SEPARATOR, 1)
            sentence.append(word)
            tags.append(label)
            if is_train:
                TAGS.add(label)
                WORDS.add(word)
        except ValueError:
            data.append((sentence, tags))
            sentence = []
            tags = []
    if len(sentence) is not 0 and (sentence, tags) not in data:
        data.append((sentence, tags))
    if is_train:
        WORDS.add(UNK)
        TAGS.add(UNK)
    print "Finished reading data from file", fname
    return data






if __name__ == '__main__':
   main()