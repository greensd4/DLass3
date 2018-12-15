import os
from optparse import OptionParser
import dynet as dy

UNK = "UUUNKKK"
SEPARATOR = " "
WORDS, TAGS , CHARS = set(), set(), set()
T2I, I2T, W2I, I2W, I2C, C2I = dict(), dict(), dict(), dict(), dict(), dict()
S2I, I2S, P2I, I2P = dict(), dict(), dict(), dict()
SUFF_LENGTH , PREF_LENGTH = 3, 3

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
    train_data, dev_data = read_train_and_dev(ftrain, fdev)
    model = dy.Model()


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


def initialize_globals(tag_type, nn_type):
    global SEPARATOR
    if tag_type.lower() == "ner":
        SEPARATOR = "\t"
    else:
        SEPARATOR = " "



def word_to_prefix(w):
    return w[0:PREF_LENGTH]


def word_to_suffix(w):
    return w[-SUFF_LENGTH:]


def index_set(set_to_dict):
    i2s, s2i = {}, {}
    for i, item in enumerate(set_to_dict):
        i2s[i] = item
        s2i[item] = i
    return i2s, s2i


def initialize_dicts():
    global T2I, I2T, W2I, I2W
    I2T, T2I = index_set(TAGS)
    I2W, W2I = index_set(WORDS)


def read_train_and_dev(ftrain, fdev):
    train_data = read_data(ftrain)
    dev_data = read_data(fdev, is_train=False)
    initialize_dicts()
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
    # For not tagged data
    print "Reading data from:", fname, " tagged data?", tagged_data, " is train?", is_train
    global TAGS, WORDS
    for line in file(fname):
        try:
            word, label = line.strip().split(SEPARATOR, 1)
            sentence.append((word, label))
            if is_train:
                TAGS.add(label)
                WORDS.add(word)
        except ValueError:
            data.append(sentence)
            sentence = []
    if len(sentence) is not 0 and sentence not in data:
        data.append(sentence)
    if is_train:
        WORDS.add(UNK)
        TAGS.add(UNK)
    print "Finished reading data from file", fname
    return data






if __name__ == '__main__':
   main()