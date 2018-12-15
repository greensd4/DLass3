import numpy as np
import dynet as dy
import itertools
import json

BACKWARD = "BACKWARD"
FORWARD = "FORWARD"

SUFF_LENGTH , PREF_LENGTH = 3, 3

def word_to_prefix(w):
    return w[0:PREF_LENGTH]


def word_to_suffix(w):
    return w[-SUFF_LENGTH:]


class BiLSTM:
    def __init__(self, layers, em_dim, lstm_dim, model):
        self.directions = {
            BACKWARD: dy.LSTMBuilder(layers, em_dim, lstm_dim, model),
            FORWARD: dy.LSTMBuilder(layers, em_dim, lstm_dim, model)
        }

    def __call__(self, sentence):
        rev_sentence = sentence[::-1]
        start_b = self.directions[BACKWARD].initial_state()
        start_f = self.directions[FORWARD].initial_state()
        out_f, out_b = start_f.transduce(sentence), start_b.transduce(rev_sentence)
        return [dy.concatenate([backward,forward]) for backward, forward in itertools.izip(out_f, out_b)]


class DoubleBiLSTM(object):
    def __init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model):
        self._biLSTM_first = BiLSTM(layers, em_dim, lstm_dim, model)
        self._biLSTM_second = BiLSTM(layers, 2*lstm_dim, in_dim, model)
        self.model = model
        self.E = self.model.add_lookup_parameters((vsize, em_dim))
        self.W = self.model.add_parameters((tags_size, 2*in_dim))
        self.b = self.model.add_parameters(tags_size)

    def represent(self, input):
        raise NotImplementedError

    def mlp(self, input):
        W, b = dy.parameter(self.W), dy.parameter(self.b)
        return [dy.softmax(W*x + b) for x in input]

    def __call__(self, sentence, renew_graph=True):
        if renew_graph:
            dy.renew_cg()
        representation = self.represent(sentence)
        result1 = self._biLSTM_first(representation)
        result2 = self._biLSTM_second(result1)
        return self.mlp(result2)

    def get_loss(self, input, expected):
        probs = self(input)
        return [-dy.log(dy.pick(prob, expect)) for prob, expect in itertools.izip(probs, expected)]

    def predict(self, input):
        probs = self(input)
        return [np.argmax(prob.npvalue()) for prob in probs]

    def batch_predictions(self, data):
        dy.renew_cg()
        probabilities = []
        all_probs = []
        for sentence in data:
            predictions = self(sentence, renew_graph=False)
            probabilities.append(predictions)
            all_probs.extend(predictions)
        dy.forward()
        return [[np.argmax(word.npvalue()) for word in sentence] for sentence in probabilities]

    def batch_loss(self, data):
        total_loss = []
        total = 0.0
        dy.renew_cg()
        for sentence, tags in data:
            probs = self(sentence, renew_graph=False)
            total += len(tags)
            total_loss.extend([-dy.log(dy.pick(prob, tag)) for prob, tag in itertools.izip(probs, tags)])

        return dy.esum(total_loss)/ total_loss

    def save_model(self, fname):
        self.model.save(fname)

    def load_model(self, fname):
        self.model.populate(fname)


class WordEmbeddingDoubleBiLSTM(DoubleBiLSTM):
    def __init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model):
        DoubleBiLSTM.__init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model)

    def represent(self, input):
        return [dy.lookup(self.E, i) for i in input]


class CharLevelDoubleBiLSTM(DoubleBiLSTM):
    def __init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, cvsize, model, in2word, char2index):
        DoubleBiLSTM.__init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, cvsize, model)
        self.LSTMc = dy.LSTMBuilder(layers, em_dim, em_dim, self.model)
        self.index2word = in2word
        self.char2index = char2index

    def represent(self, input):
        init_state = self.LSTMc.initial_state()
        transduces = []
        for word in input:
            word = self.index2word[word]
            transduces.append(init_state.transduce([dy.lookup(self.E, self.char2index[c] ) for c in word])[-1])
        return transduces


class SubWordEmbeddingDoubleBiLSTM(DoubleBiLSTM):
    def __init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model, Wp2I, Ws2I,I2W):
        DoubleBiLSTM.__init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model)
        self.Epre = self.model.add_lookup_parameters((len(Wp2I), em_dim))
        self.Esuf = self.model.add_lookup_parameters((len(Ws2I), em_dim))
        self.Wp2I = Wp2I
        self.Ws2I = Ws2I
        self.I2W = I2W

    def represent(self, input):
        representations = []
        for word in input:
            w_r = dy.lookup(self.E, word)
            p_r = dy.lookup(self.Epre, self.Wp2I[word_to_prefix(self.I2W[word])])
            s_r = dy.lookup(self.Esuf, self.Ws2I[word_to_suffix(self.I2W[word])])
            representations.append(w_r + p_r + s_r)
        return representations



class CharAndEmbeddedDoubleBiLSTM(CharLevelDoubleBiLSTM):
    def __init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, cvsize , model,  in2word, char2index):
        CharLevelDoubleBiLSTM.__init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, cvsize, model,  in2word, char2index)
        self.WE = self.model.add_lookup_parameters((vsize, em_dim))
        self.W_t = self.model.add_parameters((em_dim, 2*em_dim))
        self.b_t = self.model.add_parameters(em_dim)

    def represent(self, input):
        embedded = []
        transtduces = CharLevelDoubleBiLSTM.represent(self, input)
        for word, trans in itertools.izip(input, transtduces):
            t = dy.concatenate([dy.lookup(self.WE, word), trans])
            embedded.append(t)
        W = dy.parameter(self.W_t)
        b = dy.parameter(self.b_t)
        return [W*x+b for x in embedded]


def save_nn_and_data(fname, neural_net, params, model, I2T, wp_index, ws_index, I2W, I2C, unk_index):
    data_dict = {
        "NN_TYPE": neural_net.__class__.__name__,
        "PARAMETERS": {
            "LAYERS":params["layers"],
            "EM_DIM": params["em_dim"],
            "LSTM_DIM" :params["lstm_dim"],
            "IN_DIM":params["in_dim"],
            "TAGS_SIZE": params["tags_size"],
            "VSIZE": params["vsize"],
            "CVSIZE":params["cvsize"],

        },
        "MODEL":model.__class__.__name__,
        "WORDS_LIST": {
            "I2W": I2W,
            "I2C": I2C,
            "WS_INDEX":ws_index,
            "WP_INDEX":wp_index
        },
        "TAGS": I2T,
        "UNK": unk_index
    }
    print data_dict.__str__()
    neural_net.save_model(fname)
    data_fd = open(fname+"_data", 'w')
    json.dump(data_dict, data_fd)
    data_fd.close()


def load_nn_and_data(fname):
    data_fd = open(fname + "_data", 'r')
    loader = json.load(data_fd)

    params = loader["PARAMETERS"]
    words_list = loader["WORDS_LIST"]
    nn_type = loader["NN_TYPE"]
    unk_index = loader["UNK"]
    tags = loader["TAGS"]

    lstm_dim, vsize,cvsize = params["LSTM_DIM"], params["VSIZE"], params["CVSIZE"]
    layers, in_dim, em_dim, tags_size = params["LAYERS"], params["IN_DIM"], params["EM_DIM"], params["TAGS_SIZE"]
    I2W, I2C, ws_index, wp_index = words_list["I2W"], words_list["I2C"], words_list["WS_INDEX"], words_list["WP_INDEX"]
    C2I = {(v,k) for k,v in I2C.items()}
    model = loader["MODEL"]
    if model == dy.Model.__name__:
        model = dy.Model()
    else:
        model = dy.Model()
    if nn_type == WordEmbeddingDoubleBiLSTM.__name__:  # Option (a)
        net = WordEmbeddingDoubleBiLSTM(layers, em_dim, in_dim, lstm_dim, tags_size, len(I2W), model)

    elif nn_type == CharLevelDoubleBiLSTM.__name__:  # Option (b)
        net = CharLevelDoubleBiLSTM(layers, em_dim, in_dim, lstm_dim, tags_size, cvsize, model, I2W, C2I)

    elif nn_type == SubWordEmbeddingDoubleBiLSTM.__name__:
        net = SubWordEmbeddingDoubleBiLSTM(layers, em_dim, in_dim, lstm_dim, tags_size, len(I2W), model, wp_index, ws_index, I2W)
    else:
        net = CharAndEmbeddedDoubleBiLSTM(layers, em_dim, in_dim, lstm_dim, tags_size, vsize, cvsize, model, I2W, C2I)

    net.load_model(fname)  # loads the parameter collection
    return net, tags, I2W, I2C, unk_index



