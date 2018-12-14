import numpy as np
import dynet as dy
import cPickle
import itertools

BACKWARD = "BACKWARD"
FORWARD = "FORWARD"


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
        return [dy.concatenate([backward,forward]) for backward, forward in itertools.izip([out_f, out_b])]


class DoubleBiLSTM:
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
        return [-dy.log(dy.pick(prob, expect)) for prob, expect in itertools.izip([probs, expected])]

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
            total_loss.extend([-dy.log(dy.pick(prob, tag)) for prob, tag in itertools.izip([probs, tags])])

        return dy.esum(total_loss)/ total_loss


    def save(self, fname):
        self.model.save(fname)

    def load(self, fname):
        self.model.populate(fname)


class WordEmbeddingDoubleBiLSTM(DoubleBiLSTM):
    def __init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model):
        DoubleBiLSTM.__init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model)

    def represent(self, input):
        return [dy.lookup(self.E, i) for i in input]


class BDoubleBiLSTM(DoubleBiLSTM):
    def __init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model):
        DoubleBiLSTM.__init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model)
        self.LSTMc = dy.LSTMBuilder(layers, em_dim, em_dim, self.model)

    def represent(self, input):
        init_state = self.LSTMc.initial_state()
        transduces = []
        for word in input:
            transduces.append(init_state.transduce([dy.lookup(self.E, c) for c in word])[-1])
        return transduces

class C(DoubleBiLSTM):
    def __init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model, Wp2I, Ws2I):
        DoubleBiLSTM.__init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, model)
        self.Epre = self.model.add_lookup_parameters(len(Wp2I), em_dim)
        self.Esuf = self.model.add_lookup_parameters(len(Ws2I), em_dim)
        self.Wp2I = Wp2I
        self.Ws2I = Ws2I

    def represent(self, input):
        representations = []
        for word in input:
            w_r = dy.lookup(self.E, word)
            p_r = dy.lookup(self.Epre, self.Wp2I[word])
            s_r = dy.lookup(self.Esuf, self.Ws2I[word])
            representations.append(w_r + p_r + s_r)
        return representations

class D(BDoubleBiLSTM):
    def __init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, vsize, cvsize , model):
        DoubleBiLSTM.__init__(self, layers, em_dim, in_dim, lstm_dim, tags_size, cvsize, model)
        self.WE = self.model.add_lookup_parameters(vsize, em_dim)
        self.W = self.model.add_parameters(em_dim, 2*em_dim)
        self.b = self.model.add_parameters(em_dim)

    def represent(self, input):
        # input is list of tuples, the first value in the tuple is a word as index,
        # and the second is list of indexes of each char in word
        words, chars = zip(*input)
        chars = BDoubleBiLSTM.represent(self, list(chars))
        embedded = []
        for word, em_char in itertools.izip(words, chars):
            embedded.append(dy.concatenate([dy.lookup(self.WE, word), em_char]))
        W = dy.parameter(self.W)
        b = dy.parameter(self.b)
        return [W*x+b for x in embedded]



if __name__ == '__main__':
    s = [(5,[1, 2 ,3]),(6, [7,8,9])]
    w, c = zip(*s)
    print w
    print c