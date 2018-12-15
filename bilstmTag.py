import dynet as dy
from optparse import OptionParser
from bi_lstm_models import load_nn_and_data
option_parser = OptionParser()
option_parser.add_option()
SEPARATOR = " "
UNK = "UUUNKKK"


def main():
    (options, args) = option_parser.parse_args()
    nn_type , model_file, test_file = args
    net, tags, W2I, C2I, unk_index = load_nn_and_data(model_file, nn_type)
    data = read_data(test_file)
    total_tags = []
    for sentence in data:
        sentence = [W2I[word] if word in W2I.keys() else W2I[UNK] for word in sentence]
        tags_pred = net.predict(sentence)
        tags_pred = [tags[t] for t in tags_pred]
        total_tags = total_tags + tags_pred
    save_predictions(test_file, "test4"+options.type, total_tags)


def read_data(test_file):
    data = []
    sentence = []
    tags = []
    # For not tagged data
    print "Reading data from:", test_file
    for word in file(test_file):
        if word.strip() != "":
            sentence.append(word)
        else:
            data.append((sentence))
            sentence = []
    print "Finished reading data from file", test_file
    return data


def save_predictions(in_file, out_file, tags):
    print "Writing Predictions!"
    fd_out = open(out_file, 'w')
    fd_in = open(in_file, 'r')
    for i, line in enumerate(fd_in):
        if line.strip() == "":
            fd_out.write(line)
        else:
            line = line + SEPARATOR + tags[i]
            fd_out.write(line)
            i += 1


if __name__ == '__main__':
    main()