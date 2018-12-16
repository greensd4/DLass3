import dynet as dy
from optparse import OptionParser
from bi_lstm_models import load_nn_and_data
option_parser = OptionParser()
SEPARATOR = " "
UNK = "UUUNKKK"


option_parser.add_option("-t", "--type", dest="type", help="choose POS/NER tagging (pos/ner) - REQUIRED",
                         default=None)


def main():
    (options, args) = option_parser.parse_args()
    if options.type is None:
        option_parser.exit(status=-1, msg="ERROR: You must enter tagging type!")
    nn_type , model_file, test_file = args
    net, tags, W2I, C2I, unk_index = load_nn_and_data(model_file, nn_type)
    data = read_data(test_file)
    total_tags = []

    global SEPARATOR
    if options.type == 'ner':
        SEPARATOR = '\t'

    for sentence in data:
        sentence = [W2I[word] if word in W2I.keys() else W2I[UNK] for word in sentence]
        tags_pred = net.predict(sentence)
        tags_pred = [str(tags[unicode(t)]) for t in tags_pred]
        total_tags = total_tags + tags_pred
    save_predictions(test_file, "test4."+options.type, total_tags)


def read_data(test_file):
    data = []
    sentence = []
    # For not tagged data
    print "Reading data from:", test_file
    for word in file(test_file):
        if word.strip() != "":
            sentence.append(word.strip())
        else:
            data.append((sentence))
            sentence = []
    print "Finished reading data from file", test_file
    return data


def save_predictions(in_file, out_file, tags):
    print "Writing Predictions!"
    fd_out = open(out_file, 'w')
    fd_in = open(in_file, 'r')
    i = 0
    for line in fd_in:
        if line.strip() == "":
            fd_out.write(line)
        else:
            print tags[i]
            line = line.strip() + SEPARATOR + tags[i] + '\n'
            fd_out.write(line)
            i += 1


if __name__ == '__main__':
    main()
