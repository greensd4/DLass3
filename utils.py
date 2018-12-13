WORDS = []
TAGS = []

C2I = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
       "7": 7, "8": 8, "9": 9, "a": 10, "b": 11, "c": 12, "d": 13, "UNK": 14}
T2I = {"pos":0, "neg":1}

I2C = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6",
       7:"7", 8: "8", 9:"9", 10:"a", 11:"b", 12:"c", 13:"d", 14:"UNK"}
I2T = {0:"pos", 1:"neg"}


def read_data(data_file, is_train=True):
    fd = open(data_file, 'r')
    words = []
    print 'start reading file: ', data_file
    if not is_train:
        for line in fd.readlines():
            words.append(line.strip())
        fd.close()
        return words

    for line in fd.readlines():
        word, tag = line.strip().split("\t")
        words.append((word, tag))
    fd.close()
    print 'Done reading from: ', data_file
    return words


def createWordVec(words, is_tagged=True):
    words_vec = []
    tags_vec = []
    if not is_tagged:
        for word in words:
            curr_word = []
            for char in list(word):
                curr_word.append(C2I[char])
            words_vec.append(curr_word)
            return words_vec

    for (word, tag) in words:
        curr_word = []
        for char in list(word):
            curr_word.append(C2I[char])
        words_vec.append(curr_word)
        tags_vec.append(T2I[tag])

    return words_vec, tags_vec
