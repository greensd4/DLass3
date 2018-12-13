import random
# sequence of letters appearance in the language
pos_seq = ['a','b','c','d']
neg_seq = ['a','c','b','d']


def generate_seq(num):
    positive = []
    negative = []
    # each iteration generates pos and neg word.
    for i in range(num):
        pos_nums = []
        neg_nums = []
        pos_str = ''
        neg_str = ''
        for j in range(5):
            # gets new number for positive seq.
            pos_num_seq = "".join([random.choice("0123456789") for x in range(random.randint(1,20))])
            pos_nums.append(pos_num_seq)
            # gets new number for negative seq.
            neg_num_seq = "".join([random.choice("0123456789") for x in range(random.randint(1,20))])
            neg_nums.append(neg_num_seq)
        # concatenate the numbers and the letters.
        for k in range(len(pos_nums)-1):
            pos_str += str(pos_nums[k]) + pos_seq[k]
            neg_str += str(neg_nums[k]) + neg_seq[k]
        pos_str += str(pos_nums[-1])
        neg_str += str(neg_nums[-1])
        # adding the new word to the language.
        positive.append(pos_str)
        negative.append(neg_str)

    return positive,negative


def write_examples(pos,neg,num):
    pos_fd = open('pos_examples', 'w')
    neg_fd = open('neg_examples', 'w')
    for k in range(num):
        pos_fd.write(pos[k] + '\n')
        neg_fd.write(neg[k] + '\n')
    pos_fd.close()
    neg_fd.close()
    print 'successfully created 500 examples!'


def write_train(pos,neg,num):
    pos_count = 0
    neg_count = 0
    fd = open('pos_neg_train', 'w')
    for k in range(num):
        r = random.randint(0, 1)
        if r is 0:
            fd.write(pos[k] + '\tpos\n')
            pos_count += 1
        else:
            fd.write(neg[k] + '\tneg\n')
            neg_count += 1
    fd.close()
    if any([pos_count/num > 0.7, neg_count/num > 0.7]):
        print 'trying again!'
        write_train(pos,neg,num)
    else:
        print 'successfully created the train set!'


def write_dev(pos,neg,num):
    pos_count = 0
    neg_count = 0
    fd = open('pos_neg_test', 'w')
    for k in range(num):
        r = random.randint(0, 1)
        if r is 0:
            fd.write(pos[k] + '\tpos\n')
            pos_count += 1
        else:
            fd.write(neg[k] + '\tneg\n')
            neg_count += 1
    fd.close()
    if any([pos_count/num > 0.7, neg_count/num > 0.7]):
        print 'trying again!'
        write_train(pos,neg,num)
    else:
        print 'successfully created the test set!'


def write_test(pos,neg,num):
    pos_count = 0
    neg_count = 0
    fd = open('pos_neg_test', 'w')
    for k in range(num):
        r = random.randint(0, 1)
        if r is 0:
            fd.write(pos[k] + '\n')
            pos_count += 1
        else:
            fd.write(neg[k] + '\n')
            neg_count += 1
    fd.close()
    if any([pos_count / num > 0.7, neg_count / num > 0.7]):
        print 'trying again!'
        write_train(pos, neg, num)
    else:
        print 'successfully created the test set!'

if __name__ == '__main__':
    pos_ex,neg_ex = generate_seq(500)
    write_examples(pos_ex,neg_ex,500)

    pos_train,neg_train = generate_seq(200)
    write_train(pos_train,neg_train,200)

    pos_dev, neg_dev = generate_seq(200)
    write_dev(pos_dev, neg_dev, 200)


