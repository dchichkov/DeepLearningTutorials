#!/usr/bin/python
# -*- coding: utf-8  -*-
from __future__ import division

import csv, string, numpy, random
import theano
import theano.tensor as T

X_DIM = 28
V_DIM = 28

# map letters to letters 'a':['a'], 'b':['b'] ... 
# LVECTORS = {}
#for i in xrange(0,26): LVECTORS[chr(i + ord('a'))] = [chr(i + ord('a'))]
#LVECTORS[' '] = [' ']
#LVECTORS['-'] = ['-']
#LVECTORS["'"] = ["'"]

# map letters to binary vectors 'a':[1.0, 0.0, 0.0, ...], b:[0.0, 1.0, 0.0, ...] ...
LVECTORS = {}
for i in xrange(0,26): LVECTORS[chr(i + ord('a'))] = [(0.0, 1.0)[i == j] for j in xrange(V_DIM)]
LVECTORS[' '] = [(0.0, 1.0)[V_DIM == j] for j in xrange(V_DIM)]
LVECTORS['-'] = [(0.0, 1.0)[26 == j] for j in xrange(V_DIM)]
LVECTORS["'"] = [(0.0, 1.0)[27 == j] for j in xrange(V_DIM)]


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (ignored)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data' 

    # Using real names corpus, requires NLTK (sudo apt-get install python-nltk)
    # >>> import nltk
    # >>> nltk.download("names")
    #
    # from nltk.corpus import names
    # names = ([(name, 1) for name in names.words('male.txt')] +
    #         [(name, 0) for name in names.words('female.txt')])
    
    # fake male/female names, random letters, 5-15 in length
    #  female ends on 'a', 'e', 'i'
    #  male ends on 'k', 'o', 'r', 's', 't'
    def fake():
        gender = random.choice((0, 1))
        name = ''.join(random.choice(string.letters) for i in xrange(random.randint(5,12)))        
        name += random.choice((['a', 'e', 'i'], ['k', 'o', 'r', 's', 't'])[gender])
        return name.capitalize(), gender
    names = [fake() for i in xrange(10000)]
    print "len(names)", len(names), names[:5]   
    
    # make train/valid/test datasets
    SHIFTS = 10
    TRAIN = 6000 * SHIFTS; VALID = 1200 * SHIFTS; TEST = 600 * SHIFTS; 
    corpus = []
    for i in xrange(TRAIN + VALID + TEST):
        (word, label) = random.choice(names)
        x = [LVECTORS[l] for l in list(word.lower())]        
        space = X_DIM - len(word); assert(space > 0);
        shift = random.randint(0, space//2)
        # produce shifted vectorized word (letter vectors are concatenated)
        v = sum([LVECTORS[' ']] * shift + x + [LVECTORS[' ']] * (space - shift), [])
        corpus.append((v, label))

    random.shuffle(corpus)
    print corpus[:1]

    train_set = ([corpus[i][0] for i in xrange(0, TRAIN)], 
                 [corpus[i][1] for i in xrange(0, TRAIN)] )
    valid_set = ([corpus[i][0] for i in xrange(TRAIN, TRAIN + VALID)], 
                 [corpus[i][1] for i in xrange(TRAIN, TRAIN + VALID)] )
    test_set = ([corpus[i][0] for i in xrange(TRAIN + VALID, TRAIN + VALID + TEST)],
                [corpus[i][1] for i in xrange(TRAIN + VALID, TRAIN + VALID + TEST)] )
    # for v in train_set[0]: print v

    def shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are 
        # floats it doesn't make sense) therefore instead of returning 
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    print train_set_x.value
    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval

if __name__ == "__main__":
    load_data('')

