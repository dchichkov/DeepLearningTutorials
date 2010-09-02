#!/usr/bin/python
# -*- coding: utf-8  -*-
from __future__ import division

import mad, numpy, random, os, time, cPickle, gzip, sys
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images
import matplotlib.pyplot
import theano
import theano.tensor as T

# corpus size
N_DIM = 576  # 576 dims
(TRAIN, VALID, TEST) = (250000, 10, 10)

   
def load_data(dataset = ""):
    ''' Loads the fake 'distances' dataset:
         [[label/N, label/N, ...] : label, ...] for label in [0..N]

    :type dataset: string
    :param dataset: the path to the dataset (ignored)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'

    # make train/valid/test datasets
    train_set = ([[i/TRAIN]*N_DIM   for i in xrange(0, TRAIN)], 
                 [i for i in xrange(0, TRAIN)] )
    valid_set = ([[i/VALID]*N_DIM for i in xrange(VALID)], 
                 [i for i in xrange(VALID)] )
    test_set = ([[i/TEST]*N_DIM for i in xrange(TEST)],
                [i for i in xrange(TEST)] )
    

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
    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval


if __name__ == "__main__":    
    load_data()
