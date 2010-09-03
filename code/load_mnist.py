#!/usr/bin/python
# -*- coding: utf-8  -*-
from __future__ import division

"""
Load mnist digits
"""
__docformat__ = 'restructedtext en'

import numpy, time, cPickle, gzip, sys, os
from utils import tile_raster_images

import theano
import theano.tensor as T


A_DIM = 28   
B_DIM = 28
N_DIM = 28*28



def load_data(dataset = '../data/mnist.pkl.gz'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set = (test_set[0][:200], test_set[1][:200])
    print train_set[1][:10]

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
    import PIL.Image
    ((train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)) = \
        load_data()

    tN = 2000
    print "len(train_set_x.value.T)", len(train_set_x.value)
    for i in xrange(1): # xrange(len(train_set_x.value)//tN):
        X = [train_set_x.value[j] for j in xrange(i*tN, i*tN+tN)]
        X = numpy.asarray(X)
        arr = tile_raster_images( X,
                                  img_shape = (A_DIM, B_DIM),tile_shape = (tN//10,10),
                                  tile_spacing=(1,1), scale_rows_to_unit_interval = False,
                                  output_pixel_vals = True)
        #matplotlib.pyplot.imsave(fname = 'hhgttg010100%d.png' % i, arr = arr)#, vmin = 0.0, vmax = 1.0)
        PIL.Image.fromarray(arr).save('mnist%d.png' % i)
    
    


