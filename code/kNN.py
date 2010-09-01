#!/usr/bin/python
# -*- coding: utf-8  -*-


"""
 Statistics for distances between MP3 frames.
"""

import numpy, time, cPickle, gzip, sys, os

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from load_mp3 import *
from utils import tile_raster_images

import PIL.Image


def test_kNN( dataset ='/home/dmitry/mp3/hhgttg01010060.mp3',
              batch_size = 20, output_folder = 'dA_plots' ):

    """
    This demo is tested on /home/dmitry/mp3/hhgttg01010060.mp3

    :type batch_size: int
    :param batch_size: batch_size used for training 

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)
    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x , test_set_y ) = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    a     = T.matrix('a')  # the data is presented as vectors
    b     = T.vector('b')  # the data is presented as vectors
    
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # Exaustive kNN search             #
    ####################################
    DistancesF = T.sum( T.abs_(a - b), axis=1 )        
    calc_distances = theano.function([index], DistancesF,
         givens = {a:train_set_x[index*batch_size:(index+1)*batch_size], 
                   b:test_set_x[0]})
    
    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training set
    c = []
    for batch_index in xrange(n_train_batches):
        c.append(calc_distances(batch_index))

    # heapq.nsmallest(k, iterable)
    end_time = time.clock()

    training_time = (end_time - start_time)
    print >> sys.stderr, ('The kNN search code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((training_time)/60.))
    # image = PIL.Image.fromarray(tile_raster_images( X = da.W.value.T,
    #              img_shape = (S_DIM, F_DIM * X_DIM),tile_shape = (10,10), 
    #             tile_spacing=(1,1)))
    #image.save('filters_corruption_0.png') 
    #
 
    os.chdir('../')


if __name__ == "__main__":    
    d = "/home/dmitry/mp3/01- Hitchhikers Guide to the Galaxy"
    dataset = sorted([os.path.join(d, f) for f in os.listdir(d)])
    # dataset = ["/home/dmitry/mp3/hhgttg01010060.mp3"]
    test_kNN(dataset = dataset[:5])

