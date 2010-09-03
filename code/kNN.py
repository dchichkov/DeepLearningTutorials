#!/usr/bin/python
# -*- coding: utf-8  -*-

"""
 Naive kNN test.
"""

from __future__ import division
import numpy, time, cPickle, gzip, sys, os, heapq
import theano
import theano.tensor as T
from operator import itemgetter
from utils import tile_raster_images
from logistic_sgd import load_data
import PIL.Image



def test_kNN( dataset ='../data/mnist.pkl.gz',
              k = 9,
              distance = T.sqr,
              batch_size = 10000,
              train_set_size = float('infinity'),
              test_set_size = 1000,
              img_shape = (28, 28),
              output_folder = 'plots' ):

    """
    This demo is tested on '../data/mnist.pkl.gz'

    :type batch_size: int
    :param batch_size: batch_size used for training 

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    
    from theano import ProfileMode
    profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())

    
    # load data
    ((train_set_x, train_set_y), (valid_set_x, valid_set_y), 
        (test_set_x , test_set_y )) = load_data(dataset)
    
    
    # compute number of minibatches for training, validation and testing
    train_set_size = min(train_set_x.value.shape[0], train_set_size)
    test_set_size = min(test_set_x.value.shape[0], test_set_size)
    batch_size = train_set_size 
    n_train_batches = train_set_size / batch_size

    # allocate symbolic variables for the data
    indexMB = T.lscalar()    # index to a [mini]batch 
    indexT  = T.lscalar()    # index to a test set item
    a     = T.matrix('a')  # the data is presented as vectors
    b     = T.vector('b')  # the data is presented as vectors
    
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    #############################################
    # Exhaustive kNN queries for the test set   #
    #############################################
    
    # define theano.function
    DistancesF = T.sum( distance(a - b), axis=1 )        
    calc_distances = theano.function([indexMB, indexT], DistancesF,
         givens = {a:train_set_x[indexMB*batch_size:(indexMB+1)*batch_size], 
                   b:test_set_x[indexT]},
         )#mode=profmode)
    
    # go through test set
    start_time = time.clock()
    kNN = []
    for test_index in xrange(test_set_size):
        c = []
        for batch_index in xrange(n_train_batches):
            batchDistances = calc_distances(batch_index, test_index)
            batchKNN = [(batch_index * batch_size + i, batchDistances[i]) \
                            for i in batchDistances.argsort()[:k] ]
            c.extend(batchKNN)
            
        c.sort(key=itemgetter(1))
        kNN.append(c[:k])
        
        # print out time/ETA
        if(test_index and not test_index % 100):
            test_time = time.clock() - start_time + 0.00000001; 
            ETA = ((test_time / test_index) * test_set_size - test_time); 
            print "%.1f seconds; test %d (%d); ETA %.1f seconds" % (test_time, test_index, test_set_size, ETA),
            print "Gdists/sec = %f (3 flop/dist)" % (28*28*train_set_size*test_index / 1000000000 / test_time)
        
    end_time = time.clock()
    
    #for t in kNN: print t
    training_time = (end_time - start_time)
    profmode.print_summary()
    print >> sys.stderr, ('The kNN search code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((training_time)/60.))
    
    
    ### FIXME, why using test_set_x.value[i] directly eats memory? 
    ###        the whole thing gets transferred from the GPU?
    #  
    # X = [train_set_x.value[i] for i in xrange(train_set_x.value.shape[0])]
     
    #############################################
    # Plot Results                              #
    #############################################
    tN = 100; X = []
    for test_index in xrange(test_set_size):
        X.append(numpy.array(test_set_x.value[test_index]))                          
        for i in kNN[test_index]: X.append(numpy.array(train_set_x.value[i[0]]))
        if len(X) >= tN: break
    
    plot_file_name = os.path.split(dataset)[1] + "-kNN.png"
    PIL.Image.fromarray( tile_raster_images( numpy.asarray(X),
                              img_shape = img_shape, tile_shape = (tN//(1+k),10),
                              tile_spacing=(1,1), scale_rows_to_unit_interval = False,
                              output_pixel_vals = True)).save(plot_file_name)
    print >> sys.stderr, ("Produced plot %s/%s (1T%dNN)" % (output_folder, plot_file_name, k)) 
    
    os.chdir('../')


if __name__ == "__main__":    
    test_kNN()

