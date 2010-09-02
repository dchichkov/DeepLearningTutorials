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

def load_data(dataset = ""):
    ''' Loads random 'distances' dataset:
         [[label/N, label/N, ...] : label, ...] for label in [0..N]

    :type dataset: string
    :param dataset: the path to the dataset (ignored)
    '''

    #############
    # LOAD DATA #
    #############
    print '... generating data...',    
    N_DIM = 576  # 576 dims, 1,000,000 items, 100 queries
    (TRAIN, VALID, TEST) = (1000000, 1000, 1000)

    # make train/valid/test datasets
    train_set = ([[i/TRAIN]*N_DIM   for i in xrange(0, TRAIN)],
                 [i for i in xrange(0, TRAIN)] )
    valid_set = ([[i/VALID]*N_DIM for i in xrange(VALID)],
                 [i for i in xrange(VALID)] )
    test_set = ([[i/TEST]*N_DIM for i in xrange(TEST)],
                [i for i in xrange(TEST)] )
    print '... done.'


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

    print '... generating shared dataset...',    
    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    print ' ...done.'
    return rval




def test_kNN( dataset ='',
              k = 9,
              metric = T.abs_,
              batch_size = 10000,
              output_folder = 'plots' ):

    """
    This demo is tested on ''

    :type batch_size: int
    :param batch_size: batch_size used for training 

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    
    from load_mnist import load_data, A_DIM, B_DIM
    import PIL.Image

    datasets = load_data()
    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x , test_set_y ) = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

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
    DistancesF = T.sum( metric(a - b), axis=1 )        
    calc_distances = theano.function([indexMB, indexT], DistancesF,
         givens = {a:train_set_x[indexMB*batch_size:(indexMB+1)*batch_size], 
                   b:test_set_x[indexT]})
    
    start_time = time.clock()

    # go through training set
    kNN = []
    for test_index in xrange(test_set_x.value.shape[0]):
        c = []
        for batch_index in xrange(n_train_batches):
            batchDistances = calc_distances(batch_index, test_index)
            batchKNN = [(batch_index * batch_size + i, batchDistances[i]) \
                            for i in batchDistances.argsort()[:k] ]
            c.extend(batchKNN)
        c.sort(key=itemgetter(1))
        kNN.append(c[:k])
        
    end_time = time.clock()
     
    tN = 200; X = []
    for test_index in xrange(test_set_x.value.shape[0]):
        X.append(test_set_x.value[test_index])
        for i in kNN[test_index]: X.append(train_set_x.value[i[0]])
        if len(X) >= tN: break
        
    arr = tile_raster_images( X = numpy.asarray(X),
                                  img_shape = (A_DIM, B_DIM),tile_shape = (tN//10,10),
                                  tile_spacing=(1,1), scale_rows_to_unit_interval = False,
                                  output_pixel_vals = True)
    PIL.Image.fromarray(arr).save('mnist-kNN.png')

   
    #for t in kNN: print t
    training_time = (end_time - start_time)
    print >> sys.stderr, ('The kNN search code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((training_time)/60.))
    os.chdir('../')


if __name__ == "__main__":    
    test_kNN()

