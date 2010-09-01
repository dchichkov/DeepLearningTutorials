#!/usr/bin/python
# -*- coding: utf-8  -*-
from __future__ import division

import mad, numpy, random, os, time, cPickle, gzip, sys
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images
import matplotlib.pyplot
import theano
import theano.tensor as T

F_DIM = 1   # 2  frames
X_DIM = 36  # 36 samples
S_DIM = 8   # 8  subbands


#        SAMPLES [MDCTed]
#   S  ------------------->
#   U  |  
#   B  |   amplitudes 
#   B  |   in colors
#   A  |  blue ... red
#   N  |
#   D  V
#   S
#   
   
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (ignored)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'
    
    
    frames = []
    def fCallback(mf):
        frame = []
        channel = 0
        for sample in xrange(X_DIM):
            for subband in reversed(xrange(S_DIM)):
                frame.append((mf.subband_value(channel, sample, subband) + 1.0) / 2.0)
                # fake it
                #frame.append(subband / S_DIM * len(frames) / 2294)
        frames.append(frame)
                   
     # frequency domain filter
    for f in dataset:
        mf = mad.MadFile(f)
        mf.set_filter_callback(fCallback)
        while mf.read():
            pass
    
    # corpus size
    (TRAIN, VALID, TEST) = (60, 1, 1)
    SIZE = len(frames) // (TRAIN+VALID+TEST)
    (TRAIN, VALID, TEST) = (TRAIN * SIZE, VALID * SIZE, TEST * SIZE)
    print 'frames =', len(frames), 'TRAIN = ', TRAIN
    

    
    corpus = []    
    for i in xrange(TRAIN + VALID + TEST):        
        x = frames[i * F_DIM : i * F_DIM + F_DIM]
        # concatenate frames, set label
        corpus.append((sum(x, []), i))            
    

    # make train/valid/test datasets
    train_set = ([corpus[i][0] for i in xrange(0, TRAIN)], 
                 [corpus[i][1] for i in xrange(0, TRAIN)] )
    valid_set = ([corpus[i][0] for i in xrange(TRAIN, TRAIN + VALID)], 
                 [corpus[i][1] for i in xrange(TRAIN, TRAIN + VALID)] )
    test_set = ([corpus[i][0] for i in xrange(TRAIN + VALID, TRAIN + VALID + TEST)],
                [corpus[i][1] for i in xrange(TRAIN + VALID, TRAIN + VALID + TEST)] )
    
    print 'input = ', len(train_set[0][0])
    #for i in xrange(F_DIM  * X_DIM * S_DIM):
    #    print train_set[0][0][i],
    #    if not (i+1)%S_DIM: print
    #print 'label =', train_set[1][0]

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
    d = "/home/dmitry/mp3/01- Hitchhikers Guide to the Galaxy"
    dataset = sorted([os.path.join(d, f) for f in os.listdir(d)])
    dataset = ["/home/dmitry/mp3/hhgttg01010060.mp3"]

    # lame --preset cbr 48kbit -m mono
    ((train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)) = \
        load_data(dataset[:1])
        
    from pca import pca
    V,S,immean = pca(train_set_x.value[0:200])
    train_set_x.value = V
    print V
    
    #matplotlib.pyplot.imshow(V[0].reshape(S_DIM, F_DIM * X_DIM).T)
    #matplotlib.pyplot.imshow(train_set_x.value[0].reshape(S_DIM, F_DIM * X_DIM), vmin = 0.0, vmax = 1.0)
    #matplotlib.pyplot.show()
       
    #print "len(train_set_x)", len(train_set_x)
    tN = 200
    print "len(train_set_x.value.T)", len(train_set_x.value)
    for i in xrange(len(train_set_x.value)//tN):
        arr = tile_raster_images( X = train_set_x.value[i*tN:i*tN+tN],
                                  img_shape = (S_DIM, F_DIM * X_DIM),tile_shape = (tN//10,10),
                                  tile_spacing=(1,1), scale_rows_to_unit_interval = False,
                                  output_pixel_vals = False)
        matplotlib.pyplot.imsave(fname = 'hhgttg010100%d.png' % i, arr = arr)#, vmin = 0.0, vmax = 1.0)
