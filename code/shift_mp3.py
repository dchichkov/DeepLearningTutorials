#!/usr/bin/python
# -*- coding: utf-8  -*-
from __future__ import division

import mad, numpy, random, os, time, cPickle, gzip, sys
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images
import matplotlib.pyplot
from collections import deque
from operator import itemgetter
import theano
import theano.tensor as T

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
   
def load_mp3(filename):
    ''' Loads the mp3 file to numpy array

    :type filename: string
    :param filename: the path to the mp3 file
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data. filename = ', filename
    mf = mad.MadFile(filename)    
    samples = numpy.zeros((mf.frames() * X_DIM, S_DIM), dtype=theano.config.floatX) 
    load_mp3.sample = 0;
        
    def fCallback(mf):
        channel = 0
        for frame_sample in xrange(X_DIM):
            for subband in reversed(xrange(S_DIM)):
                samples[load_mp3.sample,subband] = mf.subband_value(channel, frame_sample, subband)
            load_mp3.sample += 1
                
     # frequency domain filter
    mf.set_filter_callback(fCallback)
    while mf.read():
        pass
    
    print "frames = ", load_mp3.sample / X_DIM, "mf.frames() = ", mf.frames()   
    return samples    


def save_wave(samples, filename):
    ''' Saves samples (numpy array) to the wave file. 
        Uses the template mp3 file to set sampling rate. 

    :type filename: string
    :param filename: the path to the input mp3 file
                     the path to the output .cdr file
    '''
    
    save_wave.sample = 0;
    def fCallback(mf):
        for frame_sample in xrange(X_DIM):
            for subband in reversed(xrange(S_DIM)):
                mf.subband_value(0, frame_sample, subband, samples[save_wave.sample, subband])
                mf.subband_value(1, frame_sample, subband, samples[save_wave.sample, subband])
            for subband in reversed(xrange(S_DIM,32)):
                mf.subband_value(0, frame_sample, subband, 0.0)
                mf.subband_value(1, frame_sample, subband, 0.0)
                                    
            save_wave.sample += 1

        if len(samples) == save_wave.sample: 
            return False # to stop            
                                                           
     # frequency domain filter
    mf = mad.MadFile(filename)
    mf.set_filter_callback(fCallback)
    
    f = open(filename + ".cdr", 'wb')
    while 1:
        buffy = mf.read()
        if buffy is None:
            break
        f.write(buffy)


def shift_kNN( train_set_files,
               test_set_file,
               k = 9,
               distance = T.sqr,
               test_set_size = X_DIM*200000,
               output_folder = 'plots' ):

    #############################################
    # Exhaustive kNN queries for the test set   #
    #############################################
    
        # allocate symbolic variables for the data
    index  = T.lscalar()    # index to a test set item
    a = T.matrix('a')  # the data is presented as vectors
    b = T.vector('b')  # the data is presented as vectors

    # load test set
    test_set_samples = load_mp3(test_set_file)[:test_set_size]
    test_set_size = len(test_set_samples)
    test_set_x = theano.shared(test_set_samples)    

    
    # go through train set    
    kNN = [[] for test_index in xrange(test_set_size)]
    
    for filename in train_set_files:
        train_set_samples = load_mp3(filename)
        train_set_x = theano.shared(train_set_samples)
        
        # frames (X_DIM samples) 
        frameQueue = deque(maxlen = X_DIM)
        
        # define theano.function
        DistancesF = T.sum( distance(a - b), axis=1 )        
        calc_distances = theano.function([index], DistancesF,
             givens = {a:train_set_x[:], b:test_set_x[index]})

        start_time = time.clock()
        for test_index in xrange(test_set_size):
            sampleDistances = calc_distances(test_index)
            
            # accumulate frame distances
            for i, frameDistances in enumerate(frameQueue):
                 frameDistances[:-i-1] += sampleDistances[i+1:]             
            frameQueue.appendleft(sampleDistances)
            
            # find kNN frame            
            if len(frameQueue) == X_DIM:
                frameDistances = frameQueue.pop()
                #fileKNN = [(filename, i, frameDistances[i]) for i in frameDistances.argsort()[:k] ]
                #kNN[test_index].extend(fileKNN)
                i = frameDistances[:-X_DIM].argmin()
                kNN[test_index - X_DIM].append((filename, i, frameDistances[i]))
                 

            # print out time/ETA
            if(test_index and not test_index % 100):
                test_time = time.clock() - start_time + 0.00000001; 
                ETA = ((test_time / test_index) * test_set_size - test_time); 
                print "%.1f seconds; test %d (%d); ETA %.1f seconds" % (test_time, test_index, test_set_size, ETA)

    # sort for nearest
    for skNN in kNN: 
        skNN.sort(key=itemgetter(2))
        

    # construct and save o file
    X = numpy.zeros((test_set_size, S_DIM), dtype=theano.config.floatX)
    for filename in train_set_files:
        train_set_samples = load_mp3(filename)
        for test_index in xrange(test_set_size - X_DIM):
            (f, i, d) = kNN[test_index][0]
            if f != filename: continue
            X[test_index] = train_set_samples[i]   
            
    save_wave(X, filename = test_set_file)


if __name__ == "__main__":    
    import PIL.Image
    d = "/home/dmitry/mp3/01- Hitchhikers Guide to the Galaxy"
    train_set_files = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".mp3")])
    test_set_file = "/home/dmitry/mp3/hhgttg01010060.mp3"
    test_set_file = "/home/dmitry/mp3/advice_to_little_girls_twain_alnl.mp3"
    test_set_file = "mp3/test.mp3"

    shift_kNN(train_set_files[1:2], test_set_file)
