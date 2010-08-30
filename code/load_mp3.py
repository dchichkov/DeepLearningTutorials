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





def test_dA( learning_rate = 0.1, training_epochs = 15, dataset ='/home/dmitry/mp3/hhgttg01010060.mp3',
        batch_size = 20, output_folder = 'dA_plots' ):

    """
    This demo is tested on /home/dmitry/mp3/hhgttg01010060.mp2

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training 

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images

    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng        = numpy.random.RandomState(123)
    theano_rng = RandomStreams( rng.randint(2**30))

    da = dA(numpy_rng = rng, theano_rng = theano_rng, input = x,
            n_visible = S_DIM * F_DIM * X_DIM, n_hidden = 1000)

    cost, updates = da.get_cost_updates(corruption_level = 0.,
                                learning_rate = learning_rate)

    
    train_da = theano.function([index], cost, updates = updates,
         givens = {x:train_set_x[index*batch_size:(index+1)*batch_size]})
    
    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost '%epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((training_time)/60.))
    image = PIL.Image.fromarray(tile_raster_images( X = da.W.value.T,
                 img_shape = (S_DIM, F_DIM * X_DIM),tile_shape = (10,10), 
                 tile_spacing=(1,1)))
    image.save('filters_corruption_0.png') 
 
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng        = numpy.random.RandomState(123)
    theano_rng = RandomStreams( rng.randint(2**30))

    da = dA(numpy_rng = rng, theano_rng = theano_rng, input = x,
            n_visible = S_DIM * F_DIM * X_DIM, n_hidden = 1000)

    cost, updates = da.get_cost_updates(corruption_level = 0.3,
                                learning_rate = learning_rate)

    
    train_da = theano.function([index], cost, updates = updates,
         givens = {x:train_set_x[index*batch_size:(index+1)*batch_size]})

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost '%epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 30% corruption code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % (training_time/60.))

    image = PIL.Image.fromarray(tile_raster_images( X = da.W.value.T,
                 img_shape = (S_DIM, F_DIM * X_DIM),tile_shape = (10,10), 
                 tile_spacing=(1,1)))
    image.save('filters_corruption_30.png') 
 
    os.chdir('../')


if __name__ == "__main__":    
    from dA import dA
    d = "/home/dmitry/mp3/01- Hitchhikers Guide to the Galaxy"
    dataset = sorted([os.path.join(d, f) for f in os.listdir(d)])
    dataset = ["/home/dmitry/mp3/hhgttg01010060.mp3"]

    # lame --preset cbr 48kbit -m mono
    ((train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)) = \
        load_data(dataset[:1])

    #print "len(train_set_x)", len(train_set_x)
    tN = 200
    print "len(train_set_x.value.T)", len(train_set_x.value)
    for i in xrange(len(train_set_x.value)//tN):
        arr = tile_raster_images( X = train_set_x.value[i*tN:i*tN+tN],
                                  img_shape = (S_DIM, F_DIM * X_DIM),tile_shape = (tN//10,10),
                                  tile_spacing=(1,1), scale_rows_to_unit_interval = False,
                                  output_pixel_vals = False)
        matplotlib.pyplot.imsave(fname = 'hhgttg010100%d.png' % i, arr = arr)


    quit()
    test_dA(dataset = dataset[:1], training_epochs = 500)
