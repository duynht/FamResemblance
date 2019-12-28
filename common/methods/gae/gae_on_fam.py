#import pylab
import numpy
import numpy.random
import gatedAutoencoder
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import scipy.io
import sys
import argparse
from collections import OrderedDict


def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful 
        eg. to display the weights of a neural network layer.
    """
    from pylab import cm, ceil
    numimages = M.shape[1]
    if layout is None:
        n0 = int(numpy.ceil(numpy.sqrt(numimages)))
        n1 = int(numpy.ceil(numpy.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * numpy.ones(((height+border)*n0+border,(width+border)*n1+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] = numpy.vstack((
                            numpy.hstack((numpy.reshape(M[:,i*n1+j],(height, width)),
                                   bordercolor*numpy.ones((height,border),dtype=float))),
                            bordercolor*numpy.ones((border,width+border),dtype=float)
                            ))
    pylab.imshow(im, cmap=cm.gray, interpolation='nearest', **kwargs)
    pylab.draw()


class GraddescentMinibatch(object):
    """ Gradient descent trainer class.

    """

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
        self.momentum      = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * self.model.layer.learningrate_modifiers[_param.name] * _grad 
            self.updates[_param] = _param + self.incs[_param]

        givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]}
        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = OrderedDict(self.inc_updates),
                givens = OrderedDict(givens))
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)
            self.model.layer.normalizefilters()

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)

def main(args):
    
    ifile = args.input
    ofile = args.output
    numfac = args.numfac
    nummap = args.nummap
    numepochs = args.numepochs
    doNorm = args.donorm
    verbose = args.verbose
    learningrate = args.learnrate
    
    print '... loading data'

    mat = scipy.io.loadmat(ifile)

    train_features_x = numpy.float32(mat['x'])
    train_features_y = numpy.float32(mat['y'])

    #NORMALIZE DATA:
    if doNorm == 1:
        eps = 0
        train_features_x -= train_features_x.mean(0)[None, :]
        train_features_y -= train_features_y.mean(0)[None, :]
        train_features_x /= train_features_x.std(0)[None, :] + train_features_x.std() * 0.1 + eps
        train_features_y /= train_features_y.std(0)[None, :] + train_features_y.std() * 0.1 + eps

    #scipy.io.savemat('train_features_norm.mat',{'nx':train_features_x,'ny':train_features_y},oned_as='column')

    #SHUFFLE TRAINING DATA TO MAKE SURE ITS NOT SORTED:
    R = numpy.random.permutation(train_features_x.shape[0])
    train_features_x = train_features_x[R, :]
    train_features_y = train_features_y[R, :]


    print train_features_x.shape
    print train_features_y.shape
    train_features_numpy = numpy.concatenate((train_features_x, train_features_y), 1, dtype=numpy.float32)
    train_features = T.cast(theano.shared(train_features_numpy),'float32')
    print train_features.type
    print '... done'

    #numfac = 600
    #nummap = 400
    numhid = 0
    weight_decay_vis = 0.0
    weight_decay_map = 0.0
    corruption_type = 'zeromask'
    corruption_level = 0.5
    init_topology = None
    batchsize = 100
    numvisX = train_features_x.shape[1]
    numvisY = train_features_y.shape[1]
    numbatches = train_features.get_value().shape[0] / batchsize


    # INSTANTIATE MODEL
    print '... instantiating model'
    numpy_rng  = numpy.random.RandomState(1)
    theano_rng = RandomStreams(1)
    model = gatedAutoencoder.FactoredGatedAutoencoder(numvisX=numvisX, 
                                                      numvisY=numvisY, 
                                                      numfac=numfac, nummap=nummap, numhid=numhid,
                                                      output_type='real', 
                                                      weight_decay_vis=weight_decay_vis, 
                                                      weight_decay_map=weight_decay_map,
                                                      corruption_type=corruption_type, 
                                                      corruption_level=corruption_level, 
                                                      init_topology = init_topology, 
                                                      numpy_rng=numpy_rng, 
                                                      theano_rng=theano_rng)

    print '... done'



    # TRAIN MODEL
    #numepochs = 100
    #learningrate = 0.01
    #trainer = gatedAutoencoder.GraddescentMinibatch(model, train_features, batchsize, learningrate)
    trainer = GraddescentMinibatch(model, train_features, batchsize, learningrate, 0.9, None, verbose)
    for epoch in xrange(numepochs):
        trainer.step()
    

    scipy.io.savemat(ofile, {'wxf':model.layer.wxf.get_value(), 'wyf':model.layer.wyf.get_value(), 'whf':model.layer.whf_in.get_value(), 'z_bias':model.layer.bmap.get_value()},oned_as='column')


    # TRAIN MODEL
    #try:
    #    pylab.subplot(1, 2, 1)
    #    dispims(model.layer.wxf.get_value(), numpy.sqrt(train_features_x.shape[1]), numpy.sqrt(train_features_x.shape[1]), 2)
    #    pylab.subplot(1, 2, 2)
    #    dispims(model.layer.wyf.get_value(), numpy.sqrt(train_features_y.shape[1]), numpy.sqrt(train_features_y.shape[1]), 2)
    #    pylab.show()
    #except Exception:
    #    pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='store', help='input file')
    parser.add_argument('-o', '--output', action='store', help='output file')
    parser.add_argument('-f', '--numfac', action='store', help='factor num', type=int)
    parser.add_argument('-m', '--nummap', action='store', help='map num', type=int)
    parser.add_argument('-l', '--learnrate', action='store', help='Learning Rate', type=float)
    parser.add_argument('-e', '--numepochs', action='store', help='Number of Iterations', type=int)
    parser.add_argument('-n', '--donorm', action='store', help='Enable Normalization', type=int)
    parser.add_argument('-v', '--verbose', action='store', help='Enable Verbosity', type=int)
    args = parser.parse_args()

    main(args)
