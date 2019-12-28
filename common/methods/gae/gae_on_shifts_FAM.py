import pylab
import numpy
import numpy.random
import gatedAutoencoder
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import scipy.io


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
    pylab.show()


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

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
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


print '... loading data'
mat = scipy.io.loadmat('extended_crops1_max1.mat')

train_features_x = mat['vl']
train_features_y = mat['vr']

print train_features_x.shape
print train_features_y.shape

#NORMALIZE DATA:
train_features_x -= train_features_x.mean(0)[None, :]
train_features_y -= train_features_y.mean(0)[None, :]
train_features_x /= train_features_x.std(0)[None, :] + train_features_x.std() * 0.1
train_features_y /= train_features_y.std(0)[None, :] + train_features_y.std() * 0.1


#SHUFFLE TRAINING DATA TO MAKE SURE ITS NOT SORTED:
R = numpy.random.permutation(train_features_x.shape[0])
train_features_x = train_features_x[R, :]
train_features_y = train_features_y[R, :]


train_features_numpy = numpy.concatenate((train_features_x, train_features_y), 1)
train_features = theano.shared(train_features_numpy)
print train_features.shape
print '... done'

numfac = 100
nummap = 25
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
numepochs = 150
learningrate = 0.01
#trainer = gatedAutoencoder.GraddescentMinibatch(model, train_features, batchsize, learningrate)
trainer = GraddescentMinibatch(model, train_features, batchsize, learningrate)
for epoch in xrange(numepochs):
    trainer.step()


# TRAIN MODEL
#try:
    pylab.subplot(1, 2, 1)
    dispims(model.layer.wxf.get_value(), 13, 13, 2)
    pylab.subplot(1, 2, 2)
    dispims(model.layer.wyf.get_value(), 13, 13, 2)
#except Exception:
#    pass 


