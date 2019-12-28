#Copyright (c) 2012, Roland Memisevic
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import numpy
#import pylab

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

SMALL = 0.000001


class FactoredGatedNetworkLayer(object):
    """ This class defines a single factored gated network layer. """
    
    def __init__(self, inputs, numvisX, numvisY, numfac, nummap, output_type,
                 corruption_type='zeromask', corruption_level=0.0,
                 weight_decay_vis=0.0, weight_decay_map=0.0,
                 subspacedims=2, filternormcutoff=1.0,
                 numpy_rng=None, theano_rng=None, params=None):
        self.numvisX = numvisX
        self.numvisY = numvisY
        self.numfac  = numfac
        self.nummap  = nummap
        self.output_type  = output_type
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.weight_decay_vis = weight_decay_vis
        self.weight_decay_map = weight_decay_map
        self.subspacedims = subspacedims
        self.numpool = self.numfac / subspacedims
        
        self.filternormcutoff = filternormcutoff
        
        self.meannxf = 0.01
        self.meannyf = 0.01
        self.meannhf = 0.01
        self.meannhf_in = 0.01
        
        self.numxf = self.numvisX * self.numfac
        self.numyf = self.numvisY * self.numfac
        self.numhf = self.nummap * self.numfac
        
        if not numpy_rng:
            numpy_rng = numpy.random.RandomState(1)
        
        if not theano_rng:
            theano_rng = RandomStreams(1)
    
        # SET UP VARIABLES AND PARAMETERS
        if params is None:
            wxf_init = numpy_rng.normal(size=(numvisX, numfac)).astype(theano.config.floatX)*0.01
            wyf_init = numpy_rng.normal(size=(numvisY, numfac)).astype(theano.config.floatX)*0.01
            
            #assert numfac == subspacedims * nummap, "numfac has to equal subspacedims*nummap"
            self.whf_init = numpy.random.rand(nummap, self.numpool).astype(theano.config.floatX)*0.01
            self.wpf_init = numpy.zeros((self.numpool, numfac)).astype(theano.config.floatX)
            for i in range(self.numpool):
                self.wpf_init[i, subspacedims*i:subspacedims*i+subspacedims] = 1.0
            self.topomask = (self.whf_init > 0.0).astype(theano.config.floatX)
            self.whf = theano.shared(value = self.whf_init, name = 'whf')
            self.wpf = theano.shared(value = self.wpf_init, name = 'wpf')
            self.whf_in = theano.shared(value = self.whf_init, name='whf_in')
            self.whf_in = self.whf
            self.wxf = theano.shared(value = wxf_init, name = 'wxf')
            self.bvisX = theano.shared(value = numpy.zeros(numvisX, dtype=theano.config.floatX), name='bvisX')
            self.wyf = theano.shared(value = wyf_init, name = 'wyf')
            self.bvisY = theano.shared(value = numpy.zeros(numvisY, dtype=theano.config.floatX), name='bvisY')
            self.bmap = theano.shared(value = numpy.zeros(nummap, dtype=theano.config.floatX), name='bmap')
            self.inputs = inputs
        else:
            self.wxf = params['wxf']
            self.wyf = params['wyf']
            self.whf_in = params['whf_in']
            self.whf = params['whf']
            self.bvisX = params['bvisX']
            self.bvisY = params['bvisY']
    
        self.params = [self.wxf, self.wyf, self.whf_in, self.whf, self.bmap, self.bvisX, self.bvisY]

        # DEFINE THE LAYER FUNCTION
        self.inputsX = self.inputs[:, :numvisX]
        self.inputsY = self.inputs[:, numvisX:]
        if self.corruption_level > 0.0:
            if self.corruption_type=='zeromask':
                self._corruptedX = theano_rng.binomial(size=self.inputsX.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsX
                self._corruptedY = theano_rng.binomial(size=self.inputsY.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsY
            elif self.corruption_type=='gaussian':
                self._corruptedX = theano_rng.normal(size=self.inputsX.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputsX
                self._corruptedY = theano_rng.normal(size=self.inputsY.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputsY
            elif self.corruption_type=='none':
                self._corruptedX = self.inputsX
                self._corruptedY = self.inputsY
            else:
                assert False, "unknown corruption type"
        else:
            self._corruptedX = self.inputsX
            self._corruptedY = self.inputsY


        # Encoder
        self._factorsX = T.dot(self._corruptedX, self.wxf)
        self._factorsY = T.dot(self._corruptedY, self.wyf)
        self._poolings = T.dot(T.nnet.softplus(self._factorsX*self._factorsY), self.wpf.T)
        self._mappings = T.nnet.sigmoid(T.dot(self._poolings, self.whf_in.T)+self.bmap)

        #Decoder
        self._factorsH = T.dot(T.dot(self._mappings, self.whf), self.wpf)
        self._outputX_acts = T.dot(self._factorsY*self._factorsH, self.wxf.T)+self.bvisX
        self._outputY_acts = T.dot(self._factorsX*self._factorsH, self.wyf.T)+self.bvisY
        self._mappingsandhiddens = self._mappings

        
        if self.output_type == 'binary':
            self._reconsX = T.nnet.sigmoid(self._outputX_acts)
            self._reconsY = T.nnet.sigmoid(self._outputY_acts)
        elif self.output_type == 'real':
            self._reconsX = self._outputX_acts
            self._reconsY = self._outputY_acts
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"

        # DEFINE WEIGHTCOST, WHICH IS USEFUL TO PASS ON TO MODEL?
        self._weightcost = self.weight_decay_vis*( (self.wxf**2).sum() + (self.wyf**2).sum()) \
            + self.weight_decay_map*(self.whf**2).sum()\
            + self.weight_decay_map*(self.whf_in**2).sum()

        # COMPILE SOME FUNCTIONS THAT MAY BE USEFUL LATER
        self.corruptedX = theano.function([self.inputs], self._corruptedX)
        self.corruptedY = theano.function([self.inputs], self._corruptedY)
        self.mappings = theano.function([self.inputs], self._mappings)
        self.poolings = theano.function([self.inputs], self._poolings)
        self.reconsX = theano.function([self.inputs], self._reconsX)
        self.reconsY = theano.function([self.inputs], self._reconsY)

    def get_hidprobs(self, X, Y):
        numbatches = X.shape[0] / self.batchsize
        hiddens = []
        for batch in range(numbatches + int(numpy.mod(X.shape[0], self.batchsize)>0)):
            hiddens.append(self.get_hidprobs_batch(X[batch*self.batchsize:(batch+1)*self.batchsize], Y[batch*self.batchsize:(batch+1)*self.batchsize]))
        return numpy.concatenate(hiddens, 0)
    
    def get_hidprobs_batch(self, X, Y):
        numcases = X.shape[0]
        if numcases < self.batchsize:
            X = numpy.concatenate((X, numpy.zeros((self.batchsize-X.shape[0], self.numin), "single")), 0)
            Y = numpy.concatenate((Y, numpy.zeros((self.batchsize-Y.shape[0], self.numout), "single")), 0)
        X_ = cm.CUDAMatrix(X)
        Y_ = cm.CUDAMatrix(Y)
        cm.dot(X_, self.Wxf, self.actsx)
        cm.dot(Y_, self.Wyf, self.actsy)
        self.infer_hids(X_, Y_, samplehid=False)
        return self.z_probs.asarray()[:numcases, :].copy()
    
    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x
        
        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def normalizefilters(self, center=True, normalize_map_filters=False):
        def inplacemult(x, v):
            x[:, :] *= v
            return x
        def inplacesubtract(x, v):
            x[:, :] -= v
            return x
        nwxf = (self.wxf.get_value().std(0)+SMALL)[numpy.newaxis, :]
        nwyf = (self.wyf.get_value().std(0)+SMALL)[numpy.newaxis, :]
        nwhf = (numpy.sqrt((self.whf.get_value()**2).sum(0))+SMALL)[numpy.newaxis, :]
        nwhf_in = (numpy.sqrt((self.whf_in.get_value()**2).sum(0))+SMALL)[numpy.newaxis, :]
        self.meannxf = 1.0 * nwxf.mean()
        self.meannyf = 1.0 * nwyf.mean()
        self.meannhf = 1.0 * nwhf.mean()
        self.meannhf_in = 1.0 * nwhf_in.mean()
        if self.meannxf > self.filternormcutoff:
            self.meannxf = self.filternormcutoff
        if self.meannyf > self.filternormcutoff:
            self.meannyf = self.filternormcutoff
        if self.meannhf > self.filternormcutoff:
            self.meannhf = self.filternormcutoff
        if self.meannhf_in > self.filternormcutoff:
            self.meannhf_in = self.filternormcutoff

        wxf = self.wxf.get_value(borrow=True)
        wyf = self.wyf.get_value(borrow=True)
        whf = self.whf.get_value(borrow=True)
        whf_in = self.whf_in.get_value(borrow=True)
        # CENTER FILTERS
        if center:
            self.wxf.set_value(inplacesubtract(wxf, wxf.mean(0)[numpy.newaxis,:]), borrow=True)
            self.wyf.set_value(inplacesubtract(wyf, wyf.mean(0)[numpy.newaxis,:]), borrow=True)
        #if normalize_map_filters:
        #    self.whf.set_value(inplacesubtract(whf, whf.mean(0)[numpy.newaxis,:]), borrow=True)
        # FIX STANDARD DEVIATION
        self.wxf.set_value(inplacemult(wxf, self.meannxf/nwxf),borrow=True)
        self.wyf.set_value(inplacemult(wyf, self.meannyf/nwyf),borrow=True)
        if normalize_map_filters:
            #assert False, "if you want to normalize whf-filters, uncomment the lines above that compute the whf stats"
            self.whf.set_value(inplacemult(whf, self.meannhf/nwhf),borrow=True)
            self.whf_in.set_value(inplacemult(whf_in, self.meannhf_in/nwhf_in),borrow=True)


class FactoredGatedAutoencoder(object):
    """ This class defines a full model including cost function, using factoredGatedNetworkLayer as component."""
    def __init__(self, numvisX, numvisY, numfac, nummap, output_type,
                 corruption_type='zeromask', corruption_level=0.0,
                 weight_decay_vis=0.0, weight_decay_map=0.0,
                 subspacedims=2, filternormcutoff=0.1,
                 inputs=None, numpy_rng=None, theano_rng=None):
        self.numvisX = numvisX
        self.numvisY = numvisY
        self.numfac  = numfac
        self.nummap  = nummap
        self.output_type  = output_type
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.weight_decay_vis = weight_decay_vis
        self.weight_decay_map = weight_decay_map
        if inputs is None:
            self.inputs = T.matrix(name = 'inputs')
        else:
            self.inputs = inputs
    
        self.numxf = self.numvisX * self.numfac
        self.numyf = self.numvisY * self.numfac
        self.numhf = self.nummap * self.numfac
    
        if not numpy_rng:
            numpy_rng = numpy.random.RandomState(1)
        
        if not theano_rng:
            theano_rng = RandomStreams(1)
    
        # SET UP MODEL USING LAYER
        self.layer = FactoredGatedNetworkLayer(self.inputs, numvisX, numvisY, numfac, nummap, output_type,
                                               corruption_type=self.corruption_type, corruption_level=corruption_level,
                                               weight_decay_vis=weight_decay_vis, weight_decay_map=weight_decay_map,
                                               subspacedims=subspacedims, filternormcutoff=filternormcutoff,
                                               numpy_rng=numpy_rng, theano_rng=theano_rng)
        self.params = self.layer.params
        self._reconsX = self.layer._reconsX
        self._reconsY = self.layer._reconsY
        
        # ATTACH COST FUNCTIONS
        if self.output_type == 'binary':
            self._costpercase = - T.sum(
                                        0.5* (self.layer.inputsY*T.log(self._reconsY) + (1.0-self.layer.inputsY)*T.log(1.0-self._reconsY))
                                        +0.5* (self.layer.inputsX*T.log(self._reconsX) + (1.0-self.layer.inputsX)*T.log(1.0-self._reconsX)),
                                        axis=1)
        elif self.output_type == 'real':
            self._costpercase = T.sum(0.5*((self.layer.inputsX-self._reconsX)**2)
                                      +0.5*((self.layer.inputsY-self._reconsY)**2), axis=1)
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"
        self._weightcost = self.layer._weightcost
        self._cost = T.mean(self._costpercase) + self._weightcost
        self._cost_pure = T.mean(self._costpercase)
        self._grads = T.grad(self._cost, self.params)
    
        # COMPILE SOME FUNCTIONS THAT MAY BE USEFUL LATER
        self.mappings = theano.function([self.inputs], self.layer._mappings)
        self.reconsX = theano.function([self.inputs], self._reconsX)
        self.reconsY = theano.function([self.inputs], self._reconsY)
        self.cost = theano.function([self.inputs], self._cost)
        self.cost_pure = theano.function([self.inputs], self._cost_pure)
        self.grads = theano.function([self.inputs], self._grads)
        def get_cudandarray_value(x):
            if type(x)==theano.sandbox.cuda.CudaNdarray:
                return numpy.array(x.__array__()).flatten()
            else:
                return x.flatten()
        self.grad = lambda x: numpy.concatenate([get_cudandarray_value(g) for g in self.grads(x)])

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x
        
        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum
    
    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])
    
    def save(self, filename):
        numpy.save(filename, self.get_params())
    
    def load(self, filename):
        self.updateparams(numpy.load(filename))


class GraddescentMinibatch(object):
    
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
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad
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
