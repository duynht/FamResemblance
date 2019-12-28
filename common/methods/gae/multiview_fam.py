#Copyright (c) 2012, Roland Memisevic
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import scipy.io
import sys
import argparse
import os
HOME = os.environ['HOME']

#import pylab
import numpy
import numpy.random
import gatedAutoencoder_quadraturepooling as gatedAutoencoder
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


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



def main(args):

    ifile = args.input
    ofile = args.output
    numfac = args.numfac
    nummap = args.nummap
    numepochs = args.numepochs
    doNorm = args.donorm
    batchsize = 500

    print '... loading data'
    mat = scipy.io.loadmat(ifile)

    train_features_x = numpy.float64(mat['x'])
    train_features_y = numpy.float64(mat['y'])
    
    
    patchsize = train_features_x.shape[1]

    if doNorm == 1:
        eps = 0
        train_features_x -= train_features_x.mean(0)[None, :]
        train_features_y -= train_features_y.mean(0)[None, :]
        train_features_x /= train_features_x.std(0)[None, :] + train_features_x.std() * 0.1 + eps
        train_features_y /= train_features_y.std(0)[None, :] + train_features_y.std() * 0.1 + eps
    
    train_features_numpy = numpy.concatenate((train_features_x, train_features_y), 1)
    numcases = train_features_numpy.shape[0]
    R = numpy.random.permutation(numcases)
    train_features_numpy = train_features_numpy[R[:numcases], :]
    train_features = theano.shared(train_features_numpy)

    print '... done'
    numbatches = train_features.get_value().shape[0] / batchsize


    # INSTANTIATE MODEL
    print '... instantiating model'
    numpy_rng  = numpy.random.RandomState(1)
    theano_rng = RandomStreams(1)
    model = gatedAutoencoder.FactoredGatedAutoencoder(numvisX=patchsize, numvisY=patchsize, numfac=numfac, nummap=nummap,
                                                      output_type='real', weight_decay_vis=0.1, weight_decay_map=0.0, subspacedims=2,
                                                      corruption_type="zero_mask", corruption_level=0.0,
                                                      numpy_rng=numpy_rng, theano_rng=theano_rng)

    print '... done'

    # TRAIN MODEL
    trainer = gatedAutoencoder.GraddescentMinibatch(model, train_features, batchsize, learningrate=0.01)
#    pylab.ion()
    for epoch in xrange(numepochs):
        trainer.step()
        if epoch % 5 == 0:
            trainer.set_learningrate(trainer.learningrate*0.8)
#            pylab.clf()
#            dispims(model.layer.wyf.get_value()[:patchsize**2, :], patchsize, patchsize, 1)

    scipy.io.savemat(ofile, {'wxf':model.layer.wxf.get_value(), 'wyf':model.layer.wyf.get_value(), 'whf':model.layer.whf_in.get_value(), 'wpf':model.layer.wpf.get_value(), 'bmap':model.layer.bmap.get_value()},oned_as='column');
    

    # TRAIN MODEL
#    try:
#        pylab.clf()
#        pylab.ioff()
#        pylab.subplot(1, 2, 1, title='Weights for X')
#        dispims(model.layer.wxf.get_value(), patchsize, patchsize, 2)
#        pylab.subplot(1, 2, 2, title='Weights for Y')
#        dispims(model.layer.wyf.get_value(), patchsize, patchsize, 2)
#        pylab.show()
#    except Exception:
#        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='store', help='input file')
    parser.add_argument('-o', '--output', action='store', help='output file')
    parser.add_argument('-f', '--numfac', action='store', help='factor num', type=int)
    parser.add_argument('-m', '--nummap', action='store', help='map num', type=int)
    parser.add_argument('-e', '--numepochs', action='store', help='Number of Iterations', type=int)
    parser.add_argument('-n', '--donorm', action='store', help='Enable Normalization', type=int)
    args = parser.parse_args()
    
    main(args)
