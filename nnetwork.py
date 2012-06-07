#!/usr/bin/env python
"""
a neural network implementation.

"""
from scipy import io
import pylab as plt
import numpy as np
import sys
#import pyximport; pyximport.install()
#from cy_sigmoid import cy_sigmoid
from nnopt import sigmoid2d, matmult
#number of iterations in training
_niter = 0

def load_pickle(fname):
    """
    recover from a pickled NeuralNetwork classifier

    """
    d = cPickle.load(open(fname,'r'))
    nlayers = d['nlayers']
    gamma = d['gamma']
    thetas = []
    for li in range(nlayers):
        thetas.append(d[li])

    nin = thetas[0].shape[0] - 1 # remove bias
    nout = thetas[-1].shape[1]   # no bias on output
    classifier = create_NN(nin, nout, thetas=thetas, gamma=gamma)
    return classifier

def create_NN(ninputs, nout, ninternal= np.array([16]), thetas=np.array([]),
              delta=0, gamma=0.5):
    """
    configure our neural network.
    The number of hidden layers is set by len(ninternal)

    arguments:
    ninputs: number of input attributes 
            **can be a numpy.array([nsamples, nproperties])
    nout: number of classifications
            **can be a numpy.array([nsamples]),
             then number of classifications = len(np.unique(nout))
    ninternal: [number of attributes for the each internal layer
               (without bias)]

    optionally initialize the Theta's in the l'th layer to your predefined array
      * needs len(thetas) == len(ninternal) + 1 
      * intial theta's should have bias term
      if thetas = [], randomly initialize the theta's uniformly over range
                      [-delta,delta], but then you should train your data
    gamma : learning rate (regularization parameter)
            defaults to .5
                  
    """
    if isinstance(ninputs, type(np.array([]))):
        ninputs = ninputs.shape[1]
    elif isinstance(ninputs, type([])):
        ninputs = np.array(ninputs).shape[1]
    if isinstance(nout, type(np.array([]))):
        nout = len(np.unique(nout))
    elif isinstance(nout, type([])):
        nout = len(np.unique(nout))

    nl = len(ninternal) + 1
    layers = []
    for idx in range(nl):
        if thetas:
            theta = thetas[idx]
        else:
            theta = np.array([])
        #input layer
        #add bias to all inputs
        #add bias to outputs for internal/hidden layers
        if idx == 0:
            lin = ninputs + 1 #add bias
        else:
            lin = ninternal[idx - 1] + 1 #add bias

        if idx == nl - 1:
            lout = nout
        else:
            lout = ninternal[idx] 

        print "Creating layer %s, (nin,nout) = (%s,%s) (with input bias)"\
            % (idx,lin, lout)
        layers.append(layer(lin, lout, theta))

    return NeuralNetwork(layers, gamma=gamma)


class layer(object):
    """
    a layer in the neural network. It takes 
    in N parameters and outputs m attributes

    N and m should include the bias term (if appropriate)
    
    output = Theta*trial_colvector

    optional:
    theta = the transformation matrix of size (N+1,m) (includes bias)
          otherwise randomly initialize theta
          we uniformly initialize theta over [-delta,delta]
          where delta = sqrt(6)/sqrt(N+m), or passed as argument

    """
    def __init__(self, n, m, theta=np.array([]), delta=0):
        # add one row for bias
        self.input = n
        self.output = m
        if len(theta):
            self.theta = theta
        else:
            # use random initialization
            if not delta:
                delta = np.sqrt(6)/np.sqrt(n + m)
            self.randomize(delta)


    def randomize(self, delta=None):
        """
        randomize the theta's for each 'fit' call in the neural network
        """
        N = self.input
        m = self.output
        if not delta:
            delta = np.sqrt(6)/np.sqrt( N + m)
        self.theta = np.random.uniform(-delta, delta, N*m).reshape(N, m)


def main():
    """
    main routine...
    """
################
# load the data 
    data = handwritingpix(samples='ex4data1.mat',
                          thetas='ex4weights.mat'                              
                          )
################
# show some samples of the data
    data.plot_samples(15)


###############
# create the neural network, one needs to set the number of inputs attributes
# (minus bias)
# and the number of outputs (classifications)
    nin = data.data.shape[1]
    nout = len(np.unique(data.y))

    nn = create_NN(nin, nout, thetas=data.thetas, gamma=0.)


class NeuralNetwork(object):
    """
    a collection of layers forming the neural network

    Notes:
    assumes labels are from [0,nclasses)
    gamma = learning rate (regularization parameter)

    """
    def __init__(self, layers, gamma=.5):
        self.layers = layers
        self.nclasses = layers[-1].theta.shape[-1]
        self.gamma = gamma
        self.nthetas = len(self.flatten_thetas())

    def unflatten_thetas(self, thetas):
        """
        in order to use scipy.fmin functions we 
        need to make the Theta dependance explicit.
        
        This routine takes a flattened array 'thetas' of
        all the internal layer 'thetas', then assigns them
        onto their respective layers
        (ordered from earliest to latest layer)

        """
        bi = 0
        for lv in self.layers:
            shape = lv.theta.shape
            ei = bi + shape[0] * shape[1]
            lv.theta = thetas[bi:ei].reshape(shape)
            bi = ei

    def flatten_thetas(self):
        """
        in order to use scipy.fmin functions we
        need to make the layer's Theta dependencies explicit.

        this routine returns a giant array of all the flattened
        internal theta's, from earliest to latest layers

        """
        z = np.array([])
        for lv in self.layers:
            z = np.hstack([z, lv.theta.flatten()])
        return z

    def costFunctionU(self, X, y, gamma=None):
        """
        routine which calls costFunction, but
        unwraps the internal parameters (theta's)
        for you

        """
        if not gamma:
            gamma = self.gamma
        thetas = self.flatten_thetas()
        return self.costFunction(thetas, X, y, gamma)

    def costFunction(self, thetas, X, y, gamma=None):
        """
        determine the cost function for this neural network
        given the training data X, classifcations y, 
        and learning rate (regularization) lambda

        Arguments:
        X : num_trials x ninputs, ninputs not including bias term
        return cost
        y : classification label for the num_trials

        """
        global _niter
        if isinstance(X, type([])):
            X = np.array(X)
        
        if not gamma:
            gamma = self.gamma
# testing: check the theta's are changing while we train: yes!
#        print "CF",thetas[0:2],thetas[-5:-2]

# update the layer's theta's
        self.unflatten_thetas(thetas)
        
        # number of trials
        N = X.shape[0] 

        # propagate the input through the entire network
        z, h = self.forward_propagate(X)
        yy = labels2vectors(y, self.nclasses)

        J = (-np.log(h) * yy.transpose() - np.log(1-h)*(1-yy.transpose())).sum()
        J = J/N

# regularize (ignoring bias):
        reg = 0.
        for l in self.layers:
            reg +=  (l.theta[1:, :]**2).sum()
        J = J + gamma*reg/(2*N)

        sys.stdout.write("\r training Iteration %s, Cost %10.6f " % (_niter, J))
        sys.stdout.flush()
        _niter += 1
        return J

    def forward_propagate(self, z, nl=100):
        """
        given an array of samples [nsamples, nproperties]
        propagate the sample through the neural network,
        Returns:
        z, a : the ouputs and activations (z(nl),acts(nl)) at layer nl
        
        Args:
        X = [nsamples, nproperties] (no bias)
        nl = number of layers to propagte through
             defaults to end (well, one hundred layers!)
        """
        if isinstance(z, type([])):
            z = np.array(z)
        # number of trials
        if len(z.shape) == 2:
            N = z.shape[0] 
            # add bias
            a = np.hstack([np.ones((N, 1)), z])
        else:
            N = 1
            # add bias
            a = np.hstack([np.ones(N), z])

        final_layer = len(self.layers) - 1
        for li, lv in enumerate(self.layers[0:nl]):
            z = np.dot(a, lv.theta)
            # add bias to input of each internal layer
            if li != final_layer:
                if N == 1:
                    a = np.hstack([np.ones(N), sigmoid(z)])
                else:
                    a = np.hstack([np.ones((N, 1)), sigmoid2d(z)])
            else:
                if N == 1:
                    a = sigmoid(z)
                else:
                    a = sigmoid2d(z)
        return z, a

    def gradientU(self, X, y, gamma=None):
        """
        Convenience function.
        routine which calls gradient, but
        unwraps the internal parameters (theta's)
        for you.

        needed for scipy.fmin functions

        """
        if not gamma:
            gamma = self.gamma
        thetas = self.flatten_thetas()
        return self.gradient(thetas, X, y, gamma)
        
    def gradient(self, thetas, X, y, gamma=None):
        """
        compute the gradient at each layer of the neural network
        
        Args:
        X = [nsamples, ninputs]
        y = [nsamples] #the training classifications
        gamma : regularization parameter

        returns the gradient for the parameters of the neural network
        (the theta's) unrolled into one large vector, ordered from
        the first layer to latest.

        """
        if isinstance(X, type([])):
            X = np.array(X)

        if not gamma:
            gamma = self.gamma

        N = X.shape[0]
        nl = len(self.layers)
# create our grad_theta arrays (init to zero):
        grads = {}
        for li, lv in enumerate(self.layers):
            grads[li] = np.zeros_like(lv.theta)
            
# vectorize sample-loop
        if 1:
            for li in range(nl, 0, -1):
                z, a = self.forward_propagate(X,li)

                if li == nl:
                    ay = labels2vectors(y, self.nclasses).transpose()
                    delta = (a - ay)
                else:
                    theta = self.layers[li].theta
                    aprime = np.hstack([np.ones((N,1)), sigmoidGradient(z)]) #add in bias
                    tmp = np.dot(deltan,theta.transpose())#nsamples x neurons(li)
                    delta = tmp*aprime
                    
#find contribution to grad
                idx = li - 1
                z, a = self.forward_propagate(X,idx)
                if idx in grads:
                    if li == nl:
                        grads[idx] = np.dot(a.transpose(), delta)/N
                    else:
                        #strip off bias
                        grads[idx] = np.dot(a.transpose(), delta[:,1:])/N
#keep this delta for the next (earlier) layer
                if li == nl:
                    deltan = delta
                else:
                    deltan = delta[:,1:]

# old, non-vectorized implementation                     
# loop over samples
        if 0:
          for si, sv in enumerate(X):

# loop over layers (latest to earliest, no grad on first layer)
            for li in range(nl, 0, -1):
                z, a = self.forward_propagate(sv, li)

                if li == nl:
                    ay = labels2vectors(y[si], self.nclasses).transpose()
                    delta = (a - ay)
                else:
                    theta = self.layers[li].theta
# requires delta from next layer (hence reverse loop over layers)
                    aprime = np.hstack([1, sigmoidGradient(z)]) #add in bias

# only use fortan.matmult if arrays are gigantic
                    if deltan.ndim == 1:
                        delta = (aprime * np.dot(theta, deltan))
                    else:
                        if theta.size > 5000:
                            tmp = matmult(theta, deltan)
                            delta = (aprime * tmp)
                        else:
                            delta = (aprime * np.dot(theta, deltan))

# add this sample's contribution to the gradient:
                idx = li - 1
                z, a = self.forward_propagate(sv, idx)
                if idx in grads:
                    if li == nl:
                        grads[idx] += np.outer(a, delta)/N
                    else:
                        #strip off bias
                        grads[idx] += np.outer(a, delta[1:])/N

#keep this delta for the next (earlier) layer
                if li == nl:
                    deltan = delta
                else:
                    deltan = delta[1:]


#now regularize the grads (bias doesn't get get regularized):
        for li, lv in enumerate(self.layers):
            theta = lv.theta
            grads[li][:, 1:] = grads[li][:, 1:] + gamma/N*theta[:, 1:]
            
#finally, flatten the gradients
        z = np.array([])
        for k in sorted(grads):
            v = grads[k]
            z = np.hstack([z, v.flatten()])
        return z

    def fit(self, X, y, gamma=None, maxiter=42, epsilon=1.e-7,
            gtol=6.e-4, raninit=True):
        """
        Train the data.
        minimize the cost function (wrt the Theta's)
        (using the conjugate gradient algorithm from scipy)
        This updates the NN.layers.theta's, so one can
        later "predict" other samples.

        Args:
        X : the training samples [nsamples x nproperties]
        y : the sample labels [nsamples], each entry in range 0<=y<nclass
        
        *for scipy.optimize.fmin_cg:
        maxiter
        epsilon
        gtol

        raninit : T/F randomly initialize the theta's [default = True]
   
        """
        global _niter
        _niter = 0
        from scipy.optimize import fmin_cg

        if raninit:
            for lv in self.layers:
                lv.randomize()

        thetas = self.flatten_thetas()
        xopt = fmin_cg(f=self.costFunction,
                       x0=thetas,
                       fprime=self.gradient,
                       args=(X, y, gamma), #extra args to costFunction 
                       maxiter=maxiter,
                       epsilon=epsilon,
                       gtol=gtol,
#                       callback=self.unflatten_thetas
                       )
        return xopt
        

    def predict(self, X):
        """
        Given a list of samples, predict their class.
        One should run nn.fit first to train the neural network.

        Args:
        X = [nsamples x nproperties]

        returns:
        y = [nsamples]
        
        """
        if isinstance(X, type([])):
            X = np.array(X)
        if len(X.shape) == 2:
            N = X.shape[0] 
        else:
            N = 1

        z, h  = self.forward_propagate(X)
        #find most-active label
        if N == 1:
            cls = np.array([h.argmax()])
        else:
            cls = h.argmax(axis=1)
        return cls

    def numericalGradients(self, X, y):
        """
        numerically estimate the gradients using finite differences
        (used to compare to 'gradient' routine)

        loop over layers, perturbing each theta-parameter one at a time


        * useful for testing gradient routine *
        """
        from copy import deepcopy

        thetas = self.flatten_thetas()
        origthetas = deepcopy(thetas)
        numgrad = np.zeros(thetas.size)
        perturb = np.zeros(thetas.size)

        delta = 1.e-4

        for p in range(numgrad.size):
            #set the perturbation vector
            perturb[p] = delta
            loss1 = self.costFunction(thetas - perturb, X, y)
            loss2 = self.costFunction(thetas + perturb, X, y)
            #calculat the numerical gradient
            numgrad[p] = (loss2 - loss1) / (2*delta)
            #reset the perturbation
            self.unflatten_thetas(origthetas)
            perturb[p] = 0
            
## OLD
# loop over layers, neurons
        idx = 0
        if 0: 
            for lv in self.layers:
                theta_orig = deepcopy(lv.theta)

# perturb each neuron and calc. grad at that neuron
                for pi in range(theta_orig.size): #strip bias
                    perturb = np.zeros(theta_orig.size)
                    perturb[pi] = gamma
                    perturb = perturb.reshape(theta_orig.shape)
                    lv.theta = theta_orig + perturb
                    loss1 = self.costFunctionU(X, y)
                    lv.theta = theta_orig - perturb
                    loss2 = self.costFunctionU(X, y)
                    numgrad[idx] = (loss2 - loss1) / (2*gamma)
                    idx += 1
                lv.theta = theta_orig

        return numgrad

   def pickle_me(self, filename=''):
        """
        dump the important parts of the classifier to a pickled file.
        (the theta's and gamma=learning rate)
        We name the file based on number of layers,
        and shape of layers
        
        args:
        filename : base filename, append layer info to this

        """
        for li, lv in enumerate(self.layers):
            if filename:
                filename += '_l%sx%s' % lv.theta.shape
            else:
                filename += 'l%sx%s' % lv.theta.shape
        filename += '.pkl'
#create our pickle
        d = {}
        d['nlayers'] = len(self.layers)
        for li, lv in enumerate(self.layers):
            d[li] = lv.theta
        d['gamma'] = self.gamma
        
        print "pickling classifier to %s" % filename
        cPickle.dump(d, open(filename, 'w'))
    

    def write_thetas(self, basename='layer_'):
        """
        Write the theta's to a file in form
        basename_%s.npy % layer_number

        args
        basename : the base filename to write the data to
       
        """
        
        for li, lv in enumerate(self.layers):
            theta = lv.theta
            name = "%s_%s.npy" % (basename, li)
            print "Saving layer %s to %s " % (li, name)
            np.save(name, theta)
        
############# end class NeuralNetwork ***************


def checkGradients(nin=3, nout=3, ninternal=np.array([5]),
                   Nsamples=5, gamma=0.):
    """
    Create a small neural network to check
    backpropagatoin gradients

    this routine compares analytical gradients to the numerical gradients
    (as computed by compute_numericalGradients)

    returns
    numerical_gradient, gradient
    
    """
    print "Creating small NN to compare numerical approx to gradient"
    print "with actual gradient. Arrays should be identical"

#create the neural network and some fake data
    X = np.random.random((Nsamples, nin))
#labels from 0 <= y < nlabels = nout
    y =  np.array(np.random.uniform(0, nout, Nsamples), dtype=int)

# neural network, delta=0 and thetas=[] --> theta is randomly inited.
    nn = create_NN(nin, nout, ninternal=ninternal, thetas=np.array([]),
                   delta=0)
    numgrad = nn.numericalGradients(X, y)
    grad = nn.gradientU(X, y, gamma)
    return numgrad, grad
        
class handwritingpix(object):
    """
    read in the machine learning data files, which we 
    use to test our python implementation of the neural network

    we assume they are 20x20 pixels 
    initializes with a filename, assumes matlab format
    though can have text=True for text format
    then requires the trained values 'y'
    """
    def __init__(self, samples, thetas):

        self.fname = samples

        data = io.loadmat(samples)
        self.data = data['X']
        self.y = data['y']
# make sure the labels are in range 0<=y<nclass 
# so we can easily index our row vector
        self.y = self.y - self.y.min()
        thetas = io.loadmat(thetas)
        self.theta1 = thetas['Theta1'].transpose()
        self.theta2 = thetas['Theta2'].transpose()
        shp = self.data.shape
#assume square image
        self.N = self.data.shape[1]
        self.Nsamples = self.data.shape[0]
        self.nx = np.sqrt(self.N)
        self.ny = np.sqrt(self.N)
        self.thetas = [self.theta1, self.theta2]

    def plot_samples(self, Nsamples=10):
        """
        randomly pick Nsamples**2 and plot them

        """
        nx = self.nx
        ny = self.ny

        n = np.random.uniform(0, self.Nsamples, Nsamples**2).astype(np.int)
        samples = self.data[n, :]
        
        data = np.zeros((Nsamples*ny, Nsamples*nx))
        print n
        for xi, xv in enumerate(samples):
            col = xi % Nsamples
            row = xi // Nsamples
#            print xi,data.shape,row,col,xv.shape
            data[row*ny:(row+1)*ny, col*nx:(col+1)*nx] = xv.reshape(20, 20)
            
        plt.imshow(data, cmap=plt.cm.gray)
        plt.show()
        


#### Utility Functions ####
def labels2vectors(y, Nclass=0):
    """
    given a vector of [nsamples] where the i'th entry is label/classification
    for the i'th sample, return an array [nlabels,nsamples],
    projecting each sample 'y' into the appropriate row

    args:
    y : [nsamples] 
    Nclass: the number of classifications,
            defaults to number of unique items in y

    **assumes classifications are in range 0 <= y < Nclass
    """
    if isinstance(y, np.array([]).__class__):
        pass
    else:
        y = np.array([y], dtype=np.int)

# number of samples
    N = y.size

# determine number of classes
    if Nclass:
        nclass = Nclass
    else:
        nclass = len(np.unique(y))


# map labels onto column vectors
    if N == 1:
        yy = np.zeros(nclass, dtype=np.uint8)
    else:
        yy = np.zeros((nclass, N), dtype=np.uint8)

    for yi, yv in enumerate(y):
        if N == 1:
            yy[yv] = 1
        else:
            yy[yv, yi] = 1
    return yy

def sigmoid(z):
    """
    compute element-wise the sigmoid of input array

    """
    return 1./(1.0 + np.exp(-z))

def sigmoidGradient(z):
    """
    compute element-wise the sigmoid-Gradient of input array

    """
    return sigmoid(z) * (1-sigmoid(z))
        
if __name__ == '__main__':
    main()
