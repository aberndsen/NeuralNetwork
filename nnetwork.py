#!/usr/bin/env python
"""
Aaron Berndsen (2012)

a neural network implementation in python, with f2py/fortran 
optimizations for large data sets

"""
import numpy as np
from scipy import io
import pylab as plt
import sys
from scipy.optimize import fmin_cg
#import pyximport; pyximport.install()
#from cy_sigmoid import cy_sigmoid
from sklearn.base import BaseEstimator
#Aaron's fortran-optimized openmp code
from nnopt import sigmoid2d, matmult
import cPickle

#number of iterations in training
_niter = 0

def main():
    """ not really used """
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
              delta=0, gamma=0.):
    """
    configure our neural network.
    The number of hidden layers is set by len(ninternal)

    arguments:
    ninputs: number of input attributes 
            **can be a numpy.array([nsamples, nproperties])
    nout: number of classifications
            **can be a numpy.array([nsamples]),
             then number of classifications = len(np.unique(nout))
    ninternal: [number of attributes for the each internal layer (without bias)]

    optionally initialize the Theta's in the l'th layer to your predefined array
      * needs len(thetas) == len(ninternal) + 1 
      * intial theta's should have bias term
      if thetas = [], randomly initialize the theta's uniformly over range
                      [-delta,delta], but then you should train your data
    gamma : learning rate (regularization parameter)
            defaults to 0.
                  
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
            % (idx, lin, lout)
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


class NeuralNetwork(BaseEstimator):
    """
    a collection of layers forming the neural network

    Notes:
    assumes labels are from [0, nclasses)
    gamma = learning rate (regularization parameter)

    """
    def __init__(self, layers, gamma=0.):
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
        gamma : regularization parameter, 
               default = None = self.gamma

        """
        global _niter
        if isinstance(X, type([])):
            X = np.array(X)
        
        if gamma == None:
            gamma = self.gamma

# testing: check the theta's are changing while we train: yes!
#        print "CF",thetas[0:2], thetas[-5:-2]

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

        if _niter == 0:
            print ("\n")
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
        if z.ndim == 2:
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
        thetas = self.flatten_thetas()
        return self.gradient(thetas, X, y, gamma)
        
    def gradient(self, thetas, X, y, gamma=None):
        """
        compute the gradient at each layer of the neural network
        
        Args:
        X = [nsamples, ninputs]
        y = [nsamples] #the training classifications
        gamma : regularization parameter
               default = None = self.gamma

        returns the gradient for the parameters of the neural network
        (the theta's) unrolled into one large vector, ordered from
        the first layer to latest.

        """
        if isinstance(X, type([])):
            X = np.array(X)

        if gamma == None:
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
#use fortran matmult if arrays are large
                    if deltan.size * theta.size > 200000:
                       tmp = matmult(deltan,theta.transpose())
                    else:
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
#                    

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

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        z : float

        """
        return np.mean(self.predict(X) == y)



    def score_weiwei(self, X, y, verbose=True):
        """
        Returns the mean accuracy on the given test data and labels
    
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Training set.
        
        y : array-like, shape = [n_samples]
        Labels for X.
        
        Returns
        -------
        z : float

        """
        pred_cls = {}
        true_cls = {}
        for cls in range(self.nclasses):
            pred_cls[cls] = set([])
            true_cls[cls] = set([])
            
        for i, s in enumerate(X):
            predict = self.predict(s)[0]
            true_cls[y[i]].add(i)
            pred_cls[y[i]].add(i)

        tot_acc = 0.
        for k in range(self.nclasses):
            hit = pred_cls[k] & true_cls[k]
            miss = pred_cls[k] - true_cls[k]
            falsepos = true_cls[k] - pred_cls[k] 
            precision = np.divide(float(len(hit)), len(pred_cls[k]))
            recall = np.divide(float(len(hit)), len(true_cls[k]))
            accuracy = (np.divide(float(len(hit)), len(true_cls[k])) * 100)
            tot_acc += accuracy
            if verbose:
                print "\nClass %s:" % k
                print 'accuracy: ', '%.0f%%' % (np.divide(float(len(hit)),len(true_cls[k])) * 100)
                print 'miss: ', '%.0f%%' % (np.divide(float(len(miss)),len(true_cls[k])) * 100)
                print 'false positives: ', '%.0f%%' % (np.divide(float(len(falsepos)),len(pred_cls[k]))* 100)
                print 'precision: ', '%.0f%%' % (precision* 100)
                print 'recall: ', '%.0f%%' % (recall* 100)

        z = tot_acc / self.nclasses
        return z

    def fit(self, X, y, gamma=None, maxiter=200, epsilon=1.e-7,
            gtol=1.e-5, raninit=True):
        """
        Train the data.
        minimize the cost function (wrt the Theta's)
        (using the conjugate gradient algorithm from scipy)
        This updates the NN.layers.theta's, so one can
        later "predict" other samples.

        Args:
        X : the training samples [nsamples x nproperties]
        y : the sample labels [nsamples], each entry in range 0<=y<nclass
        gamma : regularization parameter
               default = None = self.gamma
        *for scipy.optimize.fmin_cg:
        maxiter
        epsilon
        gtol

        raninit : T/F randomly initialize the theta's [default = True]
   
        """
        global _niter
        _niter = 0

        if gamma == None:
            gamma = self.gamma

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
                       disp=0,
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

    def learning_curve(self, X, y,
                       Xval=None, 
                       yval=None,
                       gamma=None,
                       pct=0.6,
                       plot=False):
        """
        returns the training and cross validation set errors
        for a learning curve
        (good for exploring effect of number of training samples)
        
        Args:
        X : training data
        y : training value
        Xval : test data
        yval : test value
        gamma : default None uses the objects value,
               otherwise gamma=0.
        pct (0<pct<1) : split the data as "pct" training, 1-pct testing
                       only if Xval = None
                       default pct = 0.6
        plot : False/[True] optionally plot the learning curve
        
        Note: if Xval == None, then we assume (X,y) is the entire set of data,
              and we split them up using split_data(data,target)

        returns two vectors of length(X):
        error_train : training error for the N=length(X) 
        error_val : error on x-val data, when trainined on "i" samples

        error = cost_Function(lambda=0)

        notes:
        * a high error indicates lots of bias,
          that you are probably underfitting the problem
          (so add more neurons/layers)
         
        * for lots of trials, a high gap between training_error
          and test_error (x-val error) indicates lots of variance
          (you are over-fitting, so remove some neurons/layers,
           or increase the regularization parameter)

        """
        if not Xval:
            X, y, Xval, yval = split_data(X, y, pct=pct)

        if gamma == None:
            gamma = self.gamma

        m = X.shape[0]
        ntrials = range(2,m,5)
        mm = len(ntrials)
        t_error = np.zeros(mm)
        v_error = np.zeros(mm)
        for i, v in enumerate(ntrials):
            #fit with regularization
#need at least two training items...
            if i < 2: continue
            self.fit(X[0:v+1], y[0:v+1], gamma=gamma, maxiter=100, raninit=True)
            
            # but compute error without!
            t_error[i] = self.costFunctionU(X[0:v+1], y[0:v+1], gamma=0.)
            # use entire x-val set
            v_error[i] = self.costFunctionU(Xval, yval, gamma=0.)
            
        if plot:
            plt.plot(ntrials, t_error, 'r+', label='training')
            plt.plot(ntrials, v_error, 'bx', label='x-val')
            plt.xlabel('training set size')
            plt.ylabel('error [J(gamma=0)]')
            plt.legend()
            plt.show()

        return t_error, v_error

    def validation_curve(self, X, y,
                         Xval=None, 
                         yval=None,
                         gammas=None,
                         pct=0.6,
                         plot=False):
        """
        use a cross-validation set to evaluate various regularization 
        parameters (gamma)
        
        specifically:
        train the NN, then loop over a range of regularization parameters
        and select best 'gamma' (=min(costFunction(cross-val data))

        Args:
        X : training data
        y : training value
        Xval : test data
        yval : test value
        pct (0<pct<1) : if Xval=None, split into 'pct' training
                       "1-pct" testing
        gammas : a *list* of regularization values to sample
                default None uses
                [0., 0.0001, 0.0005, 0.001, 0.05, 0.1, .5, 1, 1.5, 15]
        plot : False/[True] optionally plot the validation cure
        
        Note: if Xval == None, then we assume (X,y) is the entire set of data,
              and we split them up using split_data(data,target)

        returns:
        train_error(gamma), cross_val_error(gamma), gamma, best_gamma

        """
        if not Xval:
            X, y, Xval, yval = split_data(X, y, pct)
        
        if not gammas:
            gammas = [0., 0.0001, 0.0005, 0.001, 0.05, 0.1, .5, 1., 1.5, 15.]
        
        train_error = np.zeros(len(gammas))
        xval_error = np.zeros(len(gammas))
        for gi, gv in enumerate(gammas):
            self.fit(X, y, gamma=gv, maxiter=40, raninit=True)
            
            train_error[gi] = self.costFunctionU(X, y, gamma=gv)
            xval_error[gi] = self.costFunctionU(Xval, yval, gamma=gv)

        if plot:
            plt.plot(gammas, train_error, label='Train')
            plt.plot(gammas, xval_error, label='Cross Validation')
            plt.xlabel('gamma')
            plt.ylabel('Error [costFunction]')
            plt.legend()
            plt.show()

        return train_error, xval_error, gammas, gammas[xval_error.argmin()]

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
                    perturb[pi] = delta
                    perturb = perturb.reshape(theta_orig.shape)
                    lv.theta = theta_orig + perturb
                    loss1 = self.costFunctionU(X, y)
                    lv.theta = theta_orig - perturb
                    loss2 = self.costFunctionU(X, y)
                    numgrad[idx] = (loss2 - loss1) / (2*delta)
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
    nn = create_NN(nin, nout, ninternal=ninternal, thetas=np.array([]), delta=0)

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
        
def split_data(data, target, pct=0.6):
    """
    Given some complete set of data and their targets,
    split the indices into 'pct' training, '1-pct' cross-vals
    
    Args:
    data = input data
    target = data classifications
    pct = 0 < pct < 1, default 0.6

    returns:
    training_data, training_target, test_data, test_target

    """
    from random import shuffle
    
    L = len(target)
    index = range(L)
    cut = int(pct*L)
    while 1:
        shuffle(index)
        training_idx = index[:cut]
        training_target = target[training_idx]
        training_data = data[training_idx]

        test_idx = index[cut:]
        test_target = target[test_idx]
        test_data = data[test_idx]
        
# make sure training has samples from all classes
        if len(np.unique(training_target)) == len(np.unique(target)):
            break

    return training_data, training_target, test_data, test_target



if __name__ == '__main__':
    main()
