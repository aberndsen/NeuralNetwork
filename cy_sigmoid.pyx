"""
implement sigmoid in cython

"""
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double exp(double)

cdef double f(double z):
    return exp(z)

def cy_sigmoid(np.ndarray[double, ndim=2] z):
    cdef unsigned int NX, NY, i, j
    cdef np.ndarray[double, ndim=2] sig

    NY, NX = np.shape(z)	    
    
    sig = np.zeros((NY,NX))
    for i in xrange(NX):
        for j in xrange(NY):
            sig[j,i] = 1./(1. + exp(-z[j,i]))
    return sig
