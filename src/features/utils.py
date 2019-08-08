#-*- coding: utf-8 -*-

__author__ = 'Ronan Perry'

import numpy as np

def decimate_ptr(X, nbins=1000):
    '''
    Places the flattened data into bins, equally spaced from 
    the minimum to the maximum. Data is reassigned the index of the bin
    and reshaped.
    
    Parameters
    ----------
    X : numpy.array, shape (n, m)
        data to decimate
    bins: int
        number of bins to decimate the data into
    '''
    
    size = X.shape
    X = X.flatten()

    bins = np.linspace(min(X), max(X), nbins+1)
    bins[-1] += 1 #To include the max element in the last bin
    X = np.digitize(X, bins)
    
    return(X.reshape(size))