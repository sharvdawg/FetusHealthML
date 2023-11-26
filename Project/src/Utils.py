
import numpy as np

from numpy import asarray as arr
from numpy import atleast_2d as twod

def crossValidate(X, Y=None, K=5, i=0):
    """
    Create a K-fold cross-validation split of a data set:
    crossValidate(X,Y, 5, i) : return the ith of 5 80/20 train/test splits

    Parameters
    ----------
    X : MxN numpy array of data points to be resampled.
    Y : Mx1 numpy array of labels associated with each datum (optional)
    K : number of folds of cross-validation
    i : current fold to return (0...K-1)

    Returns
    -------
    Xtr,Xva,Ytr,Yva : (tuple of) numpy arrays for the split data set
    If Y is not present or None, returns only Xtr,Xva
    """
    nx,dx = twod(X).shape
    start = int(round( float(nx)*i/K ))
    end   = int(round( float(nx)*(i+1)/K ))

    Xte   = X[start:end,:] 
    Xtr   = np.vstack( (X[0:start,:],X[end:,:]) )
    to_return = (Xtr,Xte)

    Y = arr(Y).flatten()
    ny = len(Y)

    if ny > 0:
        assert ny == nx, 'crossValidate: X and Y must have the same length'
        if Y.ndim <= 1:
            Yte = Y[start:end]
            Ytr = np.hstack( (Y[0:start],Y[end:]) )
        else:   # in case targets are multivariate
            Yte = Y[start:end,:]
            Ytr = np.vstack( (Y[0:start,:],Y[end:,:]) )
        to_return += (Ytr,Yte)

    return to_return