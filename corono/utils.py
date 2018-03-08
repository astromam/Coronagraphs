#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:10:51 2018

@author: mndiaye
"""

import numpy as np
import scipy.special

#%%
# check python version
try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3
        

#%%
def to_dict(**kwargs):
    return kwargs

#%%
def uniform_disk(n, radius, ctr=False):
    ''' ---------------------------------------------------------
    returns an (ys x xs) array with a uniform disk of radius "radius".
    ---------------------------------------------------------  '''
    val    = 0
    if ctr is True:
        val = 1/2 
    xx,yy  = np.meshgrid(np.arange(n)-n/2+val, np.arange(n)-n/2+val)
    mydist = np.hypot(yy,xx)
    res    = np.zeros_like(mydist)
    res[mydist <= radius] = 1.0
    return res

#%%
def sft(A2, NB, m, inv=False, ctr=False):
    ''' --------------------------------------------------------------
    Explicit Fourier Transform, using the theory described in:
    http://adsabs.harvard.edu/abs/2007OExpr..1515935S

    Assumes the original array is square.
    No need to "center" the data on the origin.

    Parameters:
    ----------

    - A2 : the 2D original array
    - NB : the linear size of the result array (integer)
    - m  : m/2 = maximum spatial frequency to be computed (in l/D)
    - inv: boolean (direct or inverse) see the definition of isft()
    -------------------------------------------------------------- '''
    val    = 0
    if ctr is True:
        val = 1/2
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    
    sign = -1.0
    if inv:
        sign = 1.0

    U = np.zeros((1,NB))
    X = np.zeros((1,NA))
    
    X[0,:] = (1./NA)*(np.arange(NA)-NA/2.+val)
    U[0,:] =  (m/NB)*(np.arange(NB)-NB/2.+val)
       
#     A1 = np.exp(sign * 2.j*np.pi* U.T.dot(X))
#     A3 = np.exp(sign * 2.j*np.pi* X.T.dot(U))
    A1 = sign*1j*np.sin(2.*np.pi* U.T.dot(X))
    A1 += np.cos(2.*np.pi* U.T.dot(X))
    
    A3 = sign*1j*np.sin(2.*np.pi* X.T.dot(U))
    A3 += np.cos(2.*np.pi* X.T.dot(U))
    B  = (A1.dot(A2)).dot(A3)

    return coeff*B

#%%
def isft(A2, NB, m, ctr=False):
    ''' --------------------------------------------------------------
    Explicit inverse Fourier Transform, using the theory described in:
    http://adsabs.harvard.edu/abs/2007OExpr..1515935S

    See documentation for sft().
    -------------------------------------------------------------- '''
    return sft(A2, NB, m, inv=True, ctr=ctr)

#%%
# import Bessel function
besselJ0raw=lambda z: scipy.special.jv(0,z)
def besselJ0(z):
    temp=besselJ0raw(z)
    temp[np.isnan(temp)]=0
    return temp

#%%
def write_apod1d(fpath, test):
    np.savetxt(fpath, test)
    
#%%
def load_apod1d(fpath):
    test  = np.loadtxt(fpath)
    rApod = test[:,0]
    Apod  = test[:,1]
    
    return rApod,Apod    