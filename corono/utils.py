#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:10:51 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

import numpy as np
import scipy.special
import copy

#%% check python version
"""
Check python version
"""
try:
    reload  # Python 2.7
    reload=reload
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3
        
#%%
def to_dict(**kwargs):
    """
    Returns the arguments in the dictonary
    """
    return kwargs

#%%
def uniform_disk(n, radius, CtrBtwnPix=False):
    """
    Generates a uniform disk in a 2D array.
    
    Parameters
    ----------
    n : integer
        size of the array
    
    radius : float
        radius of the disk
    
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ----------
    res : array_like
        (ys x xs) array with a uniform disk of radius "radius".
        
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2 
    xx,yy  = np.meshgrid(np.arange(n)-n/2+val, np.arange(n)-n/2+val)
    mydist = np.hypot(yy,xx)
    res    = np.zeros_like(mydist)
    res[mydist <= radius] = 1.0
    return res

#%%
def radius_disk(n, radius, CtrBtwnPix=False):
    """
    Generates a uniform disk in a 2D array.
    
    Parameters
    ----------
    n : integer
        size of the array
    
    radius : float
        radius of the disk
    
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ----------
    res : array_like
        (ys x xs) array with a uniform disk of radius "radius".
        
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2 
    xx,yy  = np.meshgrid(np.arange(n)-n/2+val, np.arange(n)-n/2+val)
    mydist = np.hypot(yy,xx)/n
    return mydist


#%%
def sft(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square. 

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
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
def isft(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].

    See Also
    --------
    sft() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

#%% import Bessel function
def besselJ0(z):
    """
    Computes the Bessel function of zero order for a given array
    
    Parameters
    ---------
    z : array_like
        variable for which the bessel function is computed
    
    Returns
    ---------
    temp : array_like
        Bessel function of zero order for the z array
    
    """
    temp=scipy.special.jv(0,z)
    temp[np.isnan(temp)]=0
    return temp

#%%
def write_apod1d(fpath, test):
    """
    Saves 1D apodizer
    
    Parameters
    ----------
    fpath : string
        Filepath in which to save fpath
    
    test : array_like 
        Array to be saved
    
    """
    np.savetxt(fpath, test)
    
#%%
def load_apod1d(fpath):
    """
    load 1D apodizer
    
    Parameters
    ----------
    fpath : string
        Path of the file to load
        
    Returns
    ---------
    rApod : array_like
        Radial coordinate of the apodizer points to be loaded
    
    Apod : array_like
        Apodizer to be loaded
    
    """
    test  = np.loadtxt(fpath)
    rApod = test[:,0]
    Apod  = test[:,1]
    
    return rApod,Apod    

#%%
def sft_even(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square and represents an even function. 

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
                
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    
    U = np.zeros((1,NB))
    X = np.zeros((1,NA))
    
    X[0,:] = (1./NA)*(np.arange(NA)-NA/2+val)
    U[0,:] =  (m/NB)*(np.arange(NB)-NB/2+val)

    A1 = np.cos(2.*np.pi* U.T.dot(X))    
    A3 = np.cos(2.*np.pi* X.T.dot(U))
    
    B  = (A1.dot(A2)).dot(A3)

    return coeff*B

#%%
def isft_even(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].
    Assumes the original array is square and represents an even function. 

    See Also
    --------
    sft_even() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft_even(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

#%%
def update_params(params, **kwargs):
    """
    Update a dictionary of parameters
    
    Parameters
    --------
    params : dict
        the original dictionary
    
    kwargs : dict
        parameters given by the user for the keys with the values to update
    
    Returns
    --------
    params : dict
        a copy of the original dictionary with an update of the parameters
        given by the user
    
    """
    params2 = copy.deepcopy(params)
    if len(kwargs) != 0:
        for key, value in kwargs.items():
            params2[key] = value
    return params2

#%%
def sft_ein(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square. Uses the Einstein sum

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    
    sign = -1.0
    if inv:
        sign = 1.0

    U = np.empty((1,NB))
    X = np.empty((1,NA))
    
    X[0,:] = (1./NA)*(np.arange(NA)-NA/2.+val)
    U[0,:] =  (m/NB)*(np.arange(NB)-NB/2.+val)
    
    A1tmp = np.einsum('ji,jk -> ik', U, X)
    A1 = sign*1j*np.sin(2.*np.pi*A1tmp)
    A1+= np.cos(2.*np.pi*A1tmp)
        
    A3tmp = np.einsum('ji,jk -> ik', X, U)
    A3 = sign*1j*np.sin(2.*np.pi* A3tmp)
    A3 += np.cos(2.*np.pi* A3tmp)
    
#    B  = (A1.dot(A2)).dot(A3)

    Btmp = np.einsum('ij,jk -> ik', A1, A2)
    B    = np.einsum('ij,jk -> ik', Btmp, A3)
    
    print('einsum - sft')

    return coeff*B

#%%
def isft_ein(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].
    Uses the Einstein sum for computation.

    See Also
    --------
    sft() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft_ein(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

