# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:35:14 2024

@author: mndiaye
"""

import cupy as cp

#%%
"""
### Functions
"""

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
        of Lyot-style coronagraph propagation, Optics Express, vol. 15,
        issue 24, p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    
    A2 = cp.array(A2, dtype=cp.complex128)
    
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    NA    = cp.shape(A2)[0]
    coeff = m/(NA*NB)
    
    sign = -1.0
    if inv:
        sign = 1.0

    U = cp.zeros((1,NB))
    X = cp.zeros((1,NA))
    
    X[0,:] = (1./NA)*(cp.arange(NA)-NA/2.+val)
    U[0,:] =  (m/NB)*(cp.arange(NB)-NB/2.+val)
        
    XU = cp.float64( 2. * cp.pi ) * X.T.dot(U)
    A3 = sign * 1j * cp.sin(XU) + cp.cos(XU)
    A1 = A3.T

    B  = A1.dot(A2.dot(A3))

    return cp.asnumpy(coeff*B)


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

