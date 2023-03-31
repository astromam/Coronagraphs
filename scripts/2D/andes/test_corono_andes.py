#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@author: mndiaye
"""

#%%
"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

import os

#%%
"""
### Parameters
"""
# Pupil size
nPup = 400

# OPD map number
nOPD = 2000

# wavelength in m
lam = 1600e-9

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
       
    XU = 2.*np.pi* X.T.dot(U)
    A3 = sign*1j*np.sin(XU)  +np.cos(XU)
    A1 = A3.T
    
    B  = A1.dot(A2.dot(A3))

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


#%%
"""
### Working directory
"""
# File directory
fdir = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/data/').resolve()

# Directory for the pupils
fdir_pupil = fdir / 'Pupil'

# Directory for the OPD
fdir_opd   = fdir / 'OPD' / '4005'

# Filename and path for the ELT pupil
fname_elt = 'Tel-Pupil.fits'
fpath_elt = fdir_pupil / fname_elt

# Filename and path for the OPD maps
flist_opd = os.listdir(fdir_opd) 
fpath_opd = [fdir_opd / flist_opd[i] for i in range(nOPD)]

#%%
"""
### Read file
"""
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)

# Read OPD maps for the nOPD files
OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])

#%%
"""
### Compute perfect PSF
"""


#%%
# Field in the entrance pupil plane A
Fld_A = Pupil * np.exp(1j*2*np.pi*OPD_arr[0]/lam) 

# Field in the image plane D (no coronagraph)
Fld_D = sft()

#%%
"""
### Display plot
"""
plt.figure(0)
plt.clf()
plt.subplot(121)
plt.imshow(Pupil)
plt.title('ELT pupil')
plt.subplot(122)
plt.imshow(OPD_arr[0])
plt.title('OPD map')


