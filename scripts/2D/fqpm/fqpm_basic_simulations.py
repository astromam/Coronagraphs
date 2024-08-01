#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:45:21 2024

@author: mndiaye
"""

#%%
"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fftshift, fft2, ifft2 

#%%
"""
### Parameters
"""
nPup = 100
nArr = 2048


key_fqpm = True
vrtx_charge = 6

if key_fqpm:
    str_cor = 'FQPM'
else: 
    str_cor = 'VRTX'

#%%
"""
### Working directories
"""

#%%
"""
### Functions
"""
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
    # res[mydist <= radius] = 1.0
    res[mydist < radius] = 1.0
    return res

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
### Create circular aperture
"""
Pupil2d = uniform_disk(nArr, nPup/2, CtrBtwnPix=True)
LyotStop2d = uniform_disk(nArr, nPup/2, CtrBtwnPix=True)



#%%
"""
### Create FQPM
"""
FQPM2d = np.ones((nArr, nArr))
FQPM2d[:nArr//2, nArr//2:] = -1.
FQPM2d[nArr//2:, :nArr//2] = -1.


#%%
"""
### Create vortex mask
"""

x0,y0  = np.meshgrid(np.arange(nArr)-nArr/2+1/2, np.arange(nArr)-nArr/2+1/2)
theta0 = np.arctan2(y0, x0)
VRTX2d = np.zeros((nArr, nArr), dtype='complex128')
VRTX2d = np.exp(1j*vrtx_charge*theta0)



#%%
"""
### Pick a mask
"""

if key_fqpm:
    Mask2d = FQPM2d*1.
else:
    Mask2d = VRTX2d*1.


#%%
"""
### Field in the different planes of the coronagraph using fft
"""
# Psi_A = Pupil2d*1
# Psi_B = fftshift(fft2(fftshift(Psi_A), norm="ortho"))
# Psi_M = Psi_B*Mask2d
# Psi_C = fftshift(ifft2(fftshift(Psi_M), norm="ortho"))
# Psi_L = Psi_C * LyotStop2d
# Psi_D = fftshift(fft2(fftshift(Psi_L), norm="ortho"))

#%%
"""
### Field in the different planes of the coronagraph using sft
"""
# Entrance pupil plane
Psi_A = Pupil2d*1
# Focal plane before mask
Psi_B = sft(Psi_A, nArr, nArr, CtrBtwnPix=True)
# Focal plane after mask
Psi_M = Psi_B*Mask2d
# Re-imaged pupil plane before Lyot stop
Psi_C = isft(Psi_M, nArr, nArr, CtrBtwnPix=True)
# Re-imaged pupil plane after Lyot stop
Psi_L = Psi_C * LyotStop2d
# Final image plane
Psi_D = sft(Psi_L, nArr, nArr, CtrBtwnPix=True)

# Intensity in the intermediate focal plane B
Int_B = np.abs(Psi_B)**2
# Intensity in the final focal plane D
Int_D = np.abs(Psi_D)**2

# 
Int_pk = np.max(Int_B)

Int_B /= Int_pk 
Int_D /= Int_pk 

#%%
"""
### Display images
"""
vmin0 = -7
vmax0 = 0

fig = plt.figure(0, figsize=(10,4.5))
plt.clf()
ax0 = fig.add_subplot(121)
im = ax0.imshow(np.log10(Int_B), vmin=vmin0, vmax = vmax0, cmap='inferno')
ax0.set_title('w/o corono')

ax1 = fig.add_subplot(122)
im = ax1.imshow(np.log10(Int_D), vmin=vmin0, vmax = vmax0, cmap='inferno')
ax1.set_title(f'with {str_cor}')

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = fig.colorbar(im, cax=cbar_ax, cmap='inferno')
cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
#plt.tight_layout()
plt.show()

#%%
"""
### Display contrast radial cut curves
"""
Int_B_vec = Int_B[nArr//2, nArr//2:]
Int_D_vec = Int_D[nArr//2, nArr//2:]

plt.figure(1)
plt.clf()
plt.semilogy(Int_B_vec, label='w/o corono')
plt.semilogy(Int_D_vec, label=f'with {str_cor}')
plt.legend()