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

from mpl_toolkits.axes_grid1 import AxesGrid

import time

#%%
"""
### Parameters
"""
# Pupil size
nPup = 400

# Sampling of the coronagraph focal plane mask
nFPM = 100

# Image size
nImg = 400

# OPD map number
nOPD = 2000

# wavelength in m
lam = 1600e-9

# Pupil diameter in m 
D = 38.54

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# conversion lam/D to mas
lamD2mas = (lam/D)*mas2rad

# plate scale in mas per pixel
pscale = 0.3

# FPM size in lam/D in the focal plane B
mB = 4

# FoV in lam/D in the final image plane D
mD = 58.393/4


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
def profile(img, ptype='mean', step=1, mask=None, center=None, rmax=0, clip=True, exact=False):
    '''
    Azimuthal statistics of an image

    Parameters
    ----------
    img : array
        Image on which the profiles
        
    ptype : str, optional
        Type of profile. Allowed values are mean, std, var, median, min, max. Default is mean.
    
    mask : array, optional
        Mask for invalid values (must have the same size as image)
        
    center : array_like, optional
        Center of the image

    rmax : float
        Maximum radius for calculating the profile, in pixel. Default is 0 (no limit)
    
    clip : bool, optional
        Clip profile to area of image where there is a full set of data
        
    exact : bool, optional
        Performs an exact estimation of the profile. This can be very long for 
        large arrays. Default is False, which rounds the radial distance to the 
        closest 1 pixel.
    
    Returns
    -------
    prof : array
        1D profile vector
        
    rad : array
        Separation vector, in pixel
    '''
    
    # make sure we work on a copy
    img = img.copy()
    
    # array dimensions
    dimx = img.shape[1]
    dimy = img.shape[0]

    # center
    if center is None:
        center = (dimx // 2, dimy // 2)

    # masking
    if mask is not None:
        # check size
        if mask.shape != img.shape:
            raise ValueError('Image and mask don''t have the same size. Returning.')

        img[mask == 0] = np.nan
        
    # intermediate cartesian arrays
    x = np.arange(dimx, dtype=np.int64) - center[0]
    y = np.arange(dimy, dtype=np.int64) - center[1]
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    
    # rounds for faster calculation
    if not exact:
        rr = np.round(rr, decimals=0)
    
    # find unique radial values
    uniq = np.unique(rr, return_inverse=True, return_counts=True)
    r_uniq_val = uniq[0]
    r_uniq_inv = uniq[1]
    r_uniq_cnt = uniq[2]

    # number of elements
    if clip:
        extr  = np.abs(np.array((x[0], x[-1], y[0], y[-1])))
        r_max = extr.min()
        i_max = int(r_uniq_val[r_uniq_val <= r_max].size)
    else:
        r_max = r_uniq_val.max()
        i_max = r_uniq_val.size

    # limit extension of profile
    if (rmax > 0):
        r_max = rmax
        i_max = int(r_uniq_val[r_uniq_val <= r_max].size)
        
    t_max = r_uniq_cnt[0:i_max].max()

    # intermediate polar array
    polar = np.empty((i_max, t_max), dtype=img.dtype)
    polar.fill(np.nan)
    
    img_flat = img.ravel()
    for r in range(i_max):
        cnt = r_uniq_cnt[r]
        val = img_flat[r_uniq_inv == r]
        polar[r, 0:cnt] = val
            
    # calculate profile
    rad  = r_uniq_val[0:i_max]

    ptype = ptype.lower()
    if step == 1:
        # fast statistics if step=1
        if ptype == 'mean':
            prof = np.nanmean(polar, axis=1)
        elif ptype == 'std':
            prof = np.nanstd(polar, axis=1, ddof=1)
        elif ptype == 'var':
            prof = np.nanvar(polar, axis=1)
        elif ptype == 'median':
            prof = np.nanmedian(polar, axis=1)
        elif ptype == 'min':
            prof = np.nanmin(polar, axis=1)
        elif ptype == 'max':
            prof = np.nanmax(polar, axis=1)
        else:
            raise ValueError('Unknown statistics ptype = {0}. Allowed values are mean, std, var, median, min and max'.format(ptype))
    else:
        # slower if we need step > 1
        prof = np.zeros(i_max, dtype=img.dtype)
        for r in range(i_max):
            idx = ((rad[r]-step/2) <= rad) & (rad <= (rad[r]+step/2))
            val = polar[idx, :]
            
            if ptype == 'mean':
                prof[r] = np.nanmean(val)
            elif ptype == 'std':
                prof[r] = np.nanstd(val)
            elif ptype == 'var':
                prof[r] = np.nanvar(val)
            elif ptype == 'median':
                prof[r] = np.nanmedian(val)
            elif ptype == 'min':
                prof[r] = np.nanmin(val)
            elif ptype == 'max':
                prof[r] = np.nanmax(val)
            else:
                raise ValueError('Unknown statistics ptype = {0}. Allowed values are mean, std, var, median, min and max'.format(ptype))

    return prof, rad



#%%
"""
### Working directory
"""
# File directory
fdir = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/data/').resolve()

# Directory for the pupils
fdir_pupil = fdir / 'Pupil'

# Directory for the OPD
fdir_opd   = fdir / 'OPD' / '12345'

# Directory for the PSF generated by Anne-Laure Cheffaut
fdir_psf = fdir / 'PSF' 

# Filename and path for the ELT pupil
fname_elt = 'Tel-Pupil.fits'
fpath_elt = fdir_pupil / fname_elt

# Filename and path for the OPD maps
flist_opd = os.listdir(fdir_opd) 
fpath_opd = [fdir_opd / flist_opd[i] for i in range(nOPD)]

# Filename for the PSF generated by Anne-Laure Cheffaut 
fname_psf = '20230320-171503_test_1600nm.fits'
fpath_psf = fdir_psf / fname_psf

#%%
"""
### Read file
"""
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)

# Read OPD maps for the nOPD files
OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])

# Read the PSF generated by Anne-Laure Cheffaut
PSF_alc = fits.getdata(fpath_psf,)

#%% 
"""
### Cropping of the PSF generated by Anne-Laure Cheffaut
"""
# Size of the original image
nImg_alc = np.size(PSF_alc, 0)

# dimensions to crop the images to nImg
nIni = (nImg_alc-nImg)//2
nEnd = (nImg_alc+nImg)//2

# crop the images to nImg
PSF_alc1 = PSF_alc[nIni:nEnd,nIni:nEnd]

# flip image upd-down and left-right
PSF_alc1 = np.flipud(np.fliplr(PSF_alc1))

# normalize image
PSF_alc1 /= np.max(PSF_alc1) 

#%%
"""
### Coronagraphic components
"""
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)

# Lyot stop
LyotStop2d = Pupil*1

#%%
"""
### Compute perfect PSF
"""
# Field in the entrance pupil plane A
Fld_AA0 = Pupil * 1.

# Field in the image plane D (no coronagraph)
Fld_DD0 = sft(Fld_AA0, nImg, mD)

# Intensity 
Int_DD0 = np.abs(Fld_DD0)**2

# Normalized intensity
norm_peakDD0 = 1/np.max(Int_DD0)
Int_DD0 *= norm_peakDD0

#%%
"""
### Compute perfect coronographic image
"""
# focal plane B 
Fld_BB = mask2d*sft(Fld_AA0, nFPM, mB)

# pupil plane C before Lyot stop
Fld_CC = Fld_AA0 - isft(Fld_BB, nPup, mB)

# pupil plane C after Lyot stop
Fld_LL = Fld_CC*LyotStop2d

# image plane D 
Fld_DD = sft(Fld_LL, nImg, mD)

# Intensity
Int_DD = np.abs(Fld_DD)**2

# Normalized intensity
Int_DD *= norm_peakDD0

#%%
"""
### Compute PSF (with errors)
"""
t0 = time.time()
Int_D0 = np.zeros((nImg, nImg))
for iOPD in range(nOPD):
    # Field in the entrance pupil plane A
    Fld_A0 = Pupil * np.exp(1j*2*np.pi*OPD_arr[iOPD]/lam) 
    
    # Field in the image plane D (no coronagraph)
    Fld_D0 = sft(Fld_A0, nImg, mD)
    
    # Intensity 
    Int_D0 += np.abs(Fld_D0)**2
t1 = time.time()       
print(f'computation time: {t1-t0:.3f}s')  
  
# Normalized intensity
Int_D0 /= nOPD

# Normalized intensity
norm_peakD0 = 1/np.max(Int_D0)
Int_D0 *= norm_peakD0

#%%
"""
### Compute coronographic image (with errors)
"""
t0 = time.time()
Int_D = np.zeros((nImg, nImg))
for iOPD in range(nOPD):
    # pupil plane A
    Fld_A0 = Pupil * np.exp(1j*2*np.pi*OPD_arr[0]/lam) 
    
    # focal plane B 
    Fld_B = mask2d*sft(Fld_A0, nFPM, mB)
    
    # pupil plane C before Lyot stop
    Fld_C = Fld_A0 - isft(Fld_B, nPup, mB)
    
    # pupil plane C after Lyot stop
    Fld_L = Fld_C*LyotStop2d
    
    # image plane D 
    Fld_D = sft(Fld_L, nImg, mD)
    
    # Intensity
    Int_D += np.abs(Fld_D)**2    
t1 = time.time()       
print(f'computation time: {t1-t0:.3f}s')  
  
# Normalized intensity
Int_D /= nOPD

# Normalized intensity
Int_D *= norm_peakD0

#%%
"""
### Compute the radial intensity profiles of the images
"""
# computation of the averaged intensity profiles of the images   
Int_D0_prf_avg, rad_D0_prf_avg = profile(Int_D0, ptype='mean')
Int_D_prf_avg, rad_D_prf_avg = profile(Int_D, ptype='mean')
PSF_alc1_prf_avg, rad_D_prf_avg = profile(PSF_alc1, ptype='mean')

# computation of the standard deviation intensity profiles of the images
Int_D0_prf_std, rad_D0_prf_std = profile(Int_D0, ptype='std')
Int_D_prf_std, rad_D_prf_std = profile(Int_D, ptype='std')
PSF_alc1_prf_std, rad_D0_prf_std = profile(PSF_alc1, ptype='std')

#%%
"""
### Display pupil plot
"""
plt.figure(0)
plt.clf()
plt.subplot(121)
plt.imshow(Pupil)
plt.title('ELT pupil')
plt.subplot(122)
plt.imshow(OPD_arr[0])
plt.title('OPD map')

#%%
"""
### Display images
"""
# boundaries for the images in log scale
vmin0 = -7
vmax0 = 0

fig = plt.figure(1, figsize=(12,6))
plt.clf()

grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 3),
                axes_pad=0.2,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.2
                )

# Perfect PSF
im = grid[0].imshow(np.log10(Int_DD0), vmin=vmin0, vmax=vmax0, cmap='inferno')
grid[0].set_title('perfect PSF')

# AO corrected PSF (MND)
im = grid[1].imshow(np.log10(Int_D0), vmin=vmin0, vmax=vmax0, cmap='inferno')
grid[1].set_title(f'AO corrected PSF (MND)')

# AO corrected PSF (ALC)
im = grid[2].imshow(np.log10(PSF_alc1), vmin=vmin0, vmax=vmax0, cmap='inferno')
grid[2].set_title(f'AO corrected PSF (ALC)')

# Perfect coronagraphic image
im = grid[3].imshow(np.log10(Int_DD), vmin=vmin0, vmax=vmax0, cmap='inferno')
grid[3].set_title('Perfect coro. image')

# AO corrected coronagraphic image
im = grid[4].imshow(np.log10(Int_D), vmin=vmin0, vmax=vmax0, cmap='inferno')
grid[4].set_title('AO corrected coro. image')

# colorbar
cbar = grid[0].cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Intensity in log scale', rotation=270)

plt.show()

#%%
"""
### plot the radial profiles of the image intensity
"""

# convert pixel scale into lam/D scale for the x-axis
rad_D0_prf_avg_lamD = rad_D0_prf_avg * mD/nImg
rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg

# plot of the radial profiles
plt.figure(2, (8, 4.5))
plt.clf()

# AO corrected PSF (MND)
plt.plot(rad_D0_prf_avg_lamD, Int_D0_prf_avg, label='PSF (MND)')

# AO corrected PSF (ALC)
plt.plot(rad_D0_prf_avg_lamD, PSF_alc1_prf_avg, label='PSF (ALC)', ls='--')

# AO corrected coronagraphic image
plt.plot(rad_D_prf_avg_lamD, Int_D_prf_avg, label='corono')

# Focal plane mask boundary
x = np.arange(0.0, 2, 0.01)
plt.axvline(x=mB/2, color='k', ls='--')

# Focal plane mask grey area
plt.fill_between(x, 0, mB/2, color='gray', alpha=0.3)

plt.xlim(-0.05, np.max(rad_D_prf_avg_lamD)+0.05)
plt.ylim(2e-5, 2e0)
plt.xlabel(r'Angular separation [$\lambda$/D]')
plt.ylabel('Normalized intensity in log scale')
plt.yscale('log')
plt.title(f'Radial intensity profile at $\lambda$={lam*1e6:.3f}$\mu$m')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.show()


