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
import pdb
import glob

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

# OPD map number in the files
nOPD = 2000

# wavelength in m

wl = 'H'

if wl == 'H' : 

    lam_0 = 1600e-9
    #liste de 11 wavelengths pour H band
    lam_list = [1.4e-6, 1.44e-6, 1.48e-6, 1.52e-6, 1.56e-6, 1.6e-6, 1.64e-6, 1.68e-6, 1.72e-6, 1.76e-6, 1.80e-6]

elif wl == 'J' :

    lam = 1200e-9
elif wl == 'Y': 
    
    lam = 1000e-9

# Pupil diameter in m 
D = 38.54

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
user = 'Adrien'
if user == 'Adrien':
    # File directory
    fdir_dat = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/data').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/').resolve()

elif user == 'Mamadou':
    # File directory
    fdir_dat = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/data/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/plots/').resolve()

# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'
fdir_elt = '/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/data/elt/'

# Directory for the PSF generated by Anne-Laure Cheffaut
fdir_psf = fdir_dat / 'PSF' 

# Filename and path for the ELT pupil
fname_elt = 'Tel-Pupil.fits'
fpath_elt = fdir_pupil / fname_elt

# Filename for the PSF generated by Anne-Laure Cheffaut 
flist_psf = os.listdir(fdir_psf)
fpath_psf = [fdir_psf / flist_psf[i] for i in range(len(flist_psf))]

apodizer_files = np.sort(glob.glob(fdir_elt + 'pupilelt_nPup400_A**'))
#%%
"""
### Read file
"""
# Read ELT pupil 
# Pupil = fits.getdata(fpath_elt,)

Pupil = uniform_disk(nPup, nPup/2.)

    #%%
"""
### Coronagraphic components
"""
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)

# Lyot stop
LyotStop2d_base = Pupil*1


Pupil2 = Pupil*1


for i in range(1,len(Pupil)):
    for j in range(1,len(Pupil)):
        if Pupil[i,j] == 0:
            Pupil2[i-1,j-1] = 0
        
for i in range(len(Pupil)-2,0,-1):
    for j in range(len(Pupil)-2,0,-1):
        if Pupil[i,j] == 0:
            Pupil2[i+1,j+1] = 0

Pupil3 = Pupil2*1
for i in range(1,len(Pupil)):
    for j in range(1,len(Pupil)):
        if Pupil2[i,j] == 0:
            Pupil3[i-1,j-1] = 0
        
for i in range(len(Pupil)-2,0,-1):
    for j in range(len(Pupil)-2,0,-1):
        if Pupil2[i,j] == 0:
            Pupil3[i+1,j+1] = 0

#%%
for lam in lam_list :

    # conversion lradian to mas
    rad2mas = np.pi/(180.*3600*1000)
    mas2rad = 1/rad2mas

    # conversion lam/D to mas
    lamD2mas = (lam_0/D)*mas2rad

    # plate scale in mas per pixel
    pscale = 0.3

    # FPM size in lam/D in the focal plane B
    # mB = 4 

    # FoV in lam/D in the final image plane D
    mD = 58.393*(nImg/1600) * (lam_0/lam)
    # print(mD)

    """
    Pupil configuration
    """

    diam = 0.9
    obst = 0.4
    mB = 3.8 * lam_0/lam # mutiplier par lambda_0/lamdba # mB en fonction de lam or not ? 

    LyotStop2d = Pupil*(uniform_disk(nPup, diam*nPup/2)-uniform_disk(nPup, obst*nPup/2))

    #%%
    coro_config = 'lyot'

    
    Pupil_coro = Pupil  
 
    """
    ### Creating directories to save results for each configurations
    """
    
    fdir_plt = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/').resolve()
    fdir_res   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/').resolve()
    
    
    fdir_plt = fdir_plt / 'lyot_diam_{diam}_obst_{obst}_mB={mB}_bande_{band}_wl_{lam}'.format(diam=diam,obst = obst,mB=mB,band=wl,lam=lam)
    fdir_res = fdir_res / 'lyot_diam_{diam}_obst_{obst}_mB={mB}_bande_{band}_wl_{lam}'.format(diam=diam,obst = obst,mB=mB,band = wl,lam=lam)

                
    if not os.path.exists(fdir_plt):
        os.makedirs(fdir_plt)
    if not os.path.exists(fdir_res):
        os.makedirs(fdir_res)

    """
    ### Compute perfect PSF
    """
    # Field in the entrance pupil plane A
    Fld_AA0 = Pupil_coro

    # Field in the entrance pupil plane C
    Fld_CC0 = Fld_AA0 * 1.*LyotStop2d

    # Field in the image plane D (no coronagraph)*
    Fld_DD0 = sft(Fld_CC0, nImg, mD*diam) #diametre_lyot

    # Intensity 
    Int_DD0 = np.abs(Fld_DD0)**2

    # Normalized intensity
    norm_peakDD0 = 1/np.max(Int_DD0)
        
    Int_DD0 *= norm_peakDD0 #norm_peakDD0

    """
    ### Calcul of corono image without atmospheric turbulences
    """
        
    Fld_AA = Pupil_coro
    # focal plane B 
    Fld_BB = mask2d*sft(Fld_AA, nFPM, mB)

    # pupil plane C before Lyot stop
    Fld_CC = Fld_AA - isft(Fld_BB, nPup, mB)
    #definir font size
    plt.rcParams.update({'font.size': 20})            
   
    # pupil plane C after Lyot stop
    Fld_LL = Fld_CC*LyotStop2d
   
    # image plane D 
    Fld_DD = sft(Fld_LL, nImg, mD*diam) #diametre_lyot

    # Intensity
    Int_DD = np.abs(Fld_DD)**2

    # Normalized intensity
    Int_DD *= norm_peakDD0 #norm_peakDD0

    fname_Int_DD = 'wonoise_cor_'+coro_config+'.fits'
    fname_Int_DD0 = 'wonoise_'+coro_config+'.fits'
    fname_Int_DD0_no_lyot = 'wonoise_'+coro_config+'_no_lyot.fits'
    fname_fract_lum = "fraction_lum.npy"
    

    # filepath for the direct and coronagraphic images
        
    fpath_Int_DD  = fdir_res / fname_Int_DD
    fpath_Int_DD0  = fdir_res / fname_Int_DD0
    # fpath_Int_DD0_no_lyot  = fdir_res / fname_Int_DD0_no_lyot
    # save the direct and coronagraphic images
    fits.writeto(fpath_Int_DD0, Int_DD0, overwrite=True)
    fits.writeto(fpath_Int_DD, Int_DD, overwrite=True)
    # fits.writeto(fpath_Int_DD0_no_lyot, Int_DD0_no_lyot, overwrite=True)
                