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

#definir font size
plt.rcParams.update({'font.size': 20})            


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

    lam = 1600e-9
    #liste de 11 wavelengths pour H band
    # lam_list = [1.4e-6, 1.44e-6, 1.48e-6, 1.52e-6, 1.56e-6, 1.6e-6, 1.64e-6, 1.68e-6, 1.72e-6, 1.76e-6, 1.80e-6]

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
    fdir_dat = Path('/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path('/Users/asimonnin/Desktop/PhD/Andes/Data_corono/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path('/Users/asimonnin/Desktop/PhD/Andes/Data_corono/plots/').resolve()

elif user == 'Mamadou':
    # File directory
    fdir_dat = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/data/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/plots/').resolve()

# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'
fdir_elt = '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/elt/'

# Filename and path for the ELT pupil
# fname_elt = 'Tel-Pupil.fits'
fname_elt = 'ELT_pupil_400.fits'
fpath_elt = fdir_pupil / fname_elt

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

"""
### Read OPDs
"""


fdir_opd = '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/PASSATA/20240228_112033.0_Seeing_0_65' #median

# fdir_opd = '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/PASSATA/20240228_053027.0_Seeing_0_57' #JQ2

fdir_opd = Path(fdir_opd).resolve()

# Filename and path for the OPD maps
flist_opd = os.listdir(fdir_opd) 
fpath_opd = [fdir_opd / flist_opd[i] for i in range(len(flist_opd)) if flist_opd[i].endswith('.fits')]
            
fpath_opd = sorted(fpath_opd)

fpath_opd = fpath_opd[200:2750]  #5min temps exposition
        
                
t0 = time.time()
# Read OPD maps for the nOPD files
OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])




#%%

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# conversion lam/D to mas
lamD2mas = (lam/D)*mas2rad

# plate scale in mas per pixel
pscale = 0.3

# FPM size in lam/D in the focal plane B
# mB = 4 

# FoV in lam/D in the final image plane D
mD = 58.393*(nImg/1600) #* (lam_0/lam)
# print(mD)


"""
Pupil configuration
"""

### FOR RESEARCH OF BEST PARAMETERS

diametre_lyot = [0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00]#0.9,0.92
obstruction = [0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,]
mB_conf = [3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0]


throughput = np.zeros((len(diametre_lyot),len(obstruction),len(mB_conf)))


diam_k=0

resultat_no_coro_no_turb_rad = []
resultat_no_coro_no_turb_int = []

resultat_w_coro_no_turb_rad = []
resultat_w_coro_no_turb_int = []

resultat_no_coro_w_turb_rad = []
resultat_no_coro_w_turb_int = []

resultat_w_woro_w_turb_rad = []
resultat_w_woro_w_turb_int = []

for diam in diametre_lyot: 
    obst_i=0
    for obst in obstruction:
        mb_i=0
        print(diam,obst)
        LyotStop2d = Pupil*(uniform_disk(nPup, diam*nPup/2)-uniform_disk(nPup, obst*nPup/2))

#%%
        coro_config = 'lyot'
                
        for s in range(len(mB_conf)) :
            """
            ### Selection of the good apodizer configuration according to FPM
            """
            mB = mB_conf[s]

            
            Pupil_coro = Pupil  

            """
            ### Compute perfect PSF
            """
            # Field in the entrance pupil plane A
            Fld_AA0 = Pupil_coro

            int_a = np.abs(Fld_AA0)**2

            ee_a = np.sum(int_a)

            # Field in the entrance pupil plane C
            Fld_CC0 = Fld_AA0 * 1.*LyotStop2d

            int_c = np.abs(Fld_CC0)**2

            ee_c = np.sum(int_c)

            throughput[diam_k,obst_i,mb_i] = ee_c/ee_a


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


            Int_DD_prf_avg, rad_DD_prf_avg = profile(Int_DD, ptype='mean')
            Int_DD0_prf_avg, rad_DD0_prf_avg = profile(Int_DD0, ptype='mean')

            Int_DD0_prf_std, rad_DD0_prf_std = profile(Int_DD0, ptype='std')
            Int_DD_prf_std, rad_DD_prf_std = profile(Int_DD, ptype='std')


            rad_DD0_prf_avg_lamD = rad_DD0_prf_avg * mD/nImg
            rad_DD_prf_avg_lamD = rad_DD_prf_avg * mD/nImg

            rad_DD0_prf_avg_mas = rad_DD0_prf_avg_lamD * lamD2mas
            rad_DD_prf_avg_mas = rad_DD_prf_avg_lamD * lamD2mas

            resultat_w_coro_no_turb_rad.append(rad_DD_prf_avg)
            resultat_w_coro_no_turb_int.append(Int_DD_prf_avg)

            resultat_no_coro_no_turb_rad.append(rad_DD0_prf_avg)
            resultat_no_coro_no_turb_int.append(Int_DD0_prf_avg)
                

            """
            ### load OPD of Anne Laure
            """


            for i in range (nOPD):
                t1 = time.time()
                print(f'OPD reading file time: {t1-t0:.3f}s') 
            
            
                """
                ### Compute PSF (with errors)
                """



                t0 = time.time()
                Int_D0 = np.zeros((nImg, nImg))
                for iOPD in range(nOPD):
                    # Field in the entrance pupil plane A
                    Fld_A0 = Pupil_coro * np.exp(2*1j*np.pi*OPD_arr[iOPD]/lam) 
                    # print(lam,lam_0)
                    # Field in the entrance pupil plane C
                    Fld_C0 = Fld_A0 * LyotStop2d
                    # Field in the image plane D (no coronagraph)
                    Fld_D0 = sft(Fld_C0, nImg, mD*diam)#diametre_lyot mD*diam

                    # Intensity 
                    Int_D0 += np.abs(Fld_D0)**2

                t1 = time.time()       
                # print(f'PSF computation time: {t1-t0:.3f}s')  

                # Normalized intensity
                Int_D0 /= nOPD

            
                # Normalized intensity
                norm_peakD0 = 1/np.max(Int_D0)

                # norm_peak_2 = 1/np.max(Int_DD0)

                Int_D0 *= norm_peakD0

                """
                ### Compute coronographic image (with errors)
                """
                t0 = time.time()
                Int_D = np.zeros((nImg, nImg))
                for iOPD in range(nOPD):
                    # pupil plane A
                    Fld_A0 = Pupil_coro * np.exp(2*1j*np.pi*OPD_arr[iOPD]/lam)
                
                    # focal plane B 
                    Fld_B = mask2d*sft(Fld_A0, nFPM, mB)
                
                    # pupil plane C before Lyot stop
                    Fld_C = Fld_A0 - isft(Fld_B, nPup, mB)
                    # pupil plane C after Lyot stop
                    Fld_L = Fld_C*LyotStop2d
                
                    # image plane D 
                    Fld_D = sft(Fld_L, nImg, mD*diam) #diametre_lyot *diam
                
                    # Intensity
                    Int_D += np.abs(Fld_D)**2    
                t1 = time.time()       
                # print(f'Coro image computation time: {t1-t0:.3f}s')  
            

                # Normalized intensity
                Int_D /= nOPD
            
                # Normalized intensity
                Int_D *= norm_peakD0#_no_lyot #* norm_peakD0
            

            
                """
                ### Compute the radial intensity profiles of the images
                """
                # computation of the averaged intensity profiles of the images   
                Int_D0_prf_avg, rad_D0_prf_avg = profile(Int_D0, ptype='mean')
                Int_D_prf_avg, rad_D_prf_avg = profile(Int_D, ptype='mean')


                # computation of the standard deviation intensity profiles of the images
                Int_D0_prf_std, rad_D0_prf_std = profile(Int_D0, ptype='std')
                Int_D_prf_std, rad_D_prf_std = profile(Int_D, ptype='std')



                # convert pixel scale into lam/D scale for the x-axis
                rad_D0_prf_avg_lamD = rad_D0_prf_avg * mD/nImg
                rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg
                
                rad_D0_prf_avg_mas = rad_D0_prf_avg_lamD * lamD2mas
                rad_D_prf_avg_mas = rad_D_prf_avg_lamD * lamD2mas
              
                """
                ### Save images
                """
                resultat_no_coro_w_turb_rad.append(rad_D0_prf_avg_mas)
                resultat_no_coro_w_turb_int.append(Int_D0_prf_avg)

                resultat_w_woro_w_turb_rad.append(rad_D_prf_avg_mas)
                resultat_w_woro_w_turb_int.append(Int_D_prf_avg)



np.save(fdir_res / 'resultat_no_coro_no_turb_rad_{wl}.npy', resultat_no_coro_no_turb_rad)
np.save(fdir_res / 'resultat_no_coro_no_turb_int_{wl}.npy', resultat_no_coro_no_turb_int)

np.save(fdir_res / 'resultat_w_coro_no_turb_rad_{wl}.npy', resultat_w_coro_no_turb_rad)
np.save(fdir_res / 'resultat_w_coro_no_turb_int_{wl}.npy', resultat_w_coro_no_turb_int)

np.save(fdir_res / 'resultat_no_coro_w_turb_rad_{wl}.npy', resultat_no_coro_w_turb_rad)
np.save(fdir_res / 'resultat_no_coro_w_turb_int_{wl}.npy', resultat_no_coro_w_turb_int)

np.save(fdir_res / 'resultat_w_woro_w_turb_rad_{wl}.npy', resultat_w_woro_w_turb_rad)
np.save(fdir_res / 'resultat_w_woro_w_turb_int_{wl}.npy', resultat_w_woro_w_turb_int)

np.save(fdir_res / 'throughput_{wl}.npy', throughput)
