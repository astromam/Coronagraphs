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
Pupil = fits.getdata(fpath_elt,)


    #%%
"""
### Read OPDs
"""

fdir_opd_AL = Path('/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/JQ2_PASSATA/JQ2_01/20240517_181033/').resolve() #JQ2

# Filename and path for the OPD maps
flist_opd = os.listdir(fdir_opd_AL) 
fpath_opd = [fdir_opd_AL / flist_opd[i] for i in range(len(flist_opd)) if flist_opd[i].endswith('.fits')]
            
fpath_opd = sorted(fpath_opd)
                        
t0 = time.time()
# Read OPD maps for the nOPD files
data_windshake = np.asarray([fits.getdata(fpath_opd[i]) for i in range(2000)])*1e-9 # in nm


#%%


# Determine PSD of OPDs

OPD_only_pupil = data_windshake*Pupil

#%%

"""
Filtrage Carré
"""
# tab = np.zeros((400,400))
# N = 400 # Number of points in each dimension
# ind_init = (N-200)//2
# ind_end = (N+200)//2
# tab[ind_init:ind_end,ind_init:ind_end]=1
# plt.imshow(tab)
# plt.show()

#%%
"""
Filtrage Gaussien
"""
# Dimensions de la matrice
N = 400
# Créer une matrice de coordonnées
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
x, y = np.meshgrid(x, y)

# Paramètres de la gaussienne
sigma = 0.3  # Écart-type

# Calcul de la distribution gaussienne
gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Normalisation pour que les valeurs soient entre 0 et 1
gaussian /= gaussian.max()

# Affichage de la matrice résultante
plt.imshow(gaussian)
plt.show()

#%%
mean_psd = 0*OPD_only_pupil[0]
opd_filtré = 0*OPD_only_pupil
mD = 400

### filtrage de l'OPD en gaussien 
tab_gauss = np.zeros((400,400))
N = 400 # Number of points in each dimension
i=0

"""
Test filtrage OPD ALC 
"""
for opds in OPD_only_pupil : 
    tf_opd = sft(opds,400,mD)
    tf_opd_filtre = tf_opd*gaussian
    opd_filtré_o = isft(tf_opd_filtre,400,mD)
    opd_filtré[i] = np.real(opd_filtré_o)*Pupil
    i+=1

np.save('/Users/asimonnin/Desktop/PhD/Andes/results/OPD_Adrien/OPD_filtré_AL7.npy',opd_filtré)
np.save('/Users/asimonnin/Desktop/PhD/Andes/results/OPD_Adrien/OPD_AL7.npy',OPD_only_pupil)
#%%

"""
Génération PSD moyenne OPDs ALC
"""
for opds in OPD_only_pupil : 
    tf_opd = sft(opds,400,mD)  
    psd = np.abs(tf_opd)**2
    mean_psd += psd

mean_psd = mean_psd/len(OPD_only_pupil)

#%%
# plt.imshow(opd_filtré[0])
# plt.colorbar()
# plt.show()

# plt.imshow(OPD_only_pupil[0])
# plt.colorbar()
# plt.show()

#%%

"""
Mettre moyenne OPDs à 0
"""
# mean_test = np.mean(data_windshake[:,Pupil==1],axis=1)
# data_mean_0 = 0*data_windshake

# for i in range(2000):
#     data_mean_0[i] = (data_windshake[i]*Pupil) - mean_test[i]

# plt.imshow(data_mean_0[0])
# plt.colorbar()
# plt.show()

#%%
"""
Génération d'OPDs aléatoires à partir PSD moyenne (avec et sans filtrage)
"""

for i in range(2000):
    amplitude = np.sqrt(mean_psd)
    random = np.random.randn(*amplitude.shape)
    random_opd = isft(amplitude*random,400,mD)
    random_opd_real = np.real(random_opd)*Pupil ### OPDs non filtrés

    ### filtrage de l'OPD

    tf_opd_flt = sft(random_opd,400,mD)
    tf_opd_flt_tab = tf_opd_flt*gaussian
    random_opd_tab = isft(tf_opd_flt_tab,400,mD)
    random_opd_tab_real = np.real(random_opd_tab)*Pupil ### OPDs filtrés

    ## save as fits file
    fpath_new_OPD = '/Users/asimonnin/Desktop/PhD/Andes/results/OPD_Adrien/no_filtered_windshake_JQ2/OPD_'+str(i)+'.fits'
    fpath_new_OPD_tab = '/Users/asimonnin/Desktop/PhD/Andes/results/OPD_Adrien/filtered_windshake_JQ2/OPD_'+str(i)+'_filtered_2.fits'
    fits.writeto(fpath_new_OPD, random_opd_real, overwrite=True)
    fits.writeto(fpath_new_OPD_tab, random_opd_tab_real, overwrite=True)


#%%

# New set of OPDs from PASSATA 
Adrien_OPDs_filtered =  Path('/Users/asimonnin/Desktop/PhD/Andes/results/OPD_Adrien/filtered_windshake_JQ2/').resolve()
Adrien_OPDs_no_filtered =  Path('/Users/asimonnin/Desktop/PhD/Andes/results/OPD_Adrien/no_filtered_windshake_JQ2/').resolve()

fdir_opd_filtered   = Adrien_OPDs_filtered #new_spider_flare
fdir_opd_no_filtered = Adrien_OPDs_no_filtered

# Filename and path for the filtered OPD maps
flist_opd_filtered = os.listdir(fdir_opd_filtered) 
nOPD_filtered = len(flist_opd_filtered)
fpath_opd_filtered = [fdir_opd_filtered / flist_opd_filtered[i] for i in range(nOPD_filtered)]
fpath_opd_filtered = sorted(fpath_opd_filtered)

# Filename and path for the no filtered OPD maps
flist_opd_no_filtered = os.listdir(fdir_opd_no_filtered)
nOPD_no_filtered = len(flist_opd_no_filtered)
fpath_opd_no_filtered = [fdir_opd_no_filtered / flist_opd_no_filtered[i] for i in range(nOPD_no_filtered)]
fpath_opd_no_filtered = sorted(fpath_opd_no_filtered)

mean_jq2 = 125*1e-9#103*1e-9
std_jq2 =1.1*1e-8# 2*1e-9

rand_norm = np.random.normal(mean_jq2,std_jq2,2000)

"""
### Read file
"""
t0 = time.time()
# Read OPD maps for the nOPD files for filtered OPDs
OPD_arr_filtered = np.asarray([fits.getdata(fpath_opd_filtered[i]) for i in range(nOPD_filtered)])
OPD_arr_filtered = OPD_arr_filtered# convert OPD from nm `to m if new OPD with new pupil
OPD_arr_filtered_normalised = 1*OPD_arr_filtered

# Read OPD maps for the nOPD files for no filtered OPDs
OPD_arr_no_filtered = np.asarray([fits.getdata(fpath_opd_no_filtered[i]) for i in range(nOPD_no_filtered)])
OPD_arr_no_filtered = OPD_arr_no_filtered# convert OPD from nm `to m if new OPD with new pupil
OPD_arr_no_filtered_normalised = 1*OPD_arr_no_filtered



for i in range(2000):
    OPD_arr_filtered_normalised[i,Pupil==1] = (OPD_arr_filtered_normalised[i,Pupil==1] /np.std(OPD_arr_filtered_normalised[i,Pupil==1]) *rand_norm[i])
    OPD_arr_no_filtered_normalised[i,Pupil==1] = (OPD_arr_no_filtered_normalised[i,Pupil==1] /np.std(OPD_arr_no_filtered_normalised[i,Pupil==1]) *rand_norm[i])

mean_psd_new = 0*OPD_arr_filtered[0]
mean_psd_new_no_filtered = 0*OPD_arr_no_filtered[0]

for opds in OPD_arr_filtered_normalised : 

    tf_opd = sft(opds,400,400)  
    psd = np.abs(tf_opd)**2
    mean_psd_new += psd

mean_psd_new = mean_psd_new/len(OPD_arr_filtered)

for opds in OPD_arr_no_filtered_normalised :
    
    tf_opd = sft(opds,400,400)  
    psd = np.abs(tf_opd)**2
    mean_psd_new_no_filtered += psd

mean_psd_new_no_filtered = mean_psd_new_no_filtered/len(OPD_arr_no_filtered)

# %%
std_spat = np.std(OPD_arr_filtered_normalised[:,Pupil==1],axis=1)

temp_mean = np.mean(std_spat)
temp_std = np.std(std_spat)

std_spat_no_filtered = np.std(OPD_arr_no_filtered_normalised[:,Pupil==1],axis=1)

temp_mean_no_filtered = np.mean(std_spat_no_filtered)
temp_std_no_filtered = np.std(std_spat_no_filtered)

plt.imshow(np.log10(mean_psd_new))
plt.colorbar()
plt.title('PSD moyenne OPDs filtrés')
plt.show()

plt.imshow(np.log10(mean_psd_new_no_filtered))
plt.colorbar()
plt.title('PSD moyenne OPDs non filtrés')
plt.show()

plt.imshow(mean_psd)
plt.colorbar()
plt.title('PSD moyenne OPDs ALC')
plt.show()


# %%
"""
Génération OPDs avec PSD suivant une loi en f^-2
"""

# Définir le domaine de fréquence
f_min = 0.01  # Fréquence minimale (éviter zéro pour éviter l'infini)
f_max = 100.0 # Fréquence maximale
num_points = 1000
frequencies = np.linspace(f_min, f_max, num_points)

# Appliquer la loi f^-2 pour obtenir la DSP
dsp = 1 / frequencies**2

plt.imshow(dsp)
plt.show()

OPD_dsp = np.abs(sft(OPD_only_pupil[0],400,mD))**2
# %%
# Dimensions de l'image
N = 800
# Créer un espace de fréquence en 2D
kx = (np.arange(N)-N//2)/(N/2)  # Fréquences spatiales pour les colonnes
ky = (np.arange(N)-N//2)/(N/2)
kx2, ky2 = np.meshgrid(kx, ky)
k = np.sqrt(kx2**2 + ky2**2)  # Norme des vecteurs de fréquences

# Appliquer la loi f^-2 pour obtenir la DSP
# Pour éviter la division par zéro, on ajoute un petit epsilon à k
epsilon = 1e-10
dsp = 1/(k**2 + epsilon)

# Génération de phases aléatoires
# Calcul des composantes complexes du spectre
amplitude = np.sqrt(dsp)
random = np.random.randn(*amplitude.shape)
opd_ff = amplitude * random

# Transformée de Fourier inverse pour obtenir l'image spatiale
random_opd_f2 = np.real(isft(amplitude*random,800,400))

# Normalisation de l'image pour l'affichage
opd_f2 = (random_opd_f2  - np.min(random_opd_f2 )) / (np.max(random_opd_f2 ) - np.min(random_opd_f2 ))
# Affichage de l'image
plt.imshow(opd_f2, cmap='gray')
plt.colorbar()
plt.title('Image avec DSP ~ f^-2')
# plt.axis('off')
plt.show()
# %%

plt.imshow(np.log(dsp),vmax=3)
plt.colorbar()