#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:38:08 2024

@author: mndiaye
"""

#%%
"""
### Initialization
"""
import numpy as np
from pathlib import Path

from astropy.io import fits
import matplotlib.pyplot as plt

import corono as coro

from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from numpy.fft import fftshift as fftshift

import time

#%%
"""
### Parameters
"""
# Pupil size
nPup = 400

# Zero padding
kPad = 8
nPad = kPad*nPup

# Number of phase screens
nOPD = 2000

# spatial frequency extent for sft
m = nPup*1

#%%
"""
### Directories, filenames and filepaths
"""
# Directory of the ANDES data
fdir_dat = Path('/Users/mndiaye/scratch/data/andes/data/').resolve()

# Filename and path for the pupil 
fname_pupil = f'ELT_pupil_{nPup}.fits'
fpath_pupil = fdir_dat / 'Pupil' / fname_pupil

# Directory for the all the OPDs 
fdir_allopd = fdir_dat / 'OPD' / '20240517_181033' 

# filename and path for a single OPD
fname_opd = 'screenNb500.fits'
fpath_opd = fdir_allopd / fname_opd

# array of filepaths for all the OPDs
fpath_allopd = [fdir_allopd / f'screenNb{iOPD + 500}.fits' for iOPD in range(nOPD)]

#%%
"""
### Read files
"""
# read file for the Pupil
Pupil2d = fits.getdata(fpath_pupil)
# read file for a single OPD map
OPD2d = fits.getdata(fpath_opd)

#%%
"""
### Zero padding
"""
# Set the averaged value of the OPD map inside the Pupil to zero
OPD2d -= OPD2d[Pupil2d == 1].mean()

# Multiply the OPD map by the Pupil
OPD2d *= Pupil2d

# Make zero padding for the OPD map and multiply by the Pupil
OPD2d_pad = np.zeros((nPad, nPad))

# Generate indices for inserting the OPD in the padded array
ini = (nPad-nPup)//2
end = (nPad+nPup)//2

# Insert the OPD map
OPD2d_pad[ini:end,ini:end] = OPD2d


#%%
"""
### Compute PSD for a single phase screen with the FFT
"""
# Compute the PSD by successively using fftshift, fft2 and fftshift  
PSD2d_pad = np.abs(fftshift(fft2(fftshift(OPD2d_pad), norm='ortho')))**2 


#%%
"""
### Compute variance of the wavefront errors from OPD and PSD
"""
# variance of the wavefront errors from the OPD map within the whole array (not just the pupil!!!)
var_opd = np.var(OPD2d_pad)
# variance of the wavefront errors from the PSD (beware that we have not considered the surface of the pupil in the calculation...)
var_psd = np.sum(PSD2d_pad)*(1/nPad)**2

# display the two values, they should be exactly equal
print(f'variance (OPD): {var_opd}')
print(f'variance (PSD): {var_psd}')

#%%
"""
### computed the averaged PSD for all the OPDs with FFTs
"""
# define the PSD2d array
mean_PSD2d_pad = np.zeros((nPad, nPad))

# define the OPD2d array
OPD2d_pad_i = np.zeros((nPad, nPad))

t0 = time.time()
# Loop over all the OPD maps (probably not the most efficient way...)
for iOPD in range(nOPD):
    # print the iteration number every 50 maps (just for visualization)
    if iOPD % 50 == 0:
        print(f'iOPD: {iOPD:04d}')
    # read the OPD map     
    OPD2d_i = fits.getdata(fpath_allopd[iOPD])
    # set the averaged value to zero indide the Pupil
    OPD2d_i -= OPD2d_i[Pupil2d == 1].mean()
    # multiply the OPD map by the Pupil 
    OPD2d_i *= Pupil2d
    # zero pad the generated OPD2d
    OPD2d_pad_i[ini:end,ini:end] = OPD2d_i
    # compute the PSD for a single map and sum it with the previous PSDs
    mean_PSD2d_pad += np.abs(fftshift(fft2(fftshift(OPD2d_pad_i), norm='ortho')))**2 
t1 = time.time()
print(f'Computation time: {t1-t0:.2f}s')

# Normalize the PSD with the number of maps
mean_PSD2d_pad /= nOPD

#%%
"""
### Generation of new phase screen from the computed averaged PSD
"""
# generate an array with random values from -0.5 to 0.5, following a uniform distribution
random_t = np.random.rand(nPad, nPad)-0.5
# use the previous array to convert a complex exponential with values uniformely distributed over the trigonometrice circle
exp_random_t = np.exp(1j*2.*np.pi*random_t)
# multiply the previous array by the square root of the averaged PSD
random_sqrt_psd = exp_random_t *np.sqrt(mean_PSD2d_pad) 
# Use Fourier inverse transform to convert it to a random OPD map
random_OPD2d_pad = fftshift(ifft2(fftshift(random_sqrt_psd), norm='ortho'))

# use of the real or the imaginary part of the generated maps (you have 2 maps for 1, by the way!)
random_OPD2d_pad = random_OPD2d_pad.real
# crop the array 
random_OPD2d = random_OPD2d_pad[ini:end, ini:end]

# set the averaged value of the map to zero
random_OPD2d -= np.mean(random_OPD2d[Pupil2d == 1])

# multiply it by the Pupil
random_OPD2d *= Pupil2d

#%%
#%%
"""
### Compute PSD for a single phase screen with the SFT
"""
# Compute the PSD by successively using fftshift, fft2 and fftshift  
PSD2d_bis = np.abs(coro.utils.sft(OPD2d, nPup, m, CtrBtwnPix=False))**2

#%%
"""
### computed the averaged PSD for all the OPDs with SFTs
"""
# define the PSD2d array
mean_PSD2d_bis = np.zeros((nPup, nPup))

# define the OPD2d array
OPD2d_bis_i = np.zeros((nPup, nPup))

t0 = time.time()
# Loop over all the OPD maps (probably not the most efficient way...)
for iOPD in range(nOPD):
    # print the iteration number every 50 maps (just for visualization)
    if iOPD % 50 == 0:
        print(f'iOPD: {iOPD:04d}')
    # read the OPD map     
    OPD2d_bis_i = fits.getdata(fpath_allopd[iOPD])
    # set the averaged value to zero indide the Pupil
    OPD2d_bis_i -= OPD2d_bis_i[Pupil2d == 1].mean()
    # multiply the OPD map by the Pupil 
    OPD2d_bis_i *= Pupil2d
    # compute the PSD for a single map and sum it with the previous PSDs
    mean_PSD2d_bis += np.abs(coro.utils.sft(OPD2d_bis_i, nPup, m, CtrBtwnPix=False))**2
t1 = time.time()
print(f'Computation time: {t1-t0:.2f}s')

# Normalize the PSD with the number of maps
mean_PSD2d_bis /= nOPD

#%%
"""
### Generation of new phase screen from the computed averaged PSD
"""
# generate an array with random values from -0.5 to 0.5, following a uniform distribution
random_bis_t = np.random.rand(nPup, nPup)-0.5
# use the previous array to convert a complex exponential with values uniformely distributed over the trigonometrice circle
exp_random_bis_t = np.exp(1j*2.*np.pi*random_bis_t)
# multiply the previous array by the square root of the averaged PSD
random_sqrt_psd_bis = exp_random_bis_t *np.sqrt(mean_PSD2d_bis) 
# Use Fourier inverse transform to convert it to a random OPD map
random_OPD2d_bis = coro.utils.isft(random_sqrt_psd_bis, nPup, m, CtrBtwnPix=False)

# use of the real or the imaginary part of the generated maps (you have 2 maps for 1, by the way!)
random_OPD2d_bis = random_OPD2d_bis.real

# set the averaged value of the map to zero
random_OPD2d_bis -= np.mean(random_OPD2d_bis[Pupil2d == 1])

# multiply it by the Pupil
random_OPD2d_bis *= Pupil2d

#%%
"""
### Display pupil
"""
plt.figure(0)
plt.clf()
plt.subplot(121)
plt.imshow(Pupil2d)
plt.title('Pupil')
plt.subplot(122)
plt.imshow(OPD2d)
plt.title('OPD map')

#%%
"""
### Display averaged PSD
"""
plt.figure(1)
plt.clf()
plt.subplot(121)
plt.imshow(OPD2d_pad)
plt.title(r'single OPD$_{ALC}$')
plt.subplot(122)
plt.imshow(np.log10(mean_PSD2d_pad))
plt.title('averaged PSD (FFT)')

#%%
"""
### Display of the random screen and its PSD
"""
plt.figure(2)
plt.clf()
plt.subplot(121)
plt.imshow(random_OPD2d_pad)
plt.title(r'random OPD$_{MND}$ (FFT)')
plt.subplot(122)
plt.imshow(np.log10(np.abs(random_sqrt_psd)**2))
plt.title('PSD')

#%%
"""
### Display of the original map and the random map
"""
plt.figure(3)
plt.clf()
plt.subplot(121)
plt.imshow(OPD2d)
plt.title(r'OPD$_{ALC}$')
plt.subplot(122)
plt.imshow(random_OPD2d)
plt.title(r'OPD$_{MND}$ (FFT)')

#%%
"""
### Display averaged PSD
"""
plt.figure(11)
plt.clf()
plt.subplot(121)
plt.imshow(OPD2d)
plt.title(r'single OPD$_{ALC}$')
plt.subplot(122)
plt.imshow(np.log10(mean_PSD2d_bis))
plt.title('averaged PSD (SFT)')

#%%
"""
### Display of the random screen and its PSD
"""
plt.figure(12)
plt.clf()
plt.subplot(121)
plt.imshow(random_OPD2d_bis)
plt.title(r'random OPD$_{MND}$ (SFT)')
plt.subplot(122)
plt.imshow(np.log10(np.abs(random_sqrt_psd_bis)**2))
plt.title('PSD')

#%%
"""
### Display of the original map and the random map
"""
plt.figure(13)
plt.clf()
plt.subplot(121)
plt.imshow(OPD2d)
plt.title(r'OPD$_{ALC}$')
plt.subplot(122)
plt.imshow(random_OPD2d_bis)
plt.title(r'OPD$_{MND} (SFT)$')

