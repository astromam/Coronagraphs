#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:45:21 2022

@author: mndiaye
"""

#%%
"""
### Initialization
"""
import os
import pwd
import sys

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

import numpy as np
import pandas as pd
from pathlib import Path

import corono as coro

import matplotlib.pyplot as plt

from astropy.io import fits

#%%
"""
### Directory
"""
# directory of the file provided by Lyu for original apodization
# fdir_apod = Path('/Users/mndiaye/Nextcloud/SPHERE_upgrade/WP_corono/2009_apodizers').resolve()
fdir_apod = Path('/Users/mndiaye/Nextcloud/SPHERE_upgrade/WP_corono/specifications/1.3.0/').resolve()

# filename of the xls file for the original apodization
fname_apod1d_xls = '2023-08-03_IR_Apodizer_DesignA_profile_v1.3.0_python.xls'
# filepath of the xls file for the original apodization
fpath_apod1d_xls = fdir_apod / fname_apod1d_xls

# filename of the 2D apodizations (initial and binary) 
fname_apod2d_ini = 'apod_initial'
fname_apod2d_bin = 'apod_binary'

# filename in fits format
fname_apod2d_ini_fits = fname_apod2d_ini + '.fits'
fname_apod2d_bin_fits = fname_apod2d_bin + '.fits'

# filename in png format
fname_apod2d_ini_png = fname_apod2d_ini + '.png'
fname_apod2d_bin_png = fname_apod2d_bin + '.png'

# filepath for fits format
fpath_apod2d_ini_fits = fdir_apod / fname_apod2d_ini_fits
fpath_apod2d_bin_fits = fdir_apod / fname_apod2d_bin_fits

# filepath for png format
fpath_apod2d_ini_png = fdir_apod / fname_apod2d_ini_png
fpath_apod2d_bin_png = fdir_apod / fname_apod2d_bin_png

#%%
"""
### define FLoyd-Steinberg algorithm function
"""
def floyd_steinberg(array):
    
    n0 = np.shape(array)[0]
    n1 = np.shape(array)[1]
    
    new_array = array*1.
    
    for j in range(n1):
        for i in range(n0):
            
            old_pixel = new_array[i, j]
            new_pixel = 0.
            if old_pixel >0.5:
                new_pixel = 1.
            new_array[i, j] = new_pixel
            quant_error = old_pixel - new_pixel
            
            if i != n0-1: 
                new_array[i + 1, j    ] += quant_error * 7. / 16.
            if i != 0 and j != n1-1:
                new_array[i - 1, j + 1] += quant_error * 3. / 16.
            if j != n1-1:    
                new_array[i    , j + 1] += quant_error * 5. / 16.
            if i != n0-1 and j != n1-1: 
                new_array[i + 1, j + 1] += quant_error * 1. / 16.
    
    return new_array

#%%
"""
### Parameters
"""
# save file in fits file
do_fits = 1

# save file in png file
do_png = 1 

# parameter to center things between pixels or not 
val = 1/2

# number of pixels for the apodized version
nPts = 3600

#%%
"""
### Read xls file with pandas
"""
apod1d_file = pd.read_excel(fpath_apod1d_xls)

#%%
"""
### Apod file
"""
apod1d_file.info()

apod1d_file_colname = apod1d_file.columns

dist1d_d = apod1d_file[apod1d_file_colname[0]]
ampl1d_d = apod1d_file[apod1d_file_colname[1]]

dist1d_r = dist1d_d[dist1d_d >= 0]
ampl1d_r = ampl1d_d[dist1d_d >= 0]

nDim = 2*len(dist1d_r)

#%%
"""
### Polynomial approximation of the amplitude profile
"""
z = np.polyfit(dist1d_r, ampl1d_r, 16)

ampl1d_poly = np.poly1d(z)


#%%

ampl1d_r_fit = ampl1d_poly(dist1d_r) 

#%%
"""
### Conversion of the 1d radial profile into 2d profile for radius
"""
xx,yy  = np.meshgrid(np.arange(nPts)-nPts/2+val, np.arange(nPts)-nPts/2+val)
mydist2d = np.hypot(yy,xx)/nPts

mydist2d_mm = mydist2d*max(dist1d_r)/0.5

#%%
"""
### Build apodizer from polynomial function
"""
# build apodizer from polynomial function
apod2d_ini = ampl1d_poly(mydist2d_mm)
# set negative points to 0
apod2d_ini[apod2d_ini < 0.] = 0.

#%%
"""
### apply Floyd-Steinberg algorithm
"""
apod2d_bin = floyd_steinberg(apod2d_ini)

#%%
"""
### Save file
"""
if do_fits:
    fits.writeto(fpath_apod2d_ini_fits, apod2d_ini, overwrite=True)
    fits.writeto(fpath_apod2d_bin_fits, apod2d_bin, overwrite=True)
    
#%%
"""
### Plot the file
"""
plt.figure(0)
plt.clf()
plt.plot(dist1d_d, ampl1d_d)
plt.title('amplitude apodizer profile')

#%%
plt.figure(1)
plt.clf()
plt.plot(dist1d_r, ampl1d_r)
plt.plot(dist1d_r, ampl1d_r_fit, ls='--')
plt.xlabel("r")
plt.ylabel("Normalized amplitude")
plt.title('amplitude apodizer profile')

#%%
plt.figure(2)
plt.clf()
plt.plot(dist1d_r, ampl1d_r_fit-ampl1d_r)
plt.xlabel("r")
plt.ylabel("Normalized amplitude")
plt.title('absolute difference')

#%%
plt.figure(3)
plt.clf()
plt.imshow(mydist2d)

#%%
plt.figure(4)
plt.clf()
plt.imshow(apod2d_ini, vmin=0, vmax=1, cmap='inferno')
plt.title('apodizer from profile')

#%%
if nPts == 3600:
    plt.figure(5)
    plt.clf()
    plt.plot(dist1d_r, ampl1d_r)
    plt.plot(dist1d_r, ampl1d_r_fit, ls='--')
    plt.plot(dist1d_r+val/nPts, apod2d_ini[nPts//2, nPts//2:], ls='-.')
    plt.xlabel("r")
    plt.ylabel("Normalized amplitude")
    plt.title('amplitude apodizer profile')

#%%
if nPts == 3600:
    plt.figure(6)
    plt.clf()
    plt.plot(dist1d_r, ampl1d_r_fit-ampl1d_r)
    plt.plot(dist1d_r, apod2d_ini[nPts//2, nPts//2:]-ampl1d_r)
    plt.xlabel("r")
    plt.ylabel("Normalized amplitude")
    plt.title('absolute difference')

#%%
plt.figure(10)
plt.clf()
plt.subplot(121)
plt.imshow(apod2d_ini, vmin=0, vmax=1, cmap='inferno')
plt.title('apodizer from profile')
plt.subplot(122)
plt.imshow(apod2d_bin, vmin=0, vmax=1, cmap='inferno')
plt.title('apodizer from profile')

#%%
plt.figure(11)
plt.clf()
plt.imshow(apod2d_ini, cmap='Greys_r')
if do_png:
    plt.savefig(fpath_apod2d_ini_png)

plt.figure(12)
plt.clf()
plt.imshow(apod2d_bin, cmap='Greys_r')
if do_png:
    plt.savefig(fpath_apod2d_bin_png)
