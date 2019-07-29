#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:11:34 2019

@author: mndiaye
"""

#%%
"""
### Initilaization
"""
import numpy as np
import pylab as pl
import csv
from pathlib import Path
from astropy.io import fits
import vigan.ao as ao
#from pyzelda.utils import aperture

#%%
"""
### Parameters
"""
### Dimensions for csv data file
nVal = 120
nParams = 5

### Pupil dimension
nPup = 384

#%%
"""
### Paths
"""
fdir = Path('/Users/mndiaye/Dropbox/python/Coronagraphs/data/2D/turbulence/').resolve()
fname = 'Obs-Sequence-Parameters.csv'
fpath = fdir / fname

#%%
"""
### Read csv files from Alexis 
"""
DATA = np.zeros((nVal,nParams))
# DATA[:,0] = zenith distance [deg]
# DATA[:,1] = seeing ['']
# DATA[:,2] = R0 [m]
# DATA[:,3] = wind speed [m/s]
# DATA[:,4] = wind direction [deg]
cc = 0
with open(fpath, 'r') as csvFile:
   reader = csv.reader(csvFile)
   for row in reader:
       if cc>0:
           print(row[0],row[1],row[2],row[3],row[4])
           DATA[cc-1]=row
       cc = cc+1
csvFile.close()

#%%
"""
### Array of residual turbulence PSD
"""
residual_turbulence_psd_arr = np.zeros((nVal, 2*nPup, 2*nPup))

for iVal in range(nVal):    

    """
    ### Turbulence parameters
    """
    seeing   = DATA[iVal,1]
    L0       = 25
    z        = [2.5, 10]#[  0,   4, 16]
    Cn2      = [80, 20]#[ 55,  35, 10]
    v        = [DATA[iVal,3], 21.2]#[  8,  10, 15]
    arg_v    = [DATA[iVal,4], -23]#[  0, -45,  0]
    mag      = 3.29
    zenith   = DATA[iVal,0]
    azimuth  = 71
    spaf     = 0.5
    img_wave = 1.593e-6
    seed     = 12345
    
    #%%    
    """
    ### Residual turbulence PSD computation
    """
    residual_turbulence_psd_arr[iVal] = ao.residual_screen_sphere(seeing, L0, z, Cn2, v, arg_v, mag, zenith, azimuth, 
                                                            spat_filter=spaf, img_wave=img_wave, dim_pup=nPup,
                                                            n_screen=100, fit=True, servo=True, alias=True, noise=True,
                                                            diff_refr=True, psd_only=True, seed=seed)

    if iVal == 0:
        residual_turbulence_fourier = ao.residual_screen_sphere(seeing, L0, z, Cn2, v, arg_v, mag, zenith, azimuth, 
                                                        spat_filter=spaf, img_wave=img_wave, dim_pup=nPup,
                                                        n_screen=100, fit=True, servo=True, alias=True, noise=True,
                                                        diff_refr=True, psd_only=False, seed=seed)
        residual_turbulence_fourier  = residual_turbulence_fourier[..., nPup:2*nPup, nPup:2*nPup]
#        residual_turbulence_fourier *= pupil_saxo  
    
#%%
"""
### Save 
"""
### Save PSD temporal evolution
fdir = Path('/Users/mndiaye/Desktop/').resolve()
fname = 'residual_turbulence_psd_arr_nPup={0}.fits'.format(nPup)
fpath = fdir / fname 
#
fits.writeto(fpath, residual_turbulence_psd_arr, overwrite=True)

fdir = Path('/Users/mndiaye/Desktop/').resolve()
fname = 'residual_turbulence_phasescreen_arr_nPup={0}_iPSD={1}.fits'.format(nPup,0)
fpath = fdir / fname 
#
fits.writeto(fpath, residual_turbulence_fourier, overwrite=True)


    
#%%
"""
### Display PSD for iVal=0
"""
pl.figure(2)
pl.imshow(np.log10(residual_turbulence_psd_arr[0]), cmap = 'inferno')
pl.title('PSD')
pl.show()

#%%
print('\n ok')    