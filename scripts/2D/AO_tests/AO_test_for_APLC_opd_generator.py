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
import time
import os

#%%
"""
### Parameters
"""
### Dimensions for csv data file
nParams = 5
nPSD    = 120
kPSD    = 120
iPSD    = 0

### Pupil dimension
nPup = 384

### OPD number
nmap = 1

do_fits = True

#%%
"""
### File reading paths
"""
fdir_psd_csv  = Path('../../../data/2D/turbulence/').resolve()
fname_psd_csv = 'Obs-Sequence-Parameters.csv'
fpath_psd_csv = fdir_psd_csv / fname_psd_csv

#%%
"""
### File saving paths from the generated data
"""
### Save PSD temporal evolution
fdir_opd = Path('../../../data/2D/AO_tests/').resolve()
if not os.path.exists(fdir_opd):
    os.makedirs(fdir_opd)

fname_psd = 'AOres_psd_nPup={0}_nPSD={1:04d}.fits'.format(nPup,kPSD)
fpath_psd = fdir_opd / fname_psd 

fname_opd = 'AOres_opd_nPup={0}_iPSD={1:04d}_nmap={2:04d}.fits'.format(nPup,iPSD,nmap)
fpath_opd = fdir_opd / fname_opd 


#%%
"""
### Read csv files from Alexis (adpated from Elodie Choquet's file)
"""
DATA = np.zeros((nPSD,nParams))
# DATA[:,0] = zenith distance [deg]
# DATA[:,1] = seeing ['']
# DATA[:,2] = R0 [m]
# DATA[:,3] = wind speed [m/s]
# DATA[:,4] = wind direction [deg]
cc = 0
with open(fpath_psd_csv, 'r') as csvFile:
   reader = csv.reader(csvFile)
   for row in reader:
       if cc>0:
#           print(row[0],row[1],row[2],row[3],row[4])
           DATA[cc-1]=row
       cc = cc+1
csvFile.close()

#%%
"""
### Array of residual turbulence PSD
"""
residual_turbulence_psd_arr = np.zeros((kPSD, 2*nPup, 2*nPup))

for iPSD in range(kPSD):    

    """
    ### Turbulence parameters
    """
    seeing   = DATA[iPSD,1]
    L0       = 25
    z        = [2.5, 10]#[  0,   4, 16]
    Cn2      = [80, 20]#[ 55,  35, 10]
    v        = [DATA[iPSD,3], 21.2]#[  8,  10, 15]
    arg_v    = [DATA[iPSD,4], -23]#[  0, -45,  0]
    mag      = 3.29
    zenith   = DATA[iPSD,0]
    azimuth  = 71
    spaf     = 0.5
    img_wave = 1.593e-6
    seed     = 12345
    
    #%%    
    """
    ### Residual turbulence PSD computation
    """
    print('\niPSD: {0:03d}/{1:03d}'.format(iPSD+1,kPSD))
    t0 = time.time()
    residual_turbulence_psd_arr[iPSD] = ao.residual_screen_sphere(seeing, L0, z, Cn2, v, arg_v, mag, zenith, azimuth, 
                                                            spat_filter=spaf, img_wave=img_wave, dim_pup=nPup,
                                                            n_screen=nmap, fit=True, servo=True, alias=True, noise=True,
                                                            diff_refr=True, psd_only=True, seed=seed+iPSD)
    t1 = time.time()
    print('\nPSD - exec time: {0}s\n'.format(t1-t0))

#    if iPSD == 0:
#    t0 = time.time()
#    residual_turbulence_opd_arr = ao.residual_screen_sphere(seeing, L0, z, Cn2, v, arg_v, mag, zenith, azimuth, 
#                                                    spat_filter=spaf, img_wave=img_wave, dim_pup=nPup,
#                                                    n_screen=nmap, fit=True, servo=True, alias=True, noise=True,
#                                                    diff_refr=True, psd_only=False, seed=seed+iPSD)
#    residual_turbulence_opd_arr  = residual_turbulence_opd_arr[..., nPup:2*nPup, nPup:2*nPup]
#        residual_turbulence_opd_arr *= pupil_saxo
    t1 = time.time()
    print('opd generation - exec time: {0:.2f}s\n'.format(t1-t0))
    
    #%%
    """
    ### Save PSD
    """
    if do_fits:
       fits.writeto(fpath_psd, residual_turbulence_psd_arr, overwrite=True)

    #%%
    """
    ### Save OPD maps from PSDs
    """
    fname_opd = 'AOres_opd_nPup={0}_iPSD={1:04d}_nmap={2:04d}.fits'.format(nPup,iPSD,nmap)
    fpath_opd = fdir_opd / fname_opd
#    if do_fits:
#        fits.writeto(fpath_opd, residual_turbulence_opd_arr, overwrite=True)
    
#%%
#"""
#### Display PSD for iPSD=0
#"""
#pl.figure(2)
#pl.imshow(np.log10(residual_turbulence_psd_arr[0]), cmap = 'inferno')
#pl.title('PSD')
#pl.show()

#%%
print('\n ok')    