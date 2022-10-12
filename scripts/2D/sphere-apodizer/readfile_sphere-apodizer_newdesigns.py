#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:33:18 2022

@author: mndiaye
"""

"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from pathlib import Path

from astropy.io import fits

#%%
"""
### Directory
"""

fdir_pupils = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/Coronagraphs/data/2D/pupils/').resolve()

fdir_sphere = fdir_pupils / 'SPHERE'
fdir_newapod = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/Coronagraphs/results/2D/dat_pyth/vlt').resolve()

#%% vlt pupil
fname_vlt = 'pupil_vlt_nPup384.fits'
fpath_vlt = fdir_pupils / fname_vlt

#%% Old apodizer
fname_oldapod_alc1 = 'SPHERE_APO1_field_transmission_map.fits'
fpath_oldapod_alc1 = fdir_sphere / fname_oldapod_alc1

fname_oldapod_alc2 = 'SPHERE_APO2_field_transmission_map.fits'
fpath_oldapod_alc2 = fdir_sphere / fname_oldapod_alc2


#%% New apodizer
fname_newapod = 'vlt_APLC_obs=0.14_lsid=0.28_lsod=1.00_IWA=2.0_OWA=20.0_BW=0.20_nlam=05_1D_N=0384_nFPM=50.000000_rMask=2.252MaxContrastL1_tau=0.756_stdgrb.fits'
fpath_newapod = fdir_newapod / fname_newapod

#%%
fname_stp = 'sphere_stop_ST_ALC2.fits'
fpath_stp = fdir_sphere / fname_stp

#%%
"""
### Save results
"""
fdir_res = Path('/Users/mndiaye/Downloads').resolve()


#%%
"""
### Read file
"""
vltpupil = fits.getdata(fpath_vlt)
oldapod_alc1 = fits.getdata(fpath_oldapod_alc1)
oldapod_alc2 = fits.getdata(fpath_oldapod_alc2)
newapod = fits.getdata(fpath_newapod)
lyotstop = fits.getdata(fpath_stp)


#%%
"""
### Display plot
"""
fname_res = 'binary_apodizer.pdf'
fpath_res = fdir_res / fname_res

f0 = plt.figure(0)
plt.clf()
im = plt.imshow(newapod, cmap = 'inferno', extent=[-0.5,0.5,-0.5,0.5])
plt.title('NEW APODIZER')

f0.subplots_adjust(right=0.8)
cbar_ax = f0.add_axes([0.80, 0.05, 0.05, 0.9])
cbar    = f0.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Normalized amplitude', rotation=270, labelpad = 20)
plt.tight_layout()
plt.savefig(fpath_res)
plt.show() 


#%%
fname_res = 'SPHERE_APO1_field_transmission_map.pdf'
fpath_res = fdir_res / fname_res

f1 = plt.figure(1, (7, 5))
plt.clf()
im = plt.imshow(vltpupil*oldapod_alc1, cmap='inferno', extent=[-0.5,0.5,-0.5,0.5])
plt.title('APO1')

f1.subplots_adjust(right=0.8)
cbar_ax = f1.add_axes([0.80, 0.05, 0.05, 0.9])
cbar    = f1.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Normalized amplitude', rotation=270, labelpad = 20)
plt.tight_layout()
plt.savefig(fpath_res)
plt.show() 

#%%
fname_res = 'sphere_stop_ST_ALC2.pdf'
fpath_res = fdir_res / fname_res

f2 = plt.figure(2, (7, 5))
plt.clf()
im = plt.imshow(lyotstop, cmap='inferno', extent=[-0.5,0.5,-0.5,0.5])
plt.title('LYOT STOP')

plt.savefig(fpath_res)
plt.show() 

