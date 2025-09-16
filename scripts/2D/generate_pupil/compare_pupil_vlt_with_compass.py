#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:55:03 2025

@author: mndiaye
"""

#%%
"""
### Initialization
"""
# Initialization
import numpy as np
#import pyzelda.utils.aperture as aperture
#from vigan.optics import aperture

import matplotlib.pyplot as plt
from pathlib import Path

from astropy.io import fits

import xaosim

#%%
"""
### Parameters
"""
pupil_name = 'vlt_btw5'
do_dead_act = False
nPup= 800 #384
do_fits = False

do_zeropad = False
do_fits_zeropad = False
nArr = 400 #384

#%%
"""
### Pupil directory
"""
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils').resolve()
str_dead_act=''
if do_dead_act and pupil_name == 'vlt':
    str_dead_act = '_dead_act'

fname_mnd = f'pupil={pupil_name}_nPup={nPup}{str_dead_act}.fits'
fpath_mnd = fdir / fname_mnd
print(fname_mnd)


fname_jmo = 'vlt_pup_in_compass.fits'
fpath_jmo = fdir / fname_jmo


fdir_plt = Path('/Users/mndiaye/Downloads').resolve()
fname_plt = 'vlt_pupil_comparison5.pdf'
fpath_plt = fdir_plt / fname_plt 


#%%
"""
### Read pupil
"""

pupil_mnd = fits.getdata(fpath_mnd)
pupil_jmo = fits.getdata(fpath_jmo)


#%%
"""
### Display pupil
"""



plt.figure(0, (12, 4.5))
plt.clf()
plt.subplot(131)
plt.imshow(pupil_mnd)
plt.title('Pupil (MND)')
plt.subplot(132)
plt.imshow(pupil_jmo)
plt.title('Pupil (JMO)')
plt.subplot(133)
plt.imshow(pupil_jmo-pupil_mnd)
plt.title('Difference')
plt.savefig(fpath_plt)


#%%
plt.figure(1, (8, 4.5))
plt.clf()
plt.imshow(pupil_jmo-pupil_mnd)