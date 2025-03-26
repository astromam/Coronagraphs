#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:32:56 2025

@author: mndiaye
"""


"""
### Initialization
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt

#%%
"""
### Parameters
"""
nPup0 = 1028#514
nPup = 1128#564

ini = (nPup-nPup0)//2
end = (nPup+nPup0)//2

do_fits = False

#%%
"""
### Directory
"""
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/vlt_btw').resolve()

fname_old = f'vlt_btw_APLC_obs=0.14_lsid=0.28_lsod=1.00_IWA=0.0_OWA=20.0_BW=0.20_nlam=03_1D_N={nPup:04d}_nFPM=50.000000_rMask=2.252MaxContrastL1_tau=0.756_LSRobustness=1_gurobipy_deadactLSRcoeff_v9=11314.fits'
fpath_old = fdir / fname_old

fname_new = fname_old.replace(f'N={nPup:04d}', f'N={nPup0:04d}')
fpath_new = fdir / fname_new
#%%
"""
### read files
"""

apod_old = fits.getdata(fpath_old)

#%%
"""
### crop file
"""
apod_new = apod_old[ini:end, ini:end]

#%%
"""
### solve file
"""
if do_fits:
    fits.writeto(fpath_new, apod_new, overwrite=True)

#%%
"""
### plot
"""
plt.figure(0)
plt.clf()
plt.imshow(apod_old)

plt.figure(1)
plt.clf()
plt.imshow(apod_new)