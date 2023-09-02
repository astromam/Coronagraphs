#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:51:51 2023

@author: mndiaye
"""

import numpy as np
import matplotlib.pyplot as plt

import os 
import pwd
import sys

from pathlib import Path

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

from astropy.io import fits

#%%
"""
### Parameters
"""
# keyword to save fits file
do_fits = False

#%%
"""
### Directory 
"""
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()
    elif syst == 'linux':
        fdir = Path('/scratch/mndiaye/data/Coronagraphs/data/2D/pupils/').resolve()
    else:
        raise ValueError('Unknown operating system {0}'.format(syst))
else:
    raise ValueError('Unknown user {0}'.format(user))

#%%
"""
### file
"""
#filename = f'sphere_stop_ST_ALC2_nPup{N:04d}.fits'
#filepath = fdir / filename
filename_t = []

for file in os.listdir(fdir):
    if file.startswith("sphere_stop_ST_ALC2_nPup") and not file.endswith("_ud.fits") and not file.endswith("_lr.fits") and not file.endswith("_lrud.fits"):
        print(os.path.join("/mydir", file))
        filename_t.append(file)
        
filepath_t = [fdir / filename_t[i] for i in range(len(filename_t))]
nfile = len(filepath_t)

#%%
for i in range(nfile):

    filename_lr = filename_t[i].rstrip('.fits') +  '_lr.fits'
    filename_ud = filename_t[i].rstrip('.fits') +  '_ud.fits'
    filename_lrud = filename_t[i].rstrip('.fits') +  '_lrud.fits'
    
    filepath_lr = fdir / filename_lr
    filepath_ud = fdir / filename_ud
    filepath_lrud = fdir / filename_lrud
    
    #%%
    """
    ### read file
    """
    LyotStop2d = fits.getdata(filepath_t[i])
    
    #%%
    """
    ### Flip files
    """
    LyotStop2d_lr = np.fliplr(LyotStop2d)
    LyotStop2d_ud = np.flipud(LyotStop2d)
    LyotStop2d_lrud = np.fliplr(np.flipud(LyotStop2d))
    
    #%%
    """
    ### Save files
    """
    if do_fits:
        fits.writeto(filepath_lr, LyotStop2d_lr, overwrite=True)
        fits.writeto(filepath_ud, LyotStop2d_ud, overwrite=True)
        fits.writeto(filepath_lrud, LyotStop2d_lrud, overwrite=True)
    

#%%
"""
### Display Lyotstops
"""
plt.figure(0, (16, 4))
plt.clf()
plt.subplot(141)
plt.imshow(LyotStop2d)
plt.title('original')
plt.subplot(142)
plt.imshow(LyotStop2d_lr)
plt.title('left-right')
plt.subplot(143)
plt.imshow(LyotStop2d_ud)
plt.title('up-down')
plt.subplot(144)
plt.imshow(LyotStop2d_lrud)
plt.title('lrud')
