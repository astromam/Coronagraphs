#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:26:29 2023

@author: mndiaye
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:20:57 2022

@author: mndiaye
"""

"""
### Initialization
"""
import sys
import numpy as np
import os
import pwd
from pyzelda.utils import aperture, imutils
from pathlib import Path

from astropy.io import fits

import matplotlib.pyplot as plt

import time

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

#%%
"""
### Parameters
"""
# spatial sampling
nPup   = 500   # pupil

# save file
do_sav = False
# display file
do_disp = True


do_zeropad = False
do_fits_zeropad = False
nArr = 520

#%%
"""
### Directory
"""
# simulation case and directory
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('~/scratch/data/Coronagraphs').expanduser()
        sim_case = 'test' # 'test' or 'server'
    elif syst == 'linux':
        fdir = Path('/scratch/{0}/data/Coronagraphs/'.format(user)).resolve()
        sim_case = 'server' # 'test' or 'server'            
    else:
        raise ValueError('Unknown operating system {0}'.format(user))
else:
    raise ValueError('Unknown user {0}'.format(user))

### old directories
fdir_pupils = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
fdir2_pupils = fdir / 'data' / '2D' / 'pupils'

#%%
"""
### Filenames for the sources
"""
fname_LyotStop2d      = 'sphere_stop_ST_ALC2.fits'

#%%
"""
### Filepaths for the file sources
"""
fpath_LyotStop2d      = fdir_pupils / fname_LyotStop2d    


#%%
"""
### Filenames for the sources to save
"""
fname2_LyotStop2d      = f'sphere_stop_ST_ALC2_nPup{nPup:04d}.fits'

#%%
"""
### Filepaths for the file sources
"""
fpath2_LyotStop2d      = fdir2_pupils / fname2_LyotStop2d    

#%% 
"""
### File reading
"""
### Lyot Stop
LyotStop2d_tmp = fits.getdata(fpath_LyotStop2d)
LyotStop2d     = imutils.scale(LyotStop2d_tmp, 0, new_dim=(nPup,nPup), method='interp')


#%%
"""
### Save input
"""

if do_sav:
    fits.writeto(fpath2_LyotStop2d, LyotStop2d, overwrite=True)

#%%
"""
### plot display
"""

if do_disp:
    plt.figure(21, figsize=(10,6))
    plt.clf()
    plt.subplot(1,1,1)
    plt.imshow(LyotStop2d)
    plt.title('Lyot stop')

#%%
"""
### Add zero padding
"""
if do_zeropad:
    LyotStop2dpad = np.zeros((nArr, nArr))
    nBeg = (nArr-nPup)//2
    nEnd = (nArr+nPup)//2
    LyotStop2dpad[nBeg:nEnd,nBeg:nEnd] = LyotStop2d*1.
    
    plt.figure(1)
    plt.clf()
    plt.imshow(LyotStop2dpad)

    plt.show()
    
    fname_pad = f'sphere_stop_ST_ALC2_nPup={nPup}_nArr={nArr}.fits'
    fpath_pad = fdir2_pupils / fname_pad
    print(fname_pad)
    
    if do_fits_zeropad:
        fits.writeto(fpath_pad, LyotStop2dpad*1., overwrite=True)
    
