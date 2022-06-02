#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:10:20 2022

@author: mndiaye
"""



# Initialization
import numpy as np
import pyzelda.utils.aperture as aperture

import pylab as pl
from pathlib import Path

from astropy.io import fits

import xaosim
import corono as coro
import os
import pwd
import sys

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform


#%%
"""
### Parameters
"""
nArr= 2048
pupil_name = 'tmt'
do_fits = False

nPup = 1920
do_fits = True



#%%
"""
### Directories
"""
#fdir = Path('/Users/mndiaye/Dropbox/python/Coronagraphs/data/2D/pupils').resolve()
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()
    elif syst == 'linux':
        fdir = Path('/scratch/mndiaye/data/Coronagraphs/data/2D/pupils/').resolve()
    else:
        raise ValueError('Unknown operating system {0}'.format(syst))
else:
    raise ValueError('Unknown user {0}'.format(user))

if pupil_name == 'tmt':
    folder_tel = 'tmt'
    fname_pup = 'TMT_Pupil_Amplitude_MACOS_Logical_With_Obscuration.fits'
else:    
    raise NameError(f'{pupil_name}: unknown pupil name')

fpath_pup = fdir / folder_tel / fname_pup


#%%
"""
### generation of a VLT like pupil
"""
Pupil2d = fits.getdata(fpath_pup)
#nPup = np.shape(Pupil2d)[0]

nBeg = (nArr-nPup)//2
nEnd = (nArr+nPup)//2

Pupil2d_sub = Pupil2d[nBeg:nEnd, nBeg:nEnd]

#%%
"""
### save pupil
"""

if pupil_name == 'tmt':
    fname_sub = f'TMT_Pupil_Amplitude_MACOS_Logical_With_Obscuration_nArr{nPup:04d}_nPup{nPup:04d}.fits'
else:    
    raise NameError(f'{pupil_name}: unknown pupil name')
        
fpath_sub = fdir / folder_tel / fname_sub

if do_fits is True:
     fits.writeto(fpath_sub, Pupil2d_sub, overwrite=True)


#%%
"""
### display vlt-like pupil
"""
#pupil_sbr = xaosim.pupil.subaru(nPup, nPup, nPup/2, between_pix=True)
#pupil_diff = pupil*1 - pupil_sbr*1

pl.figure(0)
pl.clf()
pl.subplot(121)
pl.imshow(Pupil2d)

pl.subplot(122)
pl.imshow(Pupil2d_sub)

pl.show()

# a = coro.utils.uniform_disk(nPup, (nPup/2)*kpdiam, CtrBtwnPix=True)

# pl.figure(0)
# pl.clf()
# pl.imshow(a*pupil)

