#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:35:58 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

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
nPup= 1200
do_fits = True

pdiam, odiam = 7.92, 2.3  # tel. and obst. diameters (meters)
thick = 0.25              # adopted spider thickness (meters)
offset = 1.278            # spider intersection offset (meters)
beta = 51.75              # spider angle beta

kpdiam_t =  np.linspace(0.98, 1.0, 3)#np.linspace(0.9, 1.0, 11) #
kodiam_t =  np.linspace(1.0, 1.2, 5)#np.linspace(1.0, 2.0, 21) #
kthick_t = np.linspace(1.0, 1.2, 3)#np.linspace(1.0, 2.0, 11) # 

pdiam2_t = np.asarray(kpdiam_t)*pdiam
odiam2_t = np.asarray(kodiam_t)*odiam
thick2_t = np.asarray(kthick_t)*thick 

npdiam = len(kpdiam_t)
nodiam = len(kodiam_t)
nthick = len(kthick_t)

nIter = npdiam*nodiam*nthick

kwd_spiders = [True]
if thick <= 0.:
    kwd_spiders = False

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


#%%
"""
### generation of a VLT like pupil
"""
for ipdiam, kpdiam in enumerate(kpdiam_t):
    for iodiam, kodiam in enumerate(kodiam_t):
        for ithick, kthick in enumerate(kthick_t):
            
            iIter = ithick+ iodiam*nthick + ipdiam*nthick*nodiam
            print(f'iIter: {iIter+1:04d}/{nIter:04d}')
            
            pdiam2 = pdiam2_t[ipdiam]
            odiam2 = odiam2_t[iodiam]
            thick2 = thick2_t[ithick]            
            
            pupil = xaosim.pupil.four_spider_mask(nPup, nPup, nPup/2, 
                                                  pdiam=pdiam2, odiam=odiam2,
                                                  beta=beta, thick=thick2, offset=offset,
                                                  spiders=kwd_spiders,between_pix=True)

            if kpdiam != 1.0:
                outer_pupil = coro.utils.uniform_disk(nPup, (nPup/2)*kpdiam, CtrBtwnPix=True)
                pupil = pupil*outer_pupil           
            
            fname = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam*100)):03d}_kodiam{int(np.round(kodiam*100)):03d}_kthick{int(np.round(kthick*100)):03d}.fits' 
            fpath = fdir / fname
            
            if do_fits:
                fits.writeto(fpath, pupil*1, overwrite=True)

#%%
"""
### display vlt-like pupil
"""
#pupil_sbr = xaosim.pupil.subaru(nPup, nPup, nPup/2, between_pix=True)
#pupil_diff = pupil*1 - pupil_sbr*1

# pl.figure(0)
# pl.clf()
# pl.subplot(131)
# pl.imshow(pupil)

# pl.subplot(132)
# pl.imshow(pupil_sbr)

# pl.subplot(133)
# pl.imshow(pupil_diff)

# pl.show()

# a = coro.utils.uniform_disk(nPup, (nPup/2)*kpdiam, CtrBtwnPix=True)

# pl.figure(0)
# pl.clf()
# pl.imshow(a*pupil)

