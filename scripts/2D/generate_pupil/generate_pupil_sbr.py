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

import matplotlib.pyplot as plt
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
nPup= 2048
do_fits = False

do_margin = True

str_margin=''
if do_margin:
    str_margin = '_v3'

pdiam, odiam = 7.92, 2.3 # tel. and obst. diameters (meters)
pdiam_apo, odiam_apo, thick_apo = 7.92*0.975, 2.3+7.92*0.01, 0.25 # tel. and obst. diameters (meters) - apodizer pupil
pdiam_lys, odiam_lys, thick_lys = 7.92*0.975*0.96, 2.6457, 0.50# tel. and obst. diameters (meters) - Lyot stop pupil

#pdiam_bis, odiam_bis, thick_bis = 7.92, 2.3, 0.25
if do_margin:
#    pdiam_bis, odiam_bis, thick_bis = pdiam_apo, odiam_apo, thick_apo # tel. and obst. diameters (meters) - apodizer pupil
    pdiam_bis, odiam_bis, thick_bis = pdiam_lys, odiam_lys, thick_lys# tel. and obst. diameters (meters) - Lyot stop pupil
    
thick = 0.25              # adopted spider thickness (meters)
offset = 1.278            # spider intersection offset (meters)
beta = 51.75              # spider angle beta

kpdiam_t = [pdiam_bis/pdiam]#[0.96]#np.linspace(0.9, 1.0, 11) # np.linspace(0.98, 1.0, 3) #
kodiam_t = [odiam_bis/odiam]#[1.112]#np.linspace(1.0, 2.0, 21) # np.linspace(1.0, 1.2, 5)  #
kthick_t = [thick_bis/thick]#[2.0]#np.linspace(1.0, 2.0, 11) # np.linspace(1.0, 1.2, 3)  #

nPup_ini = nPup*pdiam/pdiam_apo
nArr = int(np.ceil(nPup_ini))
if nArr:
    nArr += 1 

pdiam2_t = np.asarray(kpdiam_t)*pdiam
odiam2_t = np.asarray(kodiam_t)*odiam
thick2_t = np.asarray(kthick_t)*thick 

npdiam = len(kpdiam_t)
nodiam = len(kodiam_t)
nthick = len(kthick_t)

nIter = npdiam*nodiam*nthick

kwd_spiders = [False,True]
if thick <= 0.:
    kwd_spiders = False

#%%
"""
### Directories
"""
#fdir = Path('/Users/mndiaye/Dropbox/python/Coronagraphs/data/2D/pupils').resolve()
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').resolve()
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
            
            pupil_sbr_tmp = xaosim.pupil.four_spider_mask(nArr, nArr, nPup_ini/2, 
                                                  pdiam=pdiam, odiam=odiam,
                                                  beta=beta, thick=thick_bis, offset=offset,
                                                  spiders=kwd_spiders,between_pix=True)

            if do_margin:
                outer_pupil = coro.utils.uniform_disk(nArr, (nPup_ini/2)*pdiam_bis/pdiam, CtrBtwnPix=True) - coro.utils.uniform_disk(nArr, (nPup/2)*odiam_bis/pdiam, CtrBtwnPix=True)
                pupil_tmp = pupil_sbr_tmp*outer_pupil
                
            
            fname = f'pupilsbr_nPup{nPup:04d}_kpdiam{int(np.round(kpdiam*1000)):04d}_kodiam{int(np.round(kodiam*1000)):04d}_kthick{int(np.round(kthick*1000)):04d}{str_margin}.fits' 
            fpath = fdir / fname
            
            nIni = (nArr-nPup)//2
            nEnd = (nArr+nPup)//2
            pupil = pupil_tmp[nIni:nEnd, nIni:nEnd]
            
            if do_fits:
                fits.writeto(fpath, pupil*1, overwrite=True)

#%%
"""
### display vlt-like pupil
"""
pupil_sbr = xaosim.pupil.subaru(nArr, nArr, nPup_ini/2, between_pix=True)
pupil_diff = pupil_tmp*1 - pupil_sbr*1

plt.figure(0)
plt.clf()
plt.subplot(131)
plt.imshow(pupil_tmp)
plt.title('new pupil')


plt.subplot(132)
plt.imshow(pupil_sbr)
plt.title('subaru pupil')

plt.subplot(133)
plt.imshow(pupil_diff)
plt.title('difference')

plt.show()

a = coro.utils.uniform_disk(nPup, (nPup/2), CtrBtwnPix=True)

plt.figure(1)
plt.clf()
plt.subplot(111)
plt.imshow(pupil)
plt.title('Pupil - Apodizer')
#plt.subplot(122)
#plt.imshow(pupil_lys)
#plt.title('Pupil - Lyot stop')

