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

import os
import pwd
import sys

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform


#%%
"""
### Parameters
"""
nPup= 200
do_fits = True

pdiam, odiam = 7.92, 2.3  # tel. and obst. diameters (meters)
thick = 0.25              # adopted spider thickness (meters)
offset = 1.278            # spider intersection offset (meters)
beta = 51.75              # spider angle beta

odiam2 = 1.5*odiam
thick2 = 1.5*thick



#%%
"""
### generation of a VLT like pupil
"""
#pupil = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0.)
pupil_sbr = xaosim.pupil.subaru(nPup, nPup, nPup/2, between_pix=True)


pupil = xaosim.pupil.four_spider_mask(nPup, nPup, nPup/2, pdiam, odiam=odiam2,
                        beta=beta, thick=thick2, offset=offset,
                        spiders=True,
                        between_pix=True)

pupil_diff = pupil*1 - pupil_sbr*1

#%%
"""
### display vlt-like pupil
"""
pl.figure(0)
pl.clf()
pl.subplot(131)
pl.imshow(pupil)

pl.subplot(132)
pl.imshow(pupil_sbr)

pl.subplot(133)
pl.imshow(pupil_diff)

pl.show()

#%%
"""
save pupil
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



fname = f'pupil=sbr_nPup={nPup}_odiam={int(odiam2*100)}_thick={int(thick2*100):03d}.fits' 
fpath = fdir / fname

if do_fits:
    fits.writeto(fpath, pupil*1, overwrite=True)
