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

#%%
"""
### Parameters
"""
pupil_name = 'vlt'
nPup= 250#384
do_fits = True

#%%
"""
### generation of a VLT like pupil
"""
if pupil_name == 'vlt':
    pupil = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0.)
elif pupil_name == 'sbr':
    pdiam, odiam = 7.92, 2.3
    thick = 0.25              # adopted spider thickness (meters)
    offset = 1.278            # spider intersection offset (meters)
    beta = 51.75              # spider angle beta

    pupil = xaosim.pupil.four_spider_mask(nPup, nPup, nPup/2, 
                                          pdiam=pdiam, odiam=odiam,
                                          beta=beta, thick=thick, offset=offset,
                                          spiders=True,between_pix=True)

#%%
"""
### display vlt-like pupil
"""
pl.figure(0)
pl.clf()
pl.imshow(pupil)

pl.show()

#%%
"""
save pupil
"""
fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils').resolve()
fname = f'pupil={pupil_name}_nPup={nPup}.fits'
fpath = fdir / fname

if do_fits:
    fits.writeto(fpath, pupil*1., overwrite=True)
