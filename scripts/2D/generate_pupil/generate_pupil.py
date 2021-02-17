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

#%%
"""
### Parameters
"""
nPup= 384
do_fits = True

#%%
"""
### generation of a VLT like pupil
"""
pupil = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0.)

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
fdir = Path('/Users/mndiaye/Dropbox/python/Coronagraphs/data/2D/pupils').resolve()
fname = 'pupil=vlt_nPup={0}.fits'.format(nPup) 
fpath = fdir / fname

if do_fits:
    fits.writeto(fpath, pupil, overwrite=True)
