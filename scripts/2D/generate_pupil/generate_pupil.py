#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:35:58 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

# Initialization
import numpy as np
#import pyzelda.utils.aperture as aperture
from vigan.optics import aperture

import pylab as pl
from pathlib import Path

from astropy.io import fits

import xaosim

#%%
"""
### Parameters
"""
pupil_name = 'vlt_btw'
do_dead_act = False
nPup= 400#384
do_fits = False


#%%
"""
### generation of a VLT like pupil
"""
x_deadact = [0.1534, -0.0984, -0.1963, 0.2766, 0.3297]
y_deadact = [-0.0768, -0.1240, -0.3542, -0.2799, -0.2799] 

if pupil_name == 'vlt':
    if do_dead_act:
        pupil = aperture.vlt_pupil(nPup, nPup, dead_actuators=[x_deadact,y_deadact],
        dead_actuator_diameter=0.025, cpix=False)
    else:
        pupil = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0.)
elif pupil_name == 'vlt_btw':
    pdiam, odiam = 8.00, 1.12
    thick = 0.04              # adopted spider thickness (meters)
    offset = 1.11            # spider intersection offset (meters)
    beta = 50.5              # spider angle beta

    pupil = xaosim.pupil.four_spider_mask(nPup, nPup, nPup/2, 
                                          pdiam=pdiam, odiam=odiam,
                                          beta=beta, thick=thick, offset=offset,
                                          spiders=True,between_pix=True)
    
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
str_dead_act=''
if do_dead_act and pupil_name == 'vlt':
    str_dead_act = '_dead_act'

fname = f'pupil={pupil_name}_nPup={nPup}{str_dead_act}.fits'
fpath = fdir / fname
print(fname)

if do_fits:
    fits.writeto(fpath, pupil*1., overwrite=True)
