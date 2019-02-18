#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:47:55 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%%
import numpy as np
import pylab as pl

from pyzelda.utils import aperture
import vigan.ao as ao

#%%

seeing   = 0.8
L0       = 25
z        = [  0,   4, 16]
Cn2      = [ 55,  35, 10]
v        = [  8,  10, 15]
arg_v    = [  0, -45,  0]
mag      = 3.29
zenith   = 16
azimuth  = 71
spaf     = 0.5
img_wave = 1.593e-6
seed     = 12345

#%%

pupil_saxo = aperture.sphere_saxo_pupil()
dim_pup_saxo = np.shape(pupil_saxo)[0]


#%%
residual_turbulence_fourier = ao.residual_screen_sphere(seeing, L0, z, Cn2, v, arg_v, mag, zenith, azimuth, 
                                                        spat_filter=spaf, img_wave=img_wave, dim_pup=dim_pup_saxo,
                                                        n_screen=100, fit=True, servo=True, alias=True, noise=True,
                                                        diff_refr=True, psd_only=False, seed=seed)
residual_turbulence_fourier  = residual_turbulence_fourier[..., dim_pup_saxo:2*dim_pup_saxo, dim_pup_saxo:2*dim_pup_saxo]
residual_turbulence_fourier *= pupil_saxo

#%%
pl.figure(0)
pl.imshow(residual_turbulence_fourier[0], cmap = 'viridis')
pl.show()