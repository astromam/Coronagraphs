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

from pathlib import Path

from astropy.io import fits

#%%

seeing   = 0.85
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
pl.title('map 0')
pl.show()

#%%
pl.figure(1)
pl.imshow(residual_turbulence_fourier[1]-residual_turbulence_fourier[0], cmap = 'viridis')
pl.title('Difference between maps 0 and 1')
pl.show()

#%%
residual_turbulence_psd = ao.residual_screen_sphere(seeing, L0, z, Cn2, v, arg_v, mag, zenith, azimuth, 
                                                        spat_filter=spaf, img_wave=img_wave, dim_pup=dim_pup_saxo,
                                                        n_screen=100, fit=True, servo=True, alias=True, noise=True,
                                                        diff_refr=True, psd_only=True, seed=seed)

#%%
pl.figure(2)
pl.imshow(np.log10(residual_turbulence_psd), cmap = 'inferno')
pl.title('PSD')
pl.show()

#%%

fdir = Path('/Users/mndiaye/Desktop/').resolve()
fname = 'residual_turbulence_psd.fits'
fpath = fdir / fname 

fits.writeto(fpath, residual_turbulence_psd, overwrite=True)




