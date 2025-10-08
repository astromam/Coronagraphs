#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 22:22:37 2025

@author: mndiaye
"""

"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

#%%
"""
### Parameters
"""
do_sav = True


#%%
"""
### File reading for pupil
"""
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/elt-andes/').resolve()
fname_ini = 'ELT_pupil_400.fits'
fpath_ini = fdir / fname_ini

"""
### File saving for symmetric pupil
"""
fname_sym = 'ELT_pupil_400_sym.fits'
fpath_sym = fdir / fname_sym

#%%
"""
### Read file
"""

pupil = fits.getdata(fpath_ini)


#%%
"""
### Generate symmetric pupil
"""
pupilsym = pupil*np.fliplr(pupil)
pupilsym *= np.flipud(pupilsym)

#%%
"""
### Save symmetric pupil
"""
fits.writeto(fpath_sym, pupilsym, overwrite=True)


#%%
"""
### Display pupil
"""
plt.figure(0, (8, 4.5))
plt.clf()
plt.subplot(121)
plt.imshow(pupil)
plt.title('ELT')
plt.subplot(122)
plt.imshow(pupilsym)
plt.title('ELT sym')

#%%
"""
### test on symmetric pupil
"""
plt.figure(1, (8, 4.5))
plt.clf()
plt.subplot(121)
plt.imshow(pupilsym-np.fliplr(pupilsym))
plt.title('test lr')
plt.subplot(122)
plt.imshow(pupilsym-np.flipud(pupilsym))
plt.title('test ud')
