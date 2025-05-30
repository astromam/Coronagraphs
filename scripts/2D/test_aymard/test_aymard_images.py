#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 14:41:47 2025

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
### filenames
"""
fdir = Path('/Users/mndiaye/Downloads/aymard').expanduser()

fname = 'zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED_I-zpl_science_p23_REDUCED_I.fits'
fpath = fdir / fname

#%%
"""
### read file
"""
image = fits.getdata(fpath)

#%%
"""
### analyse image
"""


plt.figure(0)
plt.clf()
plt.subplot(221)
plt.imshow(image[0])
plt.xlabel('label x')
plt.ylabel('label y')
plt.colorbar()
plt.subplot(222)
plt.imshow(image[1])
plt.xlabel('label x')
plt.subplot(223)
plt.imshow(image[0])
plt.xlabel('label x')
plt.ylabel('label y')
plt.subplot(224)
plt.imshow(image[1])
plt.xlabel('label x')