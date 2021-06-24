#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:17:29 2021

@author: mndiaye
"""


import numpy as np
from astropy.io import fits
from pathlib import Path
import os


#%%

fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/data/SPHERE/with_aberr/2018-04-03/sky/with_saxo/before_correction/').expanduser()


list_files0 = os.listdir(fdir)
nb_files0 = len(list_files0)

list_files = []
for i in range(nb_files0):
    if '_noise' in list_files0[i] and 'BB_H' in list_files0[i]:
        list_files.append(list_files0[i])

nb_files = len(list_files)

#%%
for i in range(nb_files):
    filepath = fdir / list_files[i]
    print(f'iteration {i+1:02d}/{nb_files:02d}')
    hdu = fits.open(filepath)
    datacube = np.rint(hdu[1].data).astype(int).clip(min=0)
    hdu_img  = fits.ImageHDU(datacube)
    hdu = fits.HDUList([hdu[0], hdu_img, hdu[2]])
    hdu.writeto(filepath, overwrite=True)