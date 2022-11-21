#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:45:29 2022

@author: mndiaye
"""

"""
### Initialization
"""
import numpy as np
from astropy.io import fits
from pathlib import Path 

import matplotlib.pyplot as plt

#%%
"""
### Directory
"""

fdir = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/Coronagraphs/results/2D/data/SPHERE/with_aberr/2018-04-03/sky/with_saxo/before_correction/res_rescale_maps2').resolve()

# fname1 = 'dir_nmap00690_i00000_f00689_bandBB_J_nlam1928_img_f_F4_15dMSun_20Myr.fits'
# fname2 = 'dir_nmap00690_i00690_f01379_bandBB_J_nlam1928_img_f_F4_15dMSun_20Myr.fits'

fname1 = 'dir_nmap00690_i00000_f00689_bandBB_H_nlam0357_img_f_F4_15dMSun_20Myr.fits'
fname2 = 'dir_nmap00690_i10350_f11039_bandBB_H_nlam0357_img_f_F4_15dMSun_20Myr.fits'

#fname2 = 'cor_nmap00690_i06210_f06899_bandBB_J_nlam1928_img_f_F4_15dMSun_20Myr.fits'

# fname1 = 'dir_nmap00690_i00000_f00689_bandBB_H_nlam1785_img_f_F4_15dMSun_20Myr.fits'
# fname2 = 'dir_nmap00690_i00690_f01379_bandBB_H_nlam1785_img_f_F4_15dMSun_20Myr.fits'

#fname1 = 'cor_nmap00690_i00000_f00689_bandBB_H_nlam1785_img_f_F4_15dMSun_20Myr.fits'
#fname2 = 'cor_nmap00690_i00690_f01379_bandBB_H_nlam1785_img_f_F4_15dMSun_20Myr.fits'


fpath1 = fdir / fname1
fpath2 = fdir / fname2

#%%
"""
### Read files
"""
cube1 = fits.getdata(fpath1)
cube2 = fits.getdata(fpath2)

cube_diff = cube2 - cube1 

print(f'Max difference: {np.max(cube_diff):.2f}')

#%%


plt.figure(0)
plt.clf()
plt.subplot(131)
plt.imshow(np.log10(cube1[0]))
plt.title('Cube 1')
plt.subplot(132)
plt.imshow(np.log10(cube2[0]))
plt.title('Cube 2')
plt.subplot(133)
plt.imshow(cube_diff[0])
plt.title('Difference')
