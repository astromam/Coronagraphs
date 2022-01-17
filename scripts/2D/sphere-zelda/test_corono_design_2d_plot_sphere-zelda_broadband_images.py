#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:43:15 2021

@author: mndiaye
"""

"""
### Initialization
"""
import sys
import os
import pwd

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

#%%
"""
### Parameters
"""
user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

nlam = 1785
nImg2d = 50

do_plot = True

#%%
"""
### Directory
"""
# simulation case and directory
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('~/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs').expanduser()
        sim_case = 'test' # 'test' or 'server'
    elif syst == 'linux':
        fdir = Path('/scratch/{0}/data/Coronagraphs/'.format(user)).resolve()
        sim_case = 'server' # 'test' or 'server'            
    else:
        raise ValueError('Unknown operating system {0}'.format(user))
else:
    raise ValueError('Unknown user {0}'.format(user))

fdir_data = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/data/SPHERE/with_aberr/2018-04-03/sky/with_saxo/before_correction').expanduser()
fdir_res = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/plots').expanduser()


#%%
"""
### List of images
"""
newlist_dir = []
newlist_cor = []
list_img = os.listdir(fdir_data)
nb_img0 = len(list_img)

for i, item_str in enumerate(list_img):
    if item_str.startswith('dir_nmap00690') and item_str.endswith('_bandBB_H_nlam1785_img_f_F4_15dMSun_20Myr.fits'):
        newlist_dir.append(item_str)

for i, item_str in enumerate(list_img):
    if item_str.startswith('cor_nmap00690') and item_str.endswith('_bandBB_H_nlam1785_img_f_F4_15dMSun_20Myr.fits'):
        newlist_cor.append(item_str)

nb_dir = len(newlist_dir)
nb_cor = len(newlist_cor)

#%%
"""
### Sum of cubes
"""
cube_dir = np.zeros((nlam, nImg2d, nImg2d)) 
cube_cor = np.zeros((nlam, nImg2d, nImg2d)) 
for i in range(nb_dir):
    filepath = fdir_data / newlist_dir[0]
    cube_dir += fits.getdata(filepath)
    
for i in range(nb_cor):
    filepath = fdir_data / newlist_cor[0]
    cube_cor += fits.getdata(filepath)

#%%
"""
### Compute broadband image
"""
poly_dir = np.sum(cube_dir, axis=0)
poly_cor = np.sum(cube_cor, axis=0)

pk_poly_dir = 1./np.max(poly_dir)

poly_dir *= pk_poly_dir
poly_cor *= pk_poly_dir

#%%
"""
### Display broadband image
"""
fname_img_dir = 'direct_image.pdf'
fpath_img_dir = fdir_res / fname_img_dir

f0 = plt.figure(0, figsize=(8,4.5))
plt.clf()
ax0 = f0.add_subplot(111)
im = ax0.imshow(np.log10(poly_dir), cmap= 'inferno')
cbar_ax = f0.add_axes([0.8, 0.15, 0.05, 0.7])
cbar    = f0.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Intensity in log scale', rotation=270, labelpad = 16)
if do_plot is True:
    plt.savefig(str(fpath_img_dir), transparent=True)
plt.tight_layout()


fname_img_cor = 'corono_image.pdf'
fpath_img_cor = fdir_res / fname_img_cor

f1 = plt.figure(1, figsize=(8,4.5))
plt.clf()
ax0 = f1.add_subplot(111)
im = ax0.imshow(np.log10(poly_cor), cmap= 'inferno')
cbar_ax = f1.add_axes([0.8, 0.15, 0.05, 0.7])
cbar    = f1.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Intensity in log scale', rotation=270, labelpad = 16)
if do_plot is True:
    plt.savefig(str(fpath_img_cor), transparent=True)
plt.tight_layout()


plt.show()