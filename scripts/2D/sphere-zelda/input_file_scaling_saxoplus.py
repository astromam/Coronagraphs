#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:20:57 2022

@author: mndiaye
"""

"""
### Initialization
"""
import sys
import numpy as np
import os
import pwd
from pyzelda.utils import aperture, imutils
from pathlib import Path

from astropy.io import fits

import matplotlib.pyplot as plt

import time

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

#%%
"""
### Parameters
"""
# spatial sampling
nPup   = 100   # pupil

# simulation configuration   
saxofudge    = 1. #80/120.
saxomap_i    = 0    # saxo first screen
saxomap_f    = 99 # saxo last screen (max phase screen 41403)

# number of sequences provided by Charles Goulas
nseq = 30
nexp = 2

# seeing for on-sky observations
seeing = 0.7

# test on the order of the min and max number of saxo phase screen
if saxomap_i <= saxomap_f:
    nsaxomap     = saxomap_f - saxomap_i + 1
else:
    raise NameError('initial saxo map (saxomap_i={0}) must be smaller than final saxo map (saxomap_f={1})!'.format(saxomap_i, saxomap_f))

do_sav = True
do_disp = False

#%%
"""
### Directories
"""    
str_aberr = 'with_aberr'    
str_date  = '2018-04-01'
str_obs   = 'internal'
str_corr  = 'before_correction'
str_saxo  = ''
imap0     = 0
nmap      = 1
beta_wfs  = 1./0.90
str_saxo_tmp  = 'wo_saxo'
str_date = '2018-04-03'
str_obs  = 'sky'
str_saxo = 'with_saxo'
nmap     = nsaxomap*1
beta_wfs = 1./0.6

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

### new directory
fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs').resolve()
fdir_dat = fdir / 'data' / '2D' / 'medres_sim' 

fdir_saxo   = fdir / 'data' / '2D' / 'medres_sim' / 'saxoplus_goulas'


fname_pupil = f'pupil.fits'
fpath_pupil = fdir_saxo / fname_pupil

Pupil2d = fits.getdata(fpath_pupil)
ind_Pupil2d = Pupil2d == 1


fname2_SAXOplusmapnm3d_all     = f'saxoplus_screen{nexp*nseq*nmap:05d}_nPup{nPup:04d}.fits'
fpath2_SAXOplusmapnm3d_all     = fdir_saxo / fname2_SAXOplusmapnm3d_all

SAXOplusmapnm3d_all = np.empty((nexp*nseq *nsaxomap, nPup, nPup))

for iexp in range(nexp):
    for iseq in range(nseq):
        
        #%%
        """
        ### Filenames for the sources
        """
        fname_SAXOplusmapnm3d     = f'exposure{iexp+1}_screen{iseq*saxomap_i}to{iseq*saxomap_i+saxomap_f}.fits'
        
        #%%
        """
        ### Filepaths for the file sources
        """
        # old directory
        fpath_SAXOplusmapnm3d     = fdir_saxo / fname_SAXOplusmapnm3d
        
        #%%
        """
        ### Filenames for the sources to save
        """
        fname2_SAXOplusmapnm3d     = f'exposure{iexp+1}_screen{iseq*saxomap_i:05d}to{iseq*saxomap_i+saxomap_f:05d}_nPup{nPup:04d}.fits'
        
        #%%
        """
        ### Filepaths for the file sources
        """
        # new directory
        # fpath2_SAXOmapnm3d     = fdir_dat / fname2_SAXOmapnm3d
        
        # old directory
        fpath2_SAXOplusmapnm3d     = fdir_saxo / fname2_SAXOplusmapnm3d
        
        #%% 
        """
        ### File reading
        """
        ### SAXO maps
        t0 = time.time()
        SAXOplusmapnm3d_tmp = fits.getdata(fpath_SAXOplusmapnm3d)[:nmap,:,:]
        #nmap = np.shape(SAXOmapnm3d_tmp)[0]
        SAXOplusmapnm3d = np.empty((nmap, nPup, nPup))
        for imap in range(nmap): 
            SAXOplusmapnm3d_tmp[imap, ind_Pupil2d] -= np.mean(SAXOplusmapnm3d_tmp[imap, ind_Pupil2d]) 
            SAXOplusmapnm3d[imap] = imutils.scale(SAXOplusmapnm3d_tmp[imap], 0, new_dim=(nPup,nPup), method='interp')
        t1 = time.time()
        print(f'scaling time for nmap={nmap}: {t1-t0:.2f}s')
        
        #%%
        """
        ### Save input
        """        
        if do_sav:
            fits.writeto(fpath2_SAXOplusmapnm3d, SAXOplusmapnm3d, overwrite=True)

        SAXOplusmapnm3d_all[iexp*nseq*nmap+iseq*nmap:iexp*nseq*nmap+(iseq+1)*nmap] = SAXOplusmapnm3d
        print(iexp*nseq*nmap+iseq*nmap)
        print(iexp*nseq*nmap+(iseq+1)*nmap)
        print('\n')
    

if do_sav:
    fits.writeto(fpath2_SAXOplusmapnm3d_all, SAXOplusmapnm3d_all, overwrite=True)

#%%
"""
### plot display
"""

if do_disp:
    imap=0
    
    OPDmap2d = SAXOplusmapnm3d[saxomap_i+imap]*1e-9
    
    vmin0 = -6
    vmax0 = 0    
    
    plt.figure(21, figsize=(10,6))
    plt.clf()
    plt.subplot(2,5,1)
#    plt.imshow(Pupil2d)
    plt.title('VLT Pupil')
    plt.subplot(2,5,2)
#    plt.imshow(Ampmap2d)
    plt.title('Amplitude errors')
    plt.subplot(2,5,3)
#    plt.imshow(Apod2d)
    plt.title('Apodizer')
    plt.subplot(2,5,4)
    plt.imshow(OPDmap2d)
    plt.title('OPD (Z/0.6 + SAXO[0])')
    #plt.subplot(2,5,5)
    #plt.imshow(np.log10(Int_D0[0]), vmin=vmin0, vmax=vmax0)
    #plt.title(r'$\log$Int$_0$')
    #plt.subplot(2,5,6)
    #plt.imshow(Int_C**0.25)
    #plt.title(r'Int$^{0.25}$ before LS')
    plt.subplot(2,5,7)
#    plt.imshow(LyotStop2d)
    plt.title('Lyot stop')
    #plt.subplot(2,5,8)
    #plt.imshow(Int_L**0.25)
    #plt.title(r'Int$^{0.25}$ after LS')
    #plt.subplot(2,5,10)
    #plt.imshow(np.log10(Int_D[0]), vmin=vmin0, vmax=vmax0)
    #plt.title(r'$\log$Int')


