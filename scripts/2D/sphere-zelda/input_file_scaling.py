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
saxomap_f    = 1379 # saxo last screen

# seeing for on-sky observations
seeing = 0.7

# test on the order of the min and max number of saxo phase screen
if saxomap_i <= saxomap_f:
    nsaxomap     = saxomap_f - saxomap_i + 1
else:
    raise NameError('initial saxo map (saxomap_i={0}) must be smaller than final saxo map (saxomap_f={1})!'.format(saxomap_i, saxomap_f))

do_sav = True
do_disp = True

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
# fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs').resolve()
# fdir_dat = fdir / 'data' / '2D' / 'medres_sim' 

### old directories
fdir_pupils = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
fdir_zelda  = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
fdir_saxo   = fdir / 'data' / '2D' / 'ZELDA' / '2018-04-03'


#%%
"""
### Filenames for the sources
"""
fname_Apod2d          = 'SPHERE_APO1_field_transmission_map.fits'
fname_Apod2d_OPDmapnm = 'apo_substrate_D1.fits'
fname_Ampmap2d        = '2018-04-03_night_sphere_pupil_clear_sky_FeII_field.fits'
fname_SAXOmapnm3d     = '2018-04-04T00-41-34-saxo_residual_turbulence_time30.0sec_seeing{:.1f}as_tiptilt1_gains0_fitting1_alias1.fits'.format(seeing)
fname_ZELDAmapnm3d    = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
fname_LyotStop2d      = 'sphere_stop_ST_ALC2.fits'

#%%
"""
### Filepaths for the file sources
"""
# new directory 
# fpath_Apod2d          = fdir_dat / fname_Apod2d
# fpath_Apod2d_OPDmapnm = fdir_dat / fname_Apod2d_OPDmapnm
# fpath_Ampmap2d        = fdir_dat / fname_Ampmap2d
# fpath_SAXOmapnm3d     = fdir_dat / fname_SAXOmapnm3d
# fpath_ZELDAmapnm3d    = fdir_dat / fname_ZELDAmapnm3d   
# fpath_LyotStop2d      = fdir_dat / fname_LyotStop2d    


# old directory
fpath_Apod2d          = fdir_pupils / fname_Apod2d
fpath_Apod2d_OPDmapnm = fdir_pupils / fname_Apod2d_OPDmapnm
fpath_Ampmap2d        = fdir_zelda / fname_Ampmap2d
fpath_ZELDAmapnm3d    = fdir_zelda / fname_ZELDAmapnm3d   
fpath_SAXOmapnm3d     = fdir_saxo / fname_SAXOmapnm3d
fpath_LyotStop2d      = fdir_pupils / fname_LyotStop2d    


#%%
"""
### Filenames for the sources to save
"""
fname2_Apod2d          = f'SPHERE_APO1_field_transmission_map_nPup{nPup:04d}.fits'
fname2_Apod2d_OPDmapnm = f'apo_substrate_D1_nPup{nPup:04d}.fits'
fname2_Ampmap2d        = f'2018-04-03_night_sphere_pupil_clear_sky_FeII_field._nPup{nPup:04d}.fits'
fname2_SAXOmapnm3d     = f'2018-04-04T00-41-34-saxo_residual_turbulence_time30.0sec_seeing{seeing:.1f}as_tiptilt1_gains0_fitting1_alias1_nPup{nPup:04d}.fits'
fname2_ZELDAmapnm3d    = f'2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd_nPup{nPup:04d}.fits'
fname2_LyotStop2d      = f'sphere_stop_ST_ALC2_nPup{nPup:04d}.fits'

#%%
"""
### Filepaths for the file sources
"""
# new directory
# fpath2_Apod2d          = fdir_dat / fname2_Apod2d
# fpath2_Apod2d_OPDmapnm = fdir_dat / fname2_Apod2d_OPDmapnm
# fpath2_Ampmap2d        = fdir_dat / fname2_Ampmap2d
# fpath2_SAXOmapnm3d     = fdir_dat / fname2_SAXOmapnm3d
# fpath2_ZELDAmapnm3d    = fdir_dat / fname2_ZELDAmapnm3d   
# fpath2_LyotStop2d      = fdir_dat / fname2_LyotStop2d  

# old directory
fpath2_Apod2d          = fdir_pupils / fname2_Apod2d
fpath2_Apod2d_OPDmapnm = fdir_pupils / fname2_Apod2d_OPDmapnm
fpath2_Ampmap2d        = fdir_zelda / fname2_Ampmap2d
fpath2_ZELDAmapnm3d    = fdir_zelda / fname2_ZELDAmapnm3d   
fpath2_SAXOmapnm3d     = fdir_saxo / fname2_SAXOmapnm3d
fpath2_LyotStop2d      = fdir_pupils / fname2_LyotStop2d    

#%% 
"""
### File reading
"""
### Pupil
Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)
    
### Apodization
Apod2d_tmp = fits.getdata(fpath_Apod2d)
Apod2d = imutils.scale(Apod2d_tmp, 0, new_dim=(nPup,nPup), method='interp')

### Apodization OPD map
Apod2d_OPDmapnm_tmp = fits.getdata(fpath_Apod2d_OPDmapnm)
Apod2d_OPDmapnm_tmp[np.isnan(Apod2d_OPDmapnm_tmp)] = 0
Apod2d_OPDmapnm = imutils.scale(Apod2d_OPDmapnm_tmp, 0, new_dim=(nPup,nPup), method='interp')

### Amplitude errors
Ampmap2d_tmp = fits.getdata(fpath_Ampmap2d)
Ampmap2d     = imutils.scale(Ampmap2d_tmp, 0, new_dim=(nPup,nPup), method='interp')

### Phase errors
ZELDAmapnm3d_tmp = fits.getdata(fpath_ZELDAmapnm3d)[imap0]
ZELDAmapnm3d     = imutils.scale(ZELDAmapnm3d_tmp, 0, new_dim=(nPup,nPup), method='interp')

### SAXO maps
SAXOmapnm3d = np.empty((nmap, nPup, nPup))
t0 = time.time()
SAXOmapnm3d_tmp = fits.getdata(fpath_SAXOmapnm3d)[:nmap,:,:]
for imap in range(nmap): 
    SAXOmapnm3d[imap] = imutils.scale(SAXOmapnm3d_tmp[imap], 0, new_dim=(nPup,nPup), method='interp')
t1 = time.time()
print(f'scaling time for nmap={nmap}: {t1-t0:.2f}')
        
### Lyot Stop
LyotStop2d_tmp = fits.getdata(fpath_LyotStop2d)
LyotStop2d     = imutils.scale(LyotStop2d_tmp, 0, new_dim=(nPup,nPup), method='interp')


#%%
"""
### Save input
"""

if do_sav:
    fits.writeto(fpath2_Apod2d, Apod2d, overwrite=True)
    fits.writeto(fpath2_Apod2d_OPDmapnm, Apod2d_OPDmapnm, overwrite=True)
    
    fits.writeto(fpath2_Ampmap2d, Ampmap2d, overwrite=True)
    fits.writeto(fpath2_ZELDAmapnm3d, ZELDAmapnm3d, overwrite=True)
    fits.writeto(fpath2_SAXOmapnm3d, SAXOmapnm3d, overwrite=True)
    fits.writeto(fpath2_LyotStop2d, LyotStop2d, overwrite=True)

#%%
"""
### plot display
"""

if do_disp:
    imap=0
    
    OPDmap2d0 = (beta_wfs*ZELDAmapnm3d+Apod2d_OPDmapnm)*1e-9
    OPDmap2d = OPDmap2d0 + SAXOmapnm3d[saxomap_i+imap]*1e-9
    
    vmin0 = -6
    vmax0 = 0    
    
    plt.figure(21, figsize=(10,6))
    plt.clf()
    plt.subplot(2,5,1)
    plt.imshow(Pupil2d)
    plt.title('VLT Pupil')
    plt.subplot(2,5,2)
    plt.imshow(Ampmap2d)
    plt.title('Amplitude errors')
    plt.subplot(2,5,3)
    plt.imshow(Apod2d)
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
    plt.imshow(LyotStop2d)
    plt.title('Lyot stop')
    #plt.subplot(2,5,8)
    #plt.imshow(Int_L**0.25)
    #plt.title(r'Int$^{0.25}$ after LS')
    #plt.subplot(2,5,10)
    #plt.imshow(np.log10(Int_D[0]), vmin=vmin0, vmax=vmax0)
    #plt.title(r'$\log$Int')


