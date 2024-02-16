#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:43:27 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyzelda.utils import imutils, zernike
from vigan.optics import aperture

ftsz = 16 
plt.rcParams.update({'font.size': ftsz})

import os
from matplotlib import cm
from astropy.io import fits
import corono as coro

#from scipy.misc import imresize
import cv2

#%% parameters
"""
Parameters
"""
plt.close('all')
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'vlt_btw2' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxContrastL1' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
    solver       = 'gurobipy' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    Binarity    = False
    FirstDerGlobalLim = 1.
    BinarityReg       = 0.1
    LSRobustness = True
    
    #nPup = corono0.params['nPup']
    nArr_th = 520
    nPup_th = 520
    nPup0_th= 500
    nFPM = 50
    Fmax2d = 22.5
    nImg2d = 45
    
    wv        = 1.593e-6
    dAper     = 8
    mas2rad   = np.pi/(180.*3600)
    rMask_m   = 287e-6/2.
    Fratio    = 40
    rMask0  = rMask_m/(wv*Fratio)
    rMask_mas = 1000.*rMask0 * (wv/dAper)/mas2rad
    print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))
    
    # use real data    
    use_real_data = True
    if use_real_data:
        nArr_re = 384 #520
        nPup_re = 384 #520
        nPup0_re = 384 #500
        
    
    # apodizer type
    ApodBinary = False
    
    # mask radius in lam0/D unit
    rMask_th = rMask0*(nPup_th/nPup0_th)
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  0.0
    rho1 = 20.0
    
    # contrast in the dark region
    cDarkHole = 6.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.756
    
    # CtrBtwnPix2
    CtrBtwnPix  = True
    CtrBtwnPix2 = False
    Pupil2dSym  = False # set it True only for optimization
    ImPart      = True
    
    if use_real_data:
        CtrBtwnPix  = False
        CtrBtwnPix2 = False
    
    #nlam
    bw   = 0.2
    nlam = 5
   
    do_fits = True

nlambis = 11 

nImg2dbis = 134 #256
Fmax2dbis0 = nImg2dbis/(2*(wv/950e-9))
  
Fmax2dbis_th = Fmax2dbis0*(nPup_th/nPup0_th)

do_plot = False    

# separation for contrast estimates
sepbis=3.0
septer=5.0

# maximum pixel shift along a given axis for apodizer 
pix_max = 10
nLinShift = 5


#%%
"""
### File reading for Pupil and Lyot stop
"""
if True:
#    fdir = Path('../../data/2D/pupils/').resolve()
    fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').resolve()
    
    fdir_apod_th = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/SPHERE/upgrade/').resolve()  

    fname_apod_th = f'apod_initial_{nArr_th:04d}.fits'
    if ApodBinary is True:
        fname_apod_th = f'apod_binary_{nArr_th:04d}.fits'
        

    if pupil_name == 'vlt_btw2':
        fname_pup = f'pupil=vlt_btw_nPup={nPup0_th}_nArr={nArr_th}.fits'
        fname_lys = f'sphere_stop_ST_ALC2_nPup={nPup0_th}_nArr={nArr_th}.fits'

    # if use_real_data is True:  
    fdir_apod_re = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/SPHERE/').resolve()  
    fname_apod_re = 'SPHERE_APO1_field_transmission_map.fits'
        # fname_lys = 'sphere_stop_ST_ALC2.fits'
        # fpath_lys = fdir_apod_re / fname_lys    
        # Pupil2d = aperture.vlt_pupil(nArr, nPup0, dead_actuator_diameter=0, cpix=True)
        # LyotStop2dtmp = fits.getdata(fpath_lys)
        
    # else:
    #     fpath_pup = fdir / fname_pup
    #     fpath_lys = fdir / fname_lys
    #     Pupil2d_th   = aperture.vlt_pupil(nArr, nPup0, dead_actuator_diameter=0, cpix=False) #fits.getdata(fpath_pup)
    #     LyotStop2d_th = fits.getdata(fpath_lys)


#%%
"""
### Working directories
"""
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth').resolve() / pupil_name

fdir_pdf = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/plots/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

#%%
"""
### Read the apodizer file
"""

fpath_apod_th = fdir_apod_th / fname_apod_th
Apod2d_th_int = fits.getdata(fpath_apod_th,)

fpath_apod_re = fdir_apod_re / fname_apod_re
Apod2d_re_amp = fits.getdata(fpath_apod_re,)

#%%
"""
### Display images
"""
f3 = plt.figure(0, (8, 4.5))
plt.clf()
ax1 = plt.subplot(121)
im = ax1.imshow(np.sqrt(Apod2d_th_int), vmin=0, vmax=1, cmap='inferno')
plt.title('Spec')

ax2 = plt.subplot(122)
im = plt.imshow(Apod2d_re_amp, vmin=0, vmax=1, cmap='inferno')
plt.title('IRDIS meas.')

f3.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                wspace=0.02, hspace=0.02)

f3.subplots_adjust(right=0.85)
cbar_ax = f3.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = f3.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Normalized amplitude', rotation=270, labelpad = 20)
plt.tight_layout()
plt.show()

#%%
"""
### Display images
"""
f3 = plt.figure(1, (8, 4.5))
plt.clf()
ax1 = plt.subplot(121)
im = ax1.imshow(np.abs(Apod2d_th_int)**2, vmin=0, vmax=1, cmap='inferno')
plt.title('Spec')

ax2 = plt.subplot(122)
im = plt.imshow(np.abs(Apod2d_re_amp)**2, vmin=0, vmax=1, cmap='inferno')
plt.title('IRDIS meas.')

f3.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                wspace=0.02, hspace=0.02)

f3.subplots_adjust(right=0.85)
cbar_ax = f3.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = f3.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Normalized intensity', rotation=270, labelpad = 20)
plt.tight_layout()
plt.show()



#%%
"""
### Radial cut of the amplitude profiles
"""
rad_th = (np.arange(nArr_th)-nArr_th//2)/(nPup0_th)
rad_re = (np.arange(nArr_re)-nArr_re//2+0.5)/(nPup0_re)

plt.figure(2, (8, 4.5))
plt.clf()
plt.plot(rad_th, Apod2d_th_int[nArr_th//2], label='th')
plt.plot(rad_re, Apod2d_re_amp[nArr_re//2], label='re')
plt.vlines(-0.5, -0., 1., ls='--', color='k')
plt.vlines(0.5, -0., 1., ls='--', color='k')
plt.hlines(0.095, -0.5, 0.5, ls=':', color='k')
plt.hlines(0.625, -0.5, 0.5, ls=':', color='k')
plt.hlines(1.0, -0.5, 0.5, ls=':', color='k')
plt.hlines(0.0, -0.5, 0.5, ls=':', color='k')
plt.xlabel('r in pupil diameter')
plt.ylabel('Normalized amplitude')
plt.legend()

#%%
"""
### Radial cut of the intensity profiles
"""
rad_th = (np.arange(nArr_th)-nArr_th//2)/(nPup0_th)
rad_re = (np.arange(nArr_re)-nArr_re//2+0.5)/(nPup0_re)

plt.figure(3, (8, 4.5))
plt.clf()
plt.plot(rad_th, np.abs(Apod2d_th_int[nArr_th//2])**2, label='th')
plt.plot(rad_re, np.abs(Apod2d_re_amp[nArr_re//2])**2, label='re')
plt.vlines(-0.5, -0., 1., ls='--', color='k')
plt.vlines(0.5, -0., 1., ls='--', color='k')
#plt.hlines(0.095, -0.5, 0.5, ls=':', color='k')
#plt.hlines(0.625, -0.5, 0.5, ls=':', color='k')
plt.hlines(1.0, -0.5, 0.5, ls=':', color='k')
plt.hlines(0.0, -0.5, 0.5, ls=':', color='k')
plt.xlabel('r in pupil diameter')
plt.ylabel('Normalized intensity')
plt.legend()
