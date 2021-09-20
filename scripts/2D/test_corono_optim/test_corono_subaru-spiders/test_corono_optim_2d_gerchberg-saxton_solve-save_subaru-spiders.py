#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:52:23 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

import numpy as np
import time
import os
from pathlib import Path

import corono as coro

from astropy.io import fits

import pwd
import sys

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

import matplotlib.pyplot as pl

#%% parameters
"""
Parameters
"""
# Telescope name
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxTau' # 'MaxTau' # ,'MaxContrastLinf' # 'MaxContrastL1' #
solver       = 'stdgrb' #,'stdgrb' #  'gurobipy', 'scipy.linprog'
slvLogToConsole = 1
slvCrossover    = 0
slvMethod       = 2
slvSparse       = 0
allLogToConsole = 1

MinIsland   = False
Binarity    = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1

#nPup = corono0.params['nPup']
nPup = 200
nFPM = 50
Fmax2d = 16#22.5
nImg2d = 4*Fmax2d#45

# telescope parameters
pdiam, odiam = 7.92, 2.3  # tel. and obst. diameters (meters)
thick = 0.25              # adopted spider thickness (meters)
offset = 1.278            # spider intersection offset (meters)
beta = 51.75              # spider angle beta

kpdiam_t = np.linspace(0.9, 1.0, 11)
kodiam_t = np.linspace(1.0, 2.0, 21)
kthick_t = np.linspace(1.0, 2.0, 11)

pdiam2_t = np.asarray(kpdiam_t)*pdiam
odiam2_t = np.asarray(kodiam_t)*odiam
thick2_t = np.asarray(kthick_t)*thick 

npdiam = len(kpdiam_t)
nodiam = len(kodiam_t)
nthick = len(kthick_t)

nIter = npdiam*nodiam*nthick

Fratio    = 64

# Focal plane mask 
mas2rad   = np.pi/(180.*3600*1000) # Conversion factor from mas to rads
rad2mas   = 1/mas2rad

# mask radius in lam0/D units
rMask_m = 453e-6/2 

#rMask = 2.8

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 5.0
rho1 = 7.0

# contrast in the dark region
cDarkHole = 7.0

# tau (integrated Pupil transmission)
tau   = 0.5

# CtrBtwnPix2
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True
ImPart = False 

#nlam
band = 'GPI_J'
#bw   = 0.1
nlam = 5
#nlam1 = 101

do_EE_D = False
do_EE_M = True
do_EE_S = True

do_fits = True
do_num_mask = True
do_apod_spiders = False

thick_apod = 0.
str_apod_spiders = '_apodnospiders'
if do_apod_spiders:
    thick_apod = thick*1
    str_apod_spiders = ''

#%%
"""
### Spectral parameters
"""
wv0_z   = 8925.96e-10
width_z = 200.0e-10
bw_z  = width_z/wv0_z
rMask_z = rMask_m/(wv0_z*Fratio) 

wv0_Y   = 10433.59e-10
width_Y = 1889.08e-10
bw_Y  = width_Y/wv0_Y
rMask_Y = rMask_m/(wv0_Y*Fratio) 

wv0_J   = 12317.58e-10
wv1_J   = (1.72/1.65)*wv0_J
width_J = 2273.20e-10
bw_J  = width_J/wv0_J
rMask_J = rMask_m/(wv0_J*Fratio) 

wv0_H   = 16444.09e-10
wv1_H   = (1.72/1.65)*wv0_H
width_H = 2984.82e-10
bw_H  = width_H/wv0_H
rMask_H = rMask_m/(wv0_H*Fratio)            

if band == 'HSC_z':
    wv0 = wv0_z
    wv1 = wv0*1
    width = width_z 
elif band == 'GPI_Y':
    wv0 = wv0_Y
    wv1 = wv0*1
    width = width_Y
elif band == 'GPI_J':
    wv0 = wv0_J
    wv1 = wv1_J
    width = width_J
elif band == 'GPI_H':
    wv0 = wv0_H
    wv1 = wv1_H
    width = width_H           
else:
    raise ValueError(f'Unknown {band} band')
    
# wavelength sampling
bw     = width/wv0 
lam0   = 1. 
dlam   = bw*lam0
lam_t  = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam)
wv_t   = wv0*lam_t

rMask     = rMask_m/(wv0*Fratio)  # mask size in lam0/D
rMask_mas = rMask * (wv0/pdiam)/mas2rad

# lam1_t  = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam1)
# wv1_t   = wv0*lam1_t
# rMask1_t = rMask_m/(wv1_t*Fratio)

rMask1min = np.round(rMask_m/((wv0_H+width_H/2)*Fratio), decimals=2)
rMask1max = np.round(rMask_m/((wv0_z-width_z/2)*Fratio), decimals=2)

nMask1 = int(np.round((rMask1max-rMask1min)*100))+1

rMask1_t = np.linspace(rMask1min, rMask1max, nMask1)

#%%
"""
File reading for Pupil and Lyot stop
"""
#fdir = Path('../../data/2D/pupils/').resolve()
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()
    elif syst == 'linux':
        fdir = Path('/scratch/mndiaye/data/Coronagraphs/data/2D/pupils/').resolve()
    else:
        raise ValueError('Unknown operating system {0}'.format(syst))
else:
    raise ValueError('Unknown user {0}'.format(user))

if pupil_name == 'sbr':
    fname_pup = f'pupil=sbr_nPup={nPup}_odiam={int(odiam*100)}_thick={int(thick*100):03d}.fits'
    fname_lys = f'pupil=sbr_nPup={nPup}_odiam={int(odiam*100)}_thick={int(thick*100):03d}.fits'
else:
    raise NameError(f'{pupil_name}: unknown pupil name')

fpath_pup = fdir / fname_pup
fpath_lys = fdir / fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)


if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

pup_ratio = odiam/pdiam

Pupil2dnospiders = coro.utils.uniform_disk(nPup, nPup//2,CtrBtwnPix=True) - \
    coro.utils.uniform_disk(nPup, pup_ratio*nPup//2,CtrBtwnPix=True)

#Input = Pupil2dnospiders*1
Input = Pupil2d*1


params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
                 Pupil2d = Input, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name,
                 slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 slvSparse = slvSparse,
                 allLogToConsole = allLogToConsole,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg,
                 ImPart = ImPart)

#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
    

#%%
"""
### Computation of a numerical mask to compute contrast outside the spiders diffraction pattern
"""

num_circ = coro.utils.uniform_disk(nImg2d, rMask_J*nImg2d/Fmax2d)
num_mask = np.ones((nImg2d, nImg2d))
str_num_mask = ''
if do_num_mask:
    val = 0
    if nImg2d %2 == 0:
        val = 1/2
    
    xx1, yy1  = np.meshgrid(np.arange(nImg2d)-nImg2d//2, np.arange(nImg2d)-nImg2d//2)
    
    beta2 = (beta+90)*np.pi/180
    beta3 = (-beta+90)*np.pi/180
    xx2 = -np.sin(beta2)*xx1 + np.cos(beta2)*yy1
    xx3 = -np.sin(beta3)*xx1 + np.cos(beta3)*yy1
    
    thick_diff = 6.4#32#(pdiam/thick2)*nImg2d/Fmax2d  
    
    num_mask[(xx2 <= thick_diff)*(xx2 >= -thick_diff)] = 0
    num_mask[(xx3 <= thick_diff)*(xx3 >= -thick_diff)] = 0
    num_mask[num_circ == 1] = 0
    
    str_num_mask = '_nummask=1'
    

#%%
"""
### computation of a array of the area in the image plane in which contrast should be computed
"""

rr_D = coro.utils.radius_disk(nImg2d, nImg2d//2, CtrBtwnPix=True)
rr_D *= Fmax2d
ind_D = (rr_D <= rho1)*(rr_D >= rho0)*(num_mask == 1)
ind_M = (num_circ == 1)
ind_S = (num_mask == 0)*(num_circ == 0)

area_D = np.zeros((nImg2d, nImg2d))
area_D[ind_D] = 1. 

area_M = np.zeros((nImg2d, nImg2d))
area_M[ind_M] = 1.  

area_S = np.zeros((nImg2d, nImg2d))
area_S[ind_S] = 1.  

pl.figure(70)
pl.clf()
pl.subplot(151)
pl.imshow(num_mask, cmap='inferno')
pl.subplot(152)
pl.imshow(area_D, cmap='inferno')
pl.subplot(153)
pl.imshow(rr_D, cmap='inferno')
pl.subplot(154)
pl.imshow(area_M, cmap='inferno')
pl.subplot(155)
pl.imshow(area_S, cmap='inferno')

#%%
if do_EE_D:
    EE_D_t = np.zeros((nIter, nMask1))
if do_EE_M:
    EE_M_t = np.zeros((nIter, nMask1))
if do_EE_S:
    EE_S_t = np.zeros((nIter, nMask1))

for ipdiam, kpdiam in enumerate(kpdiam_t):
    for iodiam, kodiam in enumerate(kodiam_t):
        for ithick, kthick in enumerate(kthick_t):
            
            iIter = ithick+ iodiam*nthick + ipdiam*nthick*nodiam
            print(f'iIter: {iIter+1:04d}/{nIter:04d}')
            
            pdiam2 = pdiam2_t[ipdiam]
            odiam2 = odiam2_t[iodiam]
            thick2 = thick2_t[ithick]
            
            fname_lys = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam*100)):03d}_kodiam{int(np.round(kodiam*100)):03d}_kthick{int(np.round(kthick*100)):03d}.fits' 
            fpath_lys = fdir / fname_lys
            LyotStop2d = fits.getdata(fpath_lys)

            params2    = coro.update_params(params, LyotStop2d = LyotStop2d) 
            if corono_name == 'APLC':
                corono2 = coro.design.APLC2d(**params2)
            else:
                raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
            
            for iMask1, rMask1 in enumerate(rMask1_t):
                
                fname_apo = f'pupilsbr_nPup{nPup}_pdiam{int(np.round(pdiam*100))}_odiam{int(np.round(odiam*100))}_thick{int(np.round(thick_apod*100)):03d}_Apod_rMask{int(np.round(rMask1*100)):03d}.fits'
                fpath_apo = fdir / fname_apo
                Apod2d = fits.getdata(fpath_apo)
                
                Int_D0 = corono2.compute_direct_intensity_2d(Apod2d, poly=True)
                Int_D  = corono2.compute_corono_intensity_2d(Apod2d, poly=True)
                Int_D /= Int_D0.max()
                if do_EE_D:
                    EE_D_t[iIter, iMask1] = np.mean(Int_D[ind_D]) 
                if do_EE_M:
                    EE_M_t[iIter, iMask1] = np.mean(Int_D[ind_M]) 
                if do_EE_S:
                    EE_S_t[iIter, iMask1] = np.mean(Int_D[ind_S])
    

#%%
"""
### EE vs rMask
"""    
fname_EE_D = f'pupilsbr_nPup{nPup}_EE_D_rho0{int(np.round(rho0*100)):03d}_rho1{int(np.round(rho1*100)):03d}' + str_num_mask + str_apod_spiders +'.fits'
fpath_EE_D = fdir / fname_EE_D

if do_fits and do_EE_D:
    fits.writeto(fpath_EE_D, EE_D_t, overwrite=True)

fname_EE_M = f'pupilsbr_nPup{nPup}_EE_M_rho0{int(np.round(rho0*100)):03d}_rho1{int(np.round(rho1*100)):03d}' + str_num_mask + str_apod_spiders +'.fits'
fpath_EE_M = fdir / fname_EE_M
       
if do_fits and do_EE_M:
    fits.writeto(fpath_EE_M, EE_M_t, overwrite=True)
    
fname_EE_S = f'pupilsbr_nPup{nPup}_EE_S_rho0{int(np.round(rho0*100)):03d}_rho1{int(np.round(rho1*100)):03d}' + str_num_mask + str_apod_spiders +'.fits'
fpath_EE_S = fdir / fname_EE_S
       
if do_fits and do_EE_S:
    fits.writeto(fpath_EE_S, EE_S_t, overwrite=True)

    