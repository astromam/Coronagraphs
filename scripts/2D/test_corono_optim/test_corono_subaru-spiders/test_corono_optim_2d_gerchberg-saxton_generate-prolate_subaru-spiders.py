#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:52:23 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

import numpy as np
import os
from pathlib import Path

import corono as coro

from astropy.io import fits

import pwd
import sys

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

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
nPup = 1200
nFPM = 50
Fmax2d = 45#22.5
nImg2d = 90#45

# telescope parameters
pdiam, odiam = 7.92, 2.3  # tel. and obst. diameters (meters)
thick = 0.25              # adopted spider thickness (meters)
offset = 1.278            # spider intersection offset (meters)
beta = 51.75              # spider angle beta

Fratio    = 64

kpdiam = pdiam/7.92
kodiam = odiam/2.3
kthick = thick/0.25

# Focal plane mask 
mas2rad   = np.pi/(180.*3600*1000) # Conversion factor from mas to rads
rad2mas   = 1/mas2rad

# mask radius in lam0/D units
rMask_m = 453e-6/2 

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  5.0
rho1 = 20.0

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
band = 'GPI_Y'
#bw   = 0.1
nlam = 1


do_fits = True

#%%
"""
### Spectral parameters
"""
wv0_z   = 8925.96e-10
width_z = 792.99e-10
bw_z    = width_z/wv0_z
rMask_z = rMask_m/(wv0_z*Fratio) 

wv0_Y   = 10433.59e-10
width_Y = 1889.08e-10
bw_Y    = width_Y/wv0_Y
rMask_Y = rMask_m/(wv0_Y*Fratio) 

wv0_J   = 12317.58e-10
wv1_J   = (1.72/1.65)*wv0_J
width_J = 2273.20e-10
bw_J    = width_J/wv0_J
rMask_J = rMask_m/(wv0_J*Fratio) 

wv0_H   = 16444.09e-10
wv1_H   = (1.72/1.65)*wv0_H
width_H = 2984.82e-10
bw_H    = width_H/wv0_H
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

rMask1min = 2.64#np.round(rMask_m/((wv0_H+width_H/2)*Fratio), decimals=2)
rMask1max = 2.64#np.round(rMask_m/((wv0_z-width_z/2)*Fratio), decimals=2)

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
    fname_pup = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam*100)):03d}_kodiam{int(np.round(kodiam*100)):03d}_kthick{int(np.round(kthick*100)):03d}.fits' 
    fname_lys = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam*100)):03d}_kodiam{int(np.round(kodiam*100)):03d}_kthick{int(np.round(kthick*100)):03d}.fits' 
else:
    raise NameError(f'{pupil_name}: unknown pupil name')

fpath_pup = fdir / fname_pup
fpath_lys = fdir / fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)

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
EE_C_t = np.zeros((nMask1))
nIt = 100

for iMask1, rMask1 in enumerate(rMask1_t):
    params1    = coro.update_params(params, rMask=rMask1, nlam=1) 
    if corono_name == 'APLC':
        corono1 = coro.design.APLC2d(**params1)
    else:
        raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
        
    Apod2d = Input*1
    EE_C = np.zeros((nIt))
    
    iIt = 0
    while iIt < nIt: 
        print(f'iteration number: {iIt:02d}')
        Psi_C = corono1.compute_corono_lyot_field_2d(Apod2d)
        Apod2d = np.abs((Apod2d - Psi_C[0])*Input)
        Apod2d /= Apod2d.max()
        EE_C[iIt] = np.sum(np.abs(Psi_C*Input)**2)
        if (iIt > 1) and (EE_C[iIt] >= EE_C[iIt-1]) and (EE_C[iIt-1] >= EE_C[iIt-2]):
            print(f'iteration: {iIt:02d}')
            EE_C_t[iMask1] = EE_C[iIt]
            break
        if iIt == nIt:
            print(f'iteration: {iIt:02d}')
            EE_C_t[iMask1] = EE_C[iIt]
            break
        iIt += 1 
            
    fname = f'pupilsbr_nPup{nPup}_pdiam{int(np.round(pdiam*100))}_odiam{int(np.round(odiam*100))}_thick{int(np.round(thick*100)):03d}_Apod_rMask{int(np.round(rMask1*100)):03d}.fits'
    fpath = fdir / fname
    
    if do_fits is True:
         fits.writeto(fpath, Apod2d, overwrite=True)
