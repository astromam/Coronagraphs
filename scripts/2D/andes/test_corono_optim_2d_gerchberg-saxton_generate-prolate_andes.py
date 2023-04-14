#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:08:30 2023

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
pupil_name   = 'elt' # elt # 'vlt' or 'sbr' or 'lvr'
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
nPup = 400 #200
nFPM = 50
Fmax2d = 45#22.5
nImg2d = 90#45
do_margin = False

str_margin=''
if do_margin:
    str_margin = '_v2'

# telescope parameters
pdiam, odiam = 7.92, 2.3 # tel. and obst. diameters (meters)
if do_margin:
    pdiam, odiam = 7.92*0.99, 2.3+7.92*0.01 # tel. and obst. diameters (meters)
thick = 0.                # adopted spider thickness (meters)
offset = 1.278            # spider intersection offset (meters)
beta = 51.75              # spider angle beta

Fratio    = 64

kpdiam0 = pdiam/(7.92)
kodiam0 = odiam/(2.3)
if do_margin:
    kpdiam0 = pdiam/(7.92*0.99)
    kodiam0 = odiam/(2.3+7.92*0.01)
kthick0 = thick/0.25

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
band = 'ELT_H'
#bw   = 0.1
nlam = 1


do_fits = True

#%%
"""
### Spectral parameters
"""
wv0_z   = 8925.96e-10
width_z = 200.0e-10
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

wv0_Hbis   = 16000.0e-10
wv1_Hbis   = (1.72/1.65)*wv0_Hbis
width_Hbis = 2984.82e-10
bw_Hbis    = width_Hbis/wv0_Hbis
rMask_Hbis = rMask_m/(wv0_Hbis*Fratio)            

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
elif band == 'ELT_H':
    wv0 = wv0_Hbis
    wv1 = wv1_Hbis
    width = width_Hbis               
else:
    raise ValueError(f'Unknown {band} band')
    
# wavelength sampling
bw     = width/wv0 
lam0   = 1. 
dlam   = bw*lam0
lam_t  = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam)
wv_t   = wv0*lam_t

rMask     = 4.0 # rMask_m/(wv0*Fratio)  # mask size in lam0/D
rMask_mas = rMask * (wv0/pdiam)/mas2rad

rMask1min = rMask*3./4. #2.64 #np.round(rMask_m/((wv0_H+width_H/2)*Fratio), decimals=2) #2.65
rMask1max = rMask*5./4. #2.64 #np.round(rMask_m/((wv0_z-width_z/2)*Fratio), decimals=2) #2.65

rMask1nb = 5

nMask1 = int(np.round((rMask1max-rMask1min)*rMask1nb))+1

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
    folder_tel = ''
    fname_pup = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam0*100)):03d}_kodiam{int(np.round(kodiam0*100)):03d}_kthick{int(np.round(kthick0*100)):03d}{str_margin}.fits' 
    fname_lys = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam0*100)):03d}_kodiam{int(np.round(kodiam0*100)):03d}_kthick{int(np.round(kthick0*100)):03d}{str_margin}.fits' 
elif pupil_name == 'tmt':
    folder_tel = 'tmt'
    fname_pup = f'TMT_Pupil_Amplitude_MACOS_Logical_With_Obscuration_nArr{nPup:04d}_nPup{nPup:04d}.fits'
    fname_lys = f'TMT_Pupil_Amplitude_MACOS_Logical_With_Obscuration_nArr{nPup:04d}_nPup{nPup:04d}.fits'
elif pupil_name == 'elt':
    folder_tel = 'elt'
    fname_pup = f'pupilelt_nPup{nPup:04d}.fits'
    fname_lys = f'pupilelt_nPup{nPup:04d}.fits'    
else:    
    raise NameError(f'{pupil_name}: unknown pupil name')

fpath_pup = fdir / folder_tel / fname_pup
fpath_lys = fdir / folder_tel / fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)

nPup = np.shape(Pupil2d)[0]

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

    if pupil_name == 'sbr':
        fname_apod = f'pupilsbr_nPup{nPup}_pdiam{int(np.round(pdiam*100))}_odiam{int(np.round(odiam*100))}_thick{int(np.round(thick*100)):03d}_Apod_rMask{int(np.round(rMask1*100)):03d}{str_margin}.fits'
    elif pupil_name == 'tmt':
        fname_apod = 'TMT_Pupil_Amplitude_MACOS_Logical_With_Obscuration_nArr1920_nPup1920_apod.fits'
    elif pupil_name == 'elt':
        fname_apod = f'pupilelt_nPup{nPup}_Apod_rMask{int(np.round(rMask1*100)):03d}{str_margin}.fits'
    else:    
        raise NameError(f'{pupil_name}: unknown pupil name')
            
    fpath_apod = fdir / folder_tel / fname_apod
    
    if do_fits is True:
         fits.writeto(fpath_apod, Apod2d, overwrite=True)
