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
Fmax2d = 45#22.5
nImg2d = 90#45

# telescope parameters
pdiam, odiam = 7.92, 2.3  # tel. and obst. diameters (meters)
thick = 0.25              # adopted spider thickness (meters)
offset = 1.278            # spider intersection offset (meters)
beta = 51.75              # spider angle beta

fac = 2.0

odiam2 = fac*odiam
thick2 = fac*thick

Fratio    = 64

# Focal plane mask 
mas2rad   = np.pi/(180.*3600*1000) # Conversion factor from mas to rads
rad2mas   = 1/mas2rad

# mask radius in lam0/D units
rMask_m = 453e-6/2 

#rMask = 2.8

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
band = 'GPI_J'
#bw   = 0.1
nlam = 5
nlam1 = 101

do_fits = True

#%%
"""
### Spectral parameters
"""
wv0_z   = 8925.96e-10
width_z = 792.99e-10
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

lam1_t  = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam1)
wv1_t   = wv0*lam1_t
rMask1_t = rMask_m/(wv1_t*Fratio)

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

if pupil_name == 'lvr':
    fname_pup = f'ATLAST_Aperture_nPup={nPup}.fits'
    fname_lys = f'ATLAST_LyotStop_nPup={nPup}.fits'
elif pupil_name == 'sbr':
    fname_pup = f'pupil=sbr_nPup={nPup}_odiam={int(odiam*100)}_thick={int(thick*100):03d}.fits'
    fname_lys = f'pupil=sbr_nPup={nPup}_odiam={int(odiam2*100)}_thick={int(thick2*100):03d}.fits'
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
if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
    
#%%
rr_D = coro.utils.radius_disk(nImg2d, nImg2d//2, CtrBtwnPix=True)
rr_D *= Fmax2d
ind_D = (rr_D <= rho1)*(rr_D >= rho0)

area_D = np.zeros((nImg2d, nImg2d))
area_D[ind_D] = 1.  

#%%

Apod1_2d_t = np.zeros((nlam1, nPup, nPup))
EE_D_t = np.zeros((nlam1))

for ilam1 in range(nlam1):
    rMask1 = rMask1_t[ilam1]
    params1    = coro.update_params(params, rMask=rMask1, nlam=1) 
    if corono_name == 'SP':
        corono1 = coro.design.SP2d(**params1)
    elif corono_name == 'APLC':
        corono1 = coro.design.APLC2d(**params1)
    else:
        raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
    
    
    nIt = 100
    Apod2d = np.zeros((nPup, nPup)) 
    Apod2d = Input*1
    EE_D = np.zeros((nIt))
    
    iIt = 0
    while iIt < nIt: 
        print(f'iteration number: {iIt:02d}')
        Psi_C = corono1.compute_corono_lyot_field_2d(Apod2d)
        Apod2d = np.abs((Apod2d - Psi_C[0])*Input)
        Apod2d /= Apod2d.max()
        Int_D0 = corono0.compute_direct_intensity_2d(Apod2d, poly=True)
        Int_D  = corono0.compute_corono_intensity_2d(Apod2d, poly=True)
        Int_D /= Int_D0.max()
        EE_D[iIt] = np.sum(area_D*Int_D**2)
        if (iIt > 1) and (EE_D[iIt] >= EE_D[iIt-1]) and (EE_D[iIt-1] >= EE_D[iIt-2]):
            print(f'iteration: {iIt:02d}')
            EE_D_t[ilam1] = EE_D[iIt]
            break
        if iIt == nIt:
            print(f'iteration: {iIt:02d}')
            EE_D_t[ilam1] = EE_D[iIt]
            break
        iIt += 1 
    
    
    Apod1_2d_t[ilam1] = Apod2d
   
    #%%
    """
    Problem defintion
    """
    if problem_name == 'MaxTau':
        # Maximization of the integrated amplitude transmission of the apodizer
        problem1 = coro.optim_2d.MaxTau(corono=corono1, **params1)
    elif problem_name == 'MaxContrastL1':
        # Maximization of the contrast under L1-norm
        problem1 = coro.optim_2d.MaxContrast(corono=corono1, Lnorm='L1',**params1)
    elif problem_name == 'MaxContrastLinf':
        # Maximization of the contrast under L-infinite norm
        problem1 = coro.optim_2d.MaxContrast(corono=corono1, Lnorm='Linf',**params1)
    else:
        raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))
    
    
    
    fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name
    if user == 'mndiaye':
        if syst == 'darwin':
            fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
        elif syst == 'linux':
            fdir = Path('/scratch/mndiaye/data/Coronagraphs/results/2D/dat_pyth/').resolve()
        else:
            raise ValueError('Unknown operating system {0}'.format(syst))
    else:
        raise ValueError('Unknown user {0}'.format(user))
    
    
    if not os.path.exists(fdir):
        os.makedirs(fdir)
        
    fname = problem1.get_filename() + f'_{band}band_gbsx.fits'
    fpath = fdir / fname
    
    if do_fits is True:
         fits.writeto(fpath, Apod2d, overwrite=True)

#%%
"""
### Display apodizer
"""
# pl.figure(0)
# pl.clf()
# pl.subplot(141)
# pl.imshow(Pupil2dnospiders)
# pl.subplot(142)
# pl.imshow(Pupil2d)
# pl.subplot(143)
# pl.imshow(LyotStop2d)
# pl.subplot(144)
# pl.imshow(Apod1_2d)

#%%
"""
### EE vs rMask
"""    
pl.figure(1)
pl.clf()
pl.plot(rMask1_t, EE_D_t)
    