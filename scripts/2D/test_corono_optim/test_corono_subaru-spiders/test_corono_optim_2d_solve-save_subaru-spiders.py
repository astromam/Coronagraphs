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

#%% parameters
"""
Parameters
"""
# Telescope name
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxTau' # 'MaxTau' # ,'MaxContrastLinf' # 'MaxContrastL1' #
solver       = 'gurobipy' #,'stdgrb' #  'gurobipy', 'scipy.linprog'
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
rho0 =  4.0
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
nlam = 3

do_fits = True

#%%
"""
### Spectral parameters
"""
if band == 'HSC_z':
    wv0   = 8925.96e-10
    width = 792.99e-10
elif band == 'GPI_Y':
    wv0   = 10433.59e-10
    width = 1889.08e-10
elif band == 'GPI_J':
    wv0   = 12317.58e-10
    width = 2273.20e-10
elif band == 'GPI_H':
    wv0   = 16444.09e-10
    width = 2984.82e-10           
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


#%%
"""
File reading for Pupil and Lyot stop
"""
#fdir = Path('../../data/2D/pupils/').resolve()
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').resolve()
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

params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
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
"""
Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_2d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
t0 = time.time()
Apod1 = problem1.solve_model()
t1 = time.time()
print('optimization time             : {0:.2f}s'.format(t1-t0))

#%% Display of the apodizer
"""
Generation of full apodizer for quarter pupil optimization
"""
Apod1_2d = np.reshape(Apod1, (corono0.nPup, corono0.nPup))

if Pupil2dSym == True:
        Apod1_2dtmp =  Apod1_2d[corono0.nPup//2:, corono0.nPup//2:]
        Apod1_2d[:corono0.nPup//2, corono0.nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
        Apod1_2d[:, :corono0.nPup//2]          = np.flip(Apod1_2d[:, corono0.nPup//2:], axis=1)
        
#%%
"""
Save apodizer
"""
#fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').resolve()
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').expanduser()
        fdir_sav = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
        sim_case = 'test' # 'test' or 'server'
    elif syst == 'linux':
        fdir = Path('/scratch/mndiaye/data/Coronagraphs/data/2D/pupils/').resolve()
        fdir_sav = Path('/scratch/mndiaye/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
        sim_case = 'server' # 'test' or 'server'            
    else:
        raise ValueError('Unknown operating system {0}'.format(user))
elif user == 'ndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').expanduser()
        fdir_sav = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
        sim_case = 'test' # 'test' or 'server'
    elif syst == 'linux':
        fdir = Path('/home/ndiaye/scratch/data/Coronagraphs/data/2D/pupils/').resolve()
        fdir_sav = Path('/home/ndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
        sim_case = 'server' # 'test' or 'server'            
    else:
        raise ValueError('Unknown operating system {0}'.format(user))    
else:
    raise ValueError('Unknown user {0}'.format(user))


if not os.path.exists(fdir):
    os.makedirs(fdir)
    
fname = problem1.get_filename() + f'_{band}band.fits'
fpath = fdir / fname

if do_fits is True:
    fits.writeto(fpath, Apod1_2d, overwrite=True)
    
    