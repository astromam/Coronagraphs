#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:30:40 2021

@author: mndiaye
"""

import numpy as np
import matplotlib.pyplot as pl
ftsz = 12 
pl.rcParams.update({'font.size': ftsz})
from pathlib import Path

import os
from matplotlib import cm
from astropy.io import fits
import corono as coro
from pyzelda.utils import imutils

import copy

import time

import pwd
import sys

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

#%% parameters
"""
Parameters
"""
pl.close('all')

test_gurobi = False
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxTau' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
    solver       = 'stdgrb' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    FirstDerGlobalLim = 1.
    
    #nPup = corono0.params['nPup']
    nPup = 1200
    nFPM = 50
    Fmax2d = 50
    nImg2d = 500

    # telescope parameters
    pdiam, odiam = 7.92, 2.3  # tel. and obst. diameters (meters)
    thick = 0.25              # adopted spider thickness (meters)
    offset = 1.278            # spider intersection offset (meters)
    beta = 51.75              # spider angle beta
    
    pdiam2 = 7.92
    odiam2 = 2.65#2.53#
    thick2 = 0.25
    Fratio = 64

    kpdiam1 = pdiam/pdiam
    kodiam1 = odiam/odiam
    kthick1 = thick/thick
    
    kpdiam2 = pdiam2/pdiam
    kodiam2 = odiam2/odiam
    kthick2 = thick2/thick
    

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
    Pupil2dSym  = True # set it True only for optimization
    
    #nlam
    band = 'GPI_J'
    # bw   = 0.1
    nlam = 1

    
    do_fits = False
    do_plot = True

nlambis = 11    
Fmax2dbis = 50
nImg2dbis = 500    

#%%
"""
### Spectral parameters
"""
wv0_z   = 8925.96e-10
width_z = 792.99e-10
bw_z  = width_z/wv0_z
rMask_z = rMask_m/(wv0_z*Fratio) 

wv0_Y   = 10433.59e-10
# if fac == 1.0:
#     wv1_Y = 1.09814232e-06
# elif fac == 1.5:
#     wv1_Y = 1.09814232e-06
# elif fac == 2.0:
#     wv1_Y = 1.090586e-06
# else:    
#     wv1_Y = (1.72/1.65)*wv0_Y
width_Y = 1889.08e-10
bw_Y  = width_Y/wv0_Y
rMask_Y = rMask_m/(wv0_Y*Fratio) 

wv0_J   = 12317.58e-10
# if fac == 1.0:
#     wv1_J = 1.2385776e-06
# elif fac == 1.5:
#     wv1_J = 1.2385776e-06
# elif fac == 2.0:
#     wv1_J = 1.2385776e-06
# else:
#     wv1_J   = (1.72/1.65)*wv0_J
width_J = 2273.20e-10
bw_J  = width_J/wv0_J
rMask_J = rMask_m/(wv0_J*Fratio) 

wv0_H   = 16444.09e-10
# if fac == 1.0:
#     wv1_H   = 1.61754562e-06
# elif fac == 1.5:
#     wv1_H   = 1.63843936e-06
# elif fac == 2.0:
#     wv1_H   = 1.6891813e-06
# else:
#     (1.72/1.65)*wv0_H
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
#    wv1 = wv1_J
    width = width_J
elif band == 'GPI_H':
    wv0 = wv0_H
#    wv1 = wv1_H
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

rMask1    = 2.72#rMask_m/(wv1*Fratio)  # mask size in lam1/D
wv1 = rMask_m/(rMask1*Fratio)

#%%
"""
File reading for Pupil and Lyot stop
"""
if True:
#    fdir = Path('../../data/2D/pupils/').resolve()
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
        fname_pup = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam1*100)):03d}_kodiam{int(np.round(kodiam1*100)):03d}_kthick{int(np.round(kthick1*100)):03d}.fits' 
#        fname_lys = f'pupil=sbr_nPup={nPup}_odiam={int(odiam2*100)}_thick={int(thick2*100):03d}.fits'
        fname_lys = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam2*100)):03d}_kodiam{int(np.round(kodiam2*100)):03d}_kthick{int(np.round(kthick2*100)):03d}.fits' 
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
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)


#%%
"""
Read files
"""
#fname_apod = f'pupilsbr_nPup{nPup}_pdiam{int(np.round(pdiam*100))}_odiam{int(np.round(odiam*100))}_thick{int(np.round(thick*100)):03d}_Apod_rMask{int(np.round(rMask1*100)):03d}.fits'
fname_apod = f'pupilsbr_nPup{nPup}_pdiam{int(np.round(pdiam*100))}_odiam{int(np.round(odiam*100))}_thick{int(np.round(thick*100)):03d}_Apod_rMask{int(np.round(rMask1*100)):03d}.fits'
fname_apod_EDA = f'pupilsbr_nPup{nPup}_pdiam{int(np.round(pdiam*100))}_odiam{int(np.round(odiam*100))}_thick{int(np.round(thick*100)):03d}_Apod_rMask{int(np.round(rMask1*100)):03d}_EDA.fits'

fpath_apod= fdir / fname_apod
fpath_apod_EDA = fdir / fname_apod_EDA

Apod1 = fits.getdata(fpath_apod,)


#%% Display of the apodizer
"""
Plot display of the apodizers
"""
# pl.figure(4)
# pl.clf()
# pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
# pl.title('Pupil transmission')

#fname = f'pupilsbr_nPup{nPup}_pdiam{int(np.round(pdiam*100))}_odiam{int(np.round(odiam*100))}_thick{int(np.round(thick*100)):03d}_Apod_rMask{int(np.round(rMask1*100)):03d}.pdf'
#fpath = fdir_pdf / fname

pl.figure(5, (12,4))
pl.clf()
pl.subplot(131)
pl.imshow(Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil')
pl.subplot(132)
pl.imshow(Apod1*Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 transmission \n Gerchberg-Saxton')
pl.subplot(133)
pl.imshow(LyotStop2d, cmap = cm.Greys_r)
pl.title('Lyot Stop')
#pl.savefig(str(fpath))

#%%
"""
### Error diffusion algorithm
"""
def EDA(array):
    
    array2 = copy.deepcopy(array)
    n0, n1 = np.shape(array)
    
    for j in range(n1):
        for i in range(n0):
            oldpix = array2[i,j]
            newpix = np.round(oldpix)
            array2[i,j] = newpix
            err = oldpix - newpix
            if (i <= n0-2):
                array2[i+1,j  ] += err*7/16
            if (i >= 1) and (j <= n1-2): 
                array2[i-1,j+1] += err*3/16
            if (j <= n1-2):
                array2[i  ,j+1] += err*5/16
            if (i <= n0-2) and (j <= n1-2):
                array2[i+1,j+1] += err*1/16

    return array2

#%%
"""
### EDA application on Apodizer
"""
t0 = time.time()
Apod2 = EDA(Apod1)
t1 = time.time()
print(f'EDA computation time: {t1-t0:.2f}s')

#%%
T_Apod1 = np.sum(np.abs(Apod1*Pupil2d))/np.sum(np.abs(Pupil2d))
T_Apod2 = np.sum(np.abs(Apod2*Pupil2d))/np.sum(np.abs(Pupil2d))

print(f'Apod1 thorughput: {T_Apod1*100:.1f}')
print(f'Apod2 thorughput: {T_Apod2*100:.1f}')


#%%
"""
### Display plots
"""
pl.figure(6, (12, 4))
pl.clf()
pl.subplot(131)
pl.imshow(Apod1*Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 transmission \n Gerchberg-Saxton')
pl.subplot(132)
pl.imshow(Apod2*Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 after EDA')
pl.subplot(133)
pl.imshow((Apod2-Apod1)*Pupil2d, cmap = cm.Greys_r)
pl.title('difference')


#%%
"""
### Save file
"""
fits.writeto(fpath_apod_EDA, Apod2)