#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:15:59 2019

@author: mndiaye
"""

import numpy as np
import pylab as pl
import corono as coro
from pyzelda.utils import zernike 

import pyzelda.zelda as zelda
#import time
#import os
#from pathlib import Path
#
#from astropy.io import fits





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
allLogToConsole = 1

MinIsland   = False
Binarity    = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1

#nPup = corono0.params['nPup']
nPup = 300
nFPM = 100
Fmax2d = 22.5
nImg2d = np.int(Fmax2d*20)

# mask radius in lam0/D units
rMask = 1.062/2

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  5.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 7.0

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True
ImPart = False 

#nlam
bw   = 0.1
nlam = 1

do_fits = True

lam0 = 1.65e-6
coeff = 0.02*lam0





#%%
"""
### Parameters
"""

kk= 10
iZern = 4


Pupil2d    = coro.utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
Pupil2d_big = coro.utils.uniform_disk(nPup*kk, nPup/2., CtrBtwnPix=CtrBtwnPix)

mask2d     = coro.utils.uniform_disk(nFPM, nFPM/2., CtrBtwnPix=CtrBtwnPix)

LyotStop2d = coro.utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
LyotStop2d_big = coro.utils.uniform_disk(nPup*kk, nPup/2.*kk, CtrBtwnPix=CtrBtwnPix)

opd_ini = zernike.zernike1(iZern, npix = nPup, outside = 0)
phi = 2.*np.pi*coeff*opd_ini/lam0
phi_big = np.asarray(np.pad(phi, np.int(nPup*((kk-1)/2)), mode='constant'))

phasor     = np.exp(1j*phi)
phasor_big = np.exp(1j*phi_big)


mB = 2*rMask
eps_R = np.exp(1j*np.pi)
eps_Z = np.exp(1j*np.pi/2)

val = 0
if nImg2d%2 == 0:
    val = 1/2        
xi2d     = (np.arange(nImg2d//2+1))* Fmax2d/nImg2d
xi2d_ctr = (np.arange(nImg2d//2)+val)* Fmax2d/nImg2d
 


#%%
"""
### Compute direct field
"""
fld_A0 = Pupil2d*1
fld_L0 = fld_A0*LyotStop2d
fld_D0 = coro.utils.sft(fld_L0, nImg2d, Fmax2d, CtrBtwnPix=CtrBtwnPix2)

int_D0tmp = np.abs(fld_D0)**2
int_D0 = int_D0tmp/np.max(int_D0tmp)


#%%
"""
### Compute coronagraphic field
"""
fld_A = Pupil2d*1
fld_B = mask2d*coro.utils.sft(fld_A, nFPM, mB, CtrBtwnPix=CtrBtwnPix)

fld_C = fld_A - (1.-eps_R)\
                *coro.utils.isft(fld_B, nPup, mB, CtrBtwnPix=CtrBtwnPix)
                
fld_L = fld_C*LyotStop2d
fld_D = coro.utils.sft(fld_L, nImg2d, Fmax2d, CtrBtwnPix=CtrBtwnPix2)

int_Dtmp = np.abs(fld_D)**2
int_D    = int_Dtmp/np.max(int_D0tmp)

int_C = np.abs(fld_C)**2




#%%
"""
### Compute coronagraphic field with an aberration
"""
fld_Awfe = Pupil2d*phasor
fld_Bwfe = mask2d*coro.utils.sft(fld_Awfe, nFPM, mB, CtrBtwnPix=CtrBtwnPix)

fld_Cwfe = fld_Awfe - (1.-eps_R)\
                *coro.utils.isft(fld_Bwfe, nPup, mB, CtrBtwnPix=CtrBtwnPix)
                
fld_Lwfe = fld_Cwfe*LyotStop2d

int_C = np.abs(fld_C)**2



#%%
"""
### Oversized field in the Lyot plane
"""
fld_AA = Pupil2d_big*1


fld_BB = mask2d*coro.utils.sft(fld_AA, nFPM, mB*kk, CtrBtwnPix=CtrBtwnPix)


fld_CC = fld_AA - (1.-eps_R)\
               *coro.utils.isft(fld_BB, nPup*kk, mB*kk, CtrBtwnPix=CtrBtwnPix)
fld_LL = fld_CC*LyotStop2d_big
fld_DD = coro.utils.sft(fld_LL, nImg2d, Fmax2d*kk, CtrBtwnPix=CtrBtwnPix2)

int_DDtmp = np.abs(fld_DD)**2
int_DD    = int_DDtmp/np.max(int_D0tmp)

int_CC = np.abs(fld_CC)**2
int_LL = np.abs(fld_LL)**2

fld_YY = mask2d*coro.utils.sft(fld_LL, nFPM, mB*kk, CtrBtwnPix=CtrBtwnPix)
fld_ZZ_tmp = fld_LL - (1.-eps_Z)\
               *coro.utils.isft(fld_YY, nPup*kk, mB*kk, CtrBtwnPix=CtrBtwnPix)
               
int_ZZ_tmp = np.abs(fld_ZZ_tmp)**2
               
int_ZZ = int_ZZ_tmp[np.int(nPup*((kk-1)/2)):np.int(nPup*((kk-1)/2))+nPup,
                      np.int(nPup*((kk-1)/2)):np.int(nPup*((kk-1)/2))+nPup]*\
                      LyotStop2d

             

#%%
"""
### Introduction of a Zernike mode
"""


fld_AAwfe = Pupil2d_big*phasor_big
fld_BBwfe = mask2d*coro.utils.sft(fld_AAwfe, nFPM, mB*kk, CtrBtwnPix=CtrBtwnPix)

fld_CCwfe = fld_AAwfe - (1.-eps_R)\
               *coro.utils.isft(fld_BBwfe, nPup*kk, mB*kk, CtrBtwnPix=CtrBtwnPix)
fld_LLwfe = fld_CCwfe*LyotStop2d_big

int_CCwfe = np.abs(fld_CCwfe)**2
int_LLwfe = np.abs(fld_LLwfe)**2

fld_YYwfe = mask2d*coro.utils.sft(fld_LLwfe, nFPM, mB*kk, CtrBtwnPix=CtrBtwnPix)
fld_ZZwfe_tmp = fld_LLwfe - (1.-eps_Z)\
               *coro.utils.isft(fld_YYwfe, nPup*kk, mB*kk, CtrBtwnPix=CtrBtwnPix)

int_ZZwfe_tmp = np.abs(fld_ZZwfe_tmp)**2  

int_ZZwfe = int_ZZwfe_tmp[np.int(nPup*((kk-1)/2)):np.int(nPup*((kk-1)/2))+nPup,
                            np.int(nPup*((kk-1)/2)):np.int(nPup*((kk-1)/2))+nPup]*\
                            LyotStop2d

#%%
"""
### zelda analysis
"""
z = zelda.Sensor('MISTIGRI')


clear_pupil = [int_LLwfe[np.int(nPup*((kk-1)/2)):np.int(nPup*((kk-1)/2))+nPup,
                            np.int(nPup*((kk-1)/2)):np.int(nPup*((kk-1)/2))+nPup]]
zelda_pupil = [int_ZZwfe]

clear_pupil = np.asarray(clear_pupil)
zelda_pupil = np.asarray(zelda_pupil)

wave = lam0


#%%
# OPD map extraction
opd_map = z.analyze(clear_pupil, zelda_pupil, wave, ratio_limit=20)



#%%
"""
### Entrance pupil
"""
#pl.figure(0)
#pl.clf()
#pl.imshow(Pupil2d)
#pl.title('Entrance pupil')
#pl.show()

"""
### Direct image display
"""
#pl.figure(1)
#pl.clf()
#pl.imshow(int_D0**0.25)
#pl.title('Direct image')
#pl.show()

"""
### Coronagraphic image display
"""
#pl.figure(2)
#pl.imshow(int_D**0.25)
#pl.title('R&R coronagraphic image')
#pl.show()

"""
### Display pupil intensity in the lyot plane
"""
#pl.figure(3)
#pl.clf()
#pl.imshow(np.abs(fld_L))
#pl.title('R&R re-imaged pupil plane')
#pl.show()
#
#pl.figure(13)
#pl.clf()
#pl.imshow(np.abs(fld_Lwfe))
#pl.title('R&R re-imaged pupil plane (with defocus)')
#pl.show()

#%%
"""
### Display pupil intensity in the lyot plane
"""
#pl.figure(4)
#pl.clf()
#pl.imshow(int_LL)
#pl.title('R&R re-imaged pupil plane (with oversized LS)')
#pl.show()

#pl.figure(5)
#pl.clf()
#pl.imshow(np.abs(int_DD)**0.25)
#pl.title('R&R coronagraphic image (with oversized LS)')
#pl.show()

#pl.figure(6)
#pl.clf()
#pl.imshow(phi)
#pl.title('Introduced Zernike mode')
#pl.show()


#pl.figure(14)
#pl.clf()
#pl.imshow(int_LLwfe)
#pl.title('R&R re-imaged pupil plane (with oversized LS and defocus)')
#pl.show()


pl.figure(7)
pl.clf()
pl.imshow(int_ZZ)
pl.title('Z re-imaged pupil plane (with oversized LS)')
pl.show()

pl.figure(17)
pl.clf()
pl.imshow(int_ZZwfe)
pl.title('Z re-imaged pupil plane (with oversized LS and defocus)')
pl.show()

pl.figure(18)
pl.clf()
pl.imshow(int_ZZ-int_ZZwfe)
pl.title('diff Z re-imaged pupil plane (with oversized LS and defocus)')
pl.show()


#%%
pl.figure(19)
pl.clf()
pl.imshow(opd_ini)
pl.title('introduced map')
pl.show()

pl.figure(20)
pl.clf()
pl.imshow(-opd_map[0])
pl.title('reconstructed map')
pl.show()


#%%
"""
### Radial intensity profiles of the images
"""
if nImg2d%2 == 0:
    xi2d = xi2d_ctr*1

pl.figure(10)
pl.clf()
pl.title('Radial intensity profiles of the images')
pl.semilogy(xi2d,int_D0[nImg2d//2,nImg2d//2:],label='w/o corono')
pl.semilogy(xi2d,int_D[nImg2d//2,nImg2d//2:],label='w/  corono')
pl.semilogy(xi2d,int_DD[nImg2d//2,nImg2d//2:],label='w/  corono (oversized LS)')
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-9, 2e0)
pl.legend()
pl.tight_layout()
#pl.savefig(str(fpath))