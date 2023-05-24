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

import matplotlib.pyplot as plt


#%% parameters
"""
Parameters
"""
# Telescope name
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'elt' # elt # 'vlt' or 'sbr' or 'lvr'

#nPup = corono0.params['nPup']
nPup = 400 #200
nFPM = 100
Fmax2d = 25#22.5
nImg2d = 400#45

# Focal plane mask 
mas2rad   = np.pi/(180.*3600*1000) # Conversion factor from mas to rads
rad2mas   = 1/mas2rad

# CtrBtwnPix2
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True
ImPart = False 

#bw   = 0.1
nlam = 1


do_fits = False

#%%

# rMask     = 1.061375/2. # pupille circulaire nPup = 300
rMask     = 1.2309889/2. # pupille ELT nPup = 400


rMask1min = rMask*1.00 #2.64 #np.round(rMask_m/((wv0_H+width_H/2)*Fratio), decimals=2) #2.65
rMask1max = rMask*1.00 #2.64 #np.round(rMask_m/((wv0_z-width_z/2)*Fratio), decimals=2) #2.65

nMask1 = 1

rMask1_t = np.linspace(rMask1min, rMask1max, nMask1)

#nMask1 = int(np.round((rMask1max-rMask1min)*rMask1nb))+1




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

if pupil_name == 'tmt':
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

#%%
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)
# Pupil2d    = coro.utils.uniform_disk(nPup, nPup/2., 
#                                  CtrBtwnPix=True)
# LyotStop2d = Pupil2d*1.


mask2d = coro.utils.uniform_disk(nFPM, nFPM/2., CtrBtwnPix=True)

nPup = np.shape(Pupil2d)[0]

Input = Pupil2d*1



#%%
"""
### Roddier & Roddier Coronagraph
"""
def compute_roddier_fldC(Apod2d, Pupil2d, mask2d, LyotStop2d, mB):
    
    fld_A       = Apod2d*Pupil2d
    fld_B       = mask2d*coro.utils.sft(fld_A, nFPM, mB, 
                                    CtrBtwnPix=True)
    
    fld_C       = fld_A - (1.- np.exp(-1j*np.pi))\
                    *coro.utils.isft(fld_B, nPup, mB, 
                                   CtrBtwnPix=True)        
    return fld_C

#%%
def compute_roddier_fldD(Apod2d, Pupil2d, mask2d, LyotStop2d, mB, mD, nImg2d):
    
    fld_A       = Apod2d*Pupil2d
    fld_B       = mask2d*coro.utils.sft(fld_A, nFPM, mB, 
                                    CtrBtwnPix=True)
    
    fld_C       = fld_A - (1.- np.exp(1j*np.pi))\
                    *coro.utils.isft(fld_B, nPup, mB, 
                                   CtrBtwnPix=True)
                    
    fld_L       = fld_C*LyotStop2d
    fld_Dtmp = coro.utils.sft(fld_L, nImg2d, mD, 
              CtrBtwnPix=False)
    return fld_Dtmp

#%%
def compute_nomask_fldD(Apod2d, Pupil2d, LyotStop2d, mD, nImg2d):
    
    fld_C       = Apod2d*Pupil2d
                    
    fld_L       = fld_C*LyotStop2d
    fld_Dtmp = coro.utils.sft(fld_L, nImg2d, mD, 
              CtrBtwnPix=False)
    return fld_Dtmp


#%%
EE_C_t = np.zeros((nMask1))
nIt = 10

for iMask1, rMask1 in enumerate(rMask1_t): 
        
    Apod2d = Input*1
    EE_C = np.zeros((nIt))
    
    iIt = 0
    while iIt < nIt: 
        print(f'iteration number: {iIt:02d}')
        mB = rMask1*2.
        mD = Fmax2d*1.
        Psi_C = compute_roddier_fldC(Apod2d, Pupil2d, mask2d, LyotStop2d, mB)
        Apod2d = np.abs((Apod2d - Psi_C)*Input)
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

    # if pupil_name == 'sbr':
    #     fname_apod = f'pupilsbr_nPup{nPup}_pdiam{int(np.round(pdiam*100))}_odiam{int(np.round(odiam*100))}_thick{int(np.round(thick*100)):03d}_Apod_rMask{int(np.round(rMask1*100)):03d}{str_margin}.fits'
    # elif pupil_name == 'tmt':
    #     fname_apod = 'TMT_Pupil_Amplitude_MACOS_Logical_With_Obscuration_nArr1920_nPup1920_apod.fits'
    # elif pupil_name == 'elt':
    #     fname_apod = f'pupilelt_nPup{nPup}_Apod_rMask{int(np.round(rMask1*100)):03d}{str_margin}.fits'
    # else:    
    #     raise NameError(f'{pupil_name}: unknown pupil name')
            
    # fpath_apod = fdir / folder_tel / fname_apod
    
    # if do_fits is True:
    #      fits.writeto(fpath_apod, Apod2d, overwrite=True)

#%%
Throughput = np.sum(np.abs(Apod2d)**2)/np.sum(np.abs(Pupil2d)**2)

print(f'T={Throughput*100:.1f}%')


         
#%%
PsiD = compute_roddier_fldD(Apod2d, Pupil2d, mask2d, LyotStop2d, mB, mD, nImg2d)
IntD1 = np.abs(PsiD)**2

PsiD0 = compute_nomask_fldD(Apod2d, Pupil2d, LyotStop2d, mD, nImg2d)
IntD0 = np.abs(PsiD0)**2

normD0 = 1./np.max(IntD0)

IntD1  *= normD0
IntD0 *= normD0 

print(f'Mask size: {rMask*2:.5f}lam/D')
print(f'Max intensity: {np.max(IntD1)}')

#%%
plt.figure(0)
plt.clf()
plt.subplot(121)
plt.imshow(Apod2d)         
plt.subplot(122)
plt.imshow(np.log10(IntD1))

#%%
r_t = np.linspace(-0.5, 0.5, nPup)

plt.figure(1)
plt.clf()
plt.plot(r_t, Pupil2d[nPup//2, :], label='pupil')
plt.plot(r_t, Apod2d[nPup//2, :], label='apod')
plt.xlabel('r (in pupil diameter)')
plt.ylabel('Normalized amplitude')
plt.xlim(-0.55, 0.55)
plt.ylim(-0.02, 1.02)
plt.legend()

#%%
plt.figure(2)
plt.clf()
plt.plot(np.log10(IntD0[nImg2d//2, nImg2d//2:]), label='no coro')
plt.plot(np.log10(IntD1[nImg2d//2, nImg2d//2:]), label='coro')
plt.ylim(-16.2, 0.2)
