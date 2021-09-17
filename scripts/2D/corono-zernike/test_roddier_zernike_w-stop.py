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
import pyzelda.ztools as ztools
#import time
import os
from pathlib import Path
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
mB = 2*rMask

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  5.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 7.0

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
CtrBtwnPix  = False
CtrBtwnPix2 = False
Pupil2dSym  = True
ImPart = False 

#nlam
bw   = 0.1
nlam = 1

do_fits = True

lam0 = 1.65e-6
coeff = 0.02*lam0

vmin0 = -100
vmax0 = 100

# Size factor
k_big    = 10
mB_big   = k_big*mB
nPup_big = nPup*k_big
# Zernike mode
iZern = 11

#%%
"""
### Directories
"""
fdir_pdf = Path('../../../results/2D/plots/corono-zernike/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

#%%
"""
### Pupils
"""
Pupil2d    = coro.utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
Pupil2d_big = coro.utils.uniform_disk(nPup_big, nPup/2., CtrBtwnPix=CtrBtwnPix)
#
LyotStop2d = coro.utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
LyotStop2d_big = coro.utils.uniform_disk(nPup_big, nPup/2.*k_big, CtrBtwnPix=CtrBtwnPix)-\
coro.utils.uniform_disk(nPup_big, nPup/2., CtrBtwnPix=CtrBtwnPix)

#Pupil2d    = ztools.aperture.disc(nPup, nPup/2., cpix=True)
#Pupil2d_big = ztools.aperture.disc(nPup_big, nPup/2., cpix=True)
#
#LyotStop2d = ztools.aperture.disc(nPup, nPup/2., cpix=True)
#LyotStop2d_big = ztools.aperture.disc(nPup_big, nPup/2.*k_big, cpix=True)

"""
### Focal plane masks
"""
# phase shift for the Roddier and Zernike masks
eps_R = np.exp(1j*np.pi)
eps_Z = np.exp(1j*np.pi/2)
eps_R2 = np.exp(-1j*np.pi)
eps_Z2 = np.exp(-1j*np.pi/2)

# focal plane mask
mask2d     = coro.utils.uniform_disk(nFPM, nFPM/2., CtrBtwnPix=CtrBtwnPix)
#mask2d     = ztools.aperture.disc(nFPM, nFPM/2., cpix=True)

"""
### Introduced aberration
"""
# introduced OPD map
opd_ini = coeff*zernike.zernike1(iZern, npix = nPup, outside = 0)
# Corresponding phase map
phi = 2.*np.pi*opd_ini/lam0
phi_big = np.asarray(np.pad(phi, np.int(nPup*((k_big-1)/2)), mode='constant'))
# Corresponding phasor
phasor     = np.exp(1j*phi)
phasor_big = np.exp(1j*phi_big)

"""
### Centering
"""
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

int_Cwfe = np.abs(fld_Cwfe)**2

#%%
"""
### post-corono ZWFS (no aberration)
"""
fld_AA = Pupil2d_big*1

fld_BB = mask2d*coro.utils.sft(fld_AA, nFPM, mB_big, CtrBtwnPix=CtrBtwnPix)


fld_CC = fld_AA - (1.-eps_R)\
               *coro.utils.isft(fld_BB, nPup_big, mB_big, CtrBtwnPix=CtrBtwnPix)

fld_LL = fld_CC*LyotStop2d_big

fld_DD = coro.utils.sft(fld_LL, nImg2d, Fmax2d*k_big, CtrBtwnPix=CtrBtwnPix2)


fld_YY = mask2d*coro.utils.sft(fld_LL, nFPM, mB_big, CtrBtwnPix=CtrBtwnPix)

fld_ZZ_tmp = fld_LL - (1.-eps_Z)\
               *coro.utils.isft(fld_YY, nPup_big, mB_big, CtrBtwnPix=CtrBtwnPix)

#int_CC = np.abs(fld_CC)**2

#int_LL = np.abs(fld_LL)**2

int_DDtmp = np.abs(fld_DD)**2

int_DD    = int_DDtmp/np.max(int_D0tmp)
               
int_ZZ_tmp = np.abs(fld_ZZ_tmp)**2
               
int_ZZ = int_ZZ_tmp[np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup,
                      np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup]*\
                      LyotStop2d       


#%%
"""
### post-corono ZWFS (with aberration)
"""

fld_AAwfe = Pupil2d_big*phasor_big

fld_BBwfe = mask2d*coro.utils.sft(fld_AAwfe, nFPM, mB_big, CtrBtwnPix=CtrBtwnPix)

fld_CCwfe = fld_AAwfe - (1.-eps_R)\
               *coro.utils.isft(fld_BBwfe, nPup_big, mB_big, CtrBtwnPix=CtrBtwnPix)

#fld_CCwfe = fld_AAwfe*1.

fld_LLwfe = fld_CCwfe*LyotStop2d_big

fld_YYwfe = mask2d*coro.utils.sft(fld_LLwfe, nFPM, mB_big, CtrBtwnPix=CtrBtwnPix)

fld_ZZwfe_tmp = fld_LLwfe - (1.-eps_Z)\
               *coro.utils.isft(fld_YYwfe, nPup_big, mB_big, CtrBtwnPix=CtrBtwnPix)

#int_CCwfe = np.abs(fld_CCwfe)**2
int_LLwfe_tmp = np.abs(fld_LLwfe)**2
int_LLwfe     = int_LLwfe_tmp[np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup,
                            np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup]

int_ZZwfe_tmp = np.abs(fld_ZZwfe_tmp)**2  

int_ZZwfe = int_ZZwfe_tmp[np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup,
                            np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup]*\
                            LyotStop2d


#%%
fld_ZZ2wfe_tmp = fld_LLwfe - (1.-eps_Z2)\
               *coro.utils.isft(fld_YYwfe, nPup_big, mB_big, CtrBtwnPix=CtrBtwnPix)


int_ZZ2wfe_tmp = np.abs(fld_ZZ2wfe_tmp)**2  

int_ZZ2wfe = int_ZZ2wfe_tmp[np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup,
                            np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup]*\
                            LyotStop2d


#%%
"""
### Using opposite R&R mask to produce reference image
"""
fld_ref2_tmp = fld_LLwfe - (1.-eps_R2)\
               *coro.utils.isft(fld_YYwfe, nPup_big, mB_big, CtrBtwnPix=CtrBtwnPix)

int_ref2_tmp = np.abs(fld_ref2_tmp)**2  

int_ref2 = int_ref2_tmp[np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup,
                            np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup]*\
                            LyotStop2d

#%%
"""
### post-corono ZWFS (with aberration and reference)
"""
fld_AAref = Pupil2d_big*phasor_big

fld_BBref = mask2d*coro.utils.sft(fld_AAref, nFPM, mB_big, CtrBtwnPix=CtrBtwnPix)

#fld_CCwfe = fld_AAwfe - (1.-eps_R)\
#               *coro.utils.isft(fld_BBwfe, nPup_big, mB_big, CtrBtwnPix=CtrBtwnPix)

fld_CCref = fld_AAref*1.

fld_LLref = fld_CCref*LyotStop2d_big

#fld_YYwfe = mask2d*coro.utils.sft(fld_LLwfe, nFPM, mB_big, CtrBtwnPix=CtrBtwnPix)

#fld_ZZwfe_tmp = fld_LLwfe - (1.-eps_Z)\
#               *coro.utils.isft(fld_YYwfe, nPup_big, mB_big, CtrBtwnPix=CtrBtwnPix)

#int_CCwfe = np.abs(fld_CCwfe)**2
int_ref_tmp = np.abs(fld_LLref)**2
int_ref     = int_ref_tmp[np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup,
                            np.int(nPup*((k_big-1)/2)):np.int(nPup*((k_big-1)/2))+nPup]

#%%
"""
### zelda analysis
"""
z = zelda.Sensor('MISTIGRI')

#clear_pupil = int_ZZwfe2#int_LLwfe
clear_pupil = [int_ref, int_ref2]
zelda_pupil = [int_ZZwfe, int_ZZwfe]

clear_pupil = np.asarray(clear_pupil)
zelda_pupil = np.asarray(zelda_pupil)

wave = lam0*1

#%%
# OPD map extraction
opd_map = z.analyze(clear_pupil, zelda_pupil, wave, ratio_limit=100)

#%%
"""
### test with the vZWFS
"""
b_term, expi = ztools.create_reference_wave(z.mask_diameter, z.mask_depth,
                                                                z.mask_substrate, z.mask_Fratio,
                                                                z.pupil_diameter, z.pupil, wave)
#%%

vopd_map = (int_ZZ2wfe - int_ZZ2wfe)/(2*b_term.real)

#%%
pl.figure(30)
pl.clf()
pl.imshow(int_ZZ2wfe - int_ZZ2wfe)
pl.show()

#%%
fname_pdf = 'ZWFS_images_before-after_aberr_w-stop.pdf'
fpath_pdf = fdir_pdf / fname_pdf

pl.figure(0, (12,4))
pl.clf()

pl.subplot(131)
pl.imshow(int_ZZwfe, cmap='inferno', vmin=0, vmax=1.5)
pl.title(r'$Z_{WFS}$ image (with aberration)')

pl.subplot(132)
pl.imshow(int_ZZ, cmap='inferno', vmin=0, vmax=1.5)
pl.title(r'$Z_{WFS}$ image (no aberration)')

pl.subplot(133)
pl.imshow(int_ZZwfe-int_ZZ, cmap='inferno', vmin=-0.5, vmax=0.5)
pl.title(r'$\Delta Z_{WFS}$ image (difference)')
pl.tight_layout()

#pl.savefig(str(fpath_pdf), tight=True, transparent=True)


#%%
"""
### Tests with different reference images
"""

fname_pdf = 'ZWFS_images_before-after_ZWFSmask_ref=00_w-stop.pdf'
fpath_pdf = fdir_pdf / fname_pdf

pl.figure(1, (16,4))
pl.clf()
pl.subplot(141)
pl.imshow(int_LLwfe, cmap = 'inferno', vmin=0, vmax=1.5)
pl.title(r'ZWFS image before mask')

pl.subplot(142)
pl.imshow(int_ZZwfe, cmap = 'inferno', vmin=0, vmax=1.5)
pl.title(r'ZWFS image after mask')

pl.subplot(143)
pl.imshow(int_ref, cmap = 'inferno', vmin=0, vmax=1.5)
pl.title(r'Reference image (no mask)')

pl.subplot(144)
pl.imshow(-opd_map[0], cmap = 'inferno', vmin=vmin0, vmax=vmax0)
pl.title(r'Reconstructed OPD')

pl.tight_layout()
#pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
fname_pdf = 'ZWFS_images_before-after_ZWFSmask_ref=RR_w-stop.pdf'
fpath_pdf = fdir_pdf / fname_pdf

pl.figure(2, (16,4))
pl.clf()
pl.subplot(141)
pl.imshow(int_LLwfe, cmap = 'inferno', vmin=0, vmax=1.5)
pl.title(r'ZWFS image before mask')

pl.subplot(142)
pl.imshow(int_ZZwfe, cmap = 'inferno', vmin=0, vmax=1.5)
pl.title(r'ZWFS image after mask')

pl.subplot(143)
pl.imshow(int_ref2, cmap = 'inferno', vmin=0, vmax=1.5)
pl.title(r'Reference image (R&R)')

pl.subplot(144)
pl.imshow(-opd_map[1], cmap = 'inferno', vmin=vmin0, vmax=vmax0)
pl.title(r'Reconstructed OPD')

pl.tight_layout()
#pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
"""
### Reconstrcution error (reference, no mask)
"""
k_scl= 2

fname_pdf = 'ZWFS_reconstruction_ref=00_w-stop.pdf'
fpath_pdf = fdir_pdf / fname_pdf


f2 = pl.figure(3, (14,4))
pl.clf()

ax0 = f2.add_subplot(131)
im = ax0.imshow(opd_ini*1e9, cmap= 'inferno', vmin=vmin0, vmax=vmax0)
ax0.set_title('Introduced OPD')

ax1 = f2.add_subplot(132)
im = ax1.imshow(-opd_map[0], cmap = 'inferno', vmin=vmin0, vmax=vmax0)
ax1.set_title('Reconstructed OPD')

ax2 = f2.add_subplot(133)
im = ax2.imshow((-opd_map[0]-opd_ini*1e9)*k_scl, cmap = 'inferno', vmin=vmin0, vmax=vmax0)
ax2.set_title('OPD difference (x{0:.1f})'.format(k_scl))


f2.subplots_adjust(right=0.85)
cbar_ax = f2.add_axes([0.9, 0.1, 0.04, 0.8])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('OPD in nm', rotation=270, labelpad = 10)

#pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
"""
### Reconstrcution error (reference, R&R)
"""
k_scl= 2

fname_pdf = 'ZWFS_reconstruction_ref=RR_w-stop.pdf'
fpath_pdf = fdir_pdf / fname_pdf


f2 = pl.figure(4, (14,4))
pl.clf()

ax0 = f2.add_subplot(131)
im = ax0.imshow(opd_ini*1e9, cmap= 'inferno', vmin=vmin0, vmax=vmax0)
ax0.set_title('Introduced OPD')

ax1 = f2.add_subplot(132)
im = ax1.imshow(-opd_map[1], cmap = 'inferno', vmin=vmin0, vmax=vmax0)
ax1.set_title('Reconstructed OPD')

ax2 = f2.add_subplot(133)
im = ax2.imshow((-opd_map[1]-opd_ini*1e9)*k_scl, cmap = 'inferno', vmin=vmin0, vmax=vmax0)
ax2.set_title('OPD difference (x{0:.1f})'.format(k_scl))


f2.subplots_adjust(right=0.85)
cbar_ax = f2.add_axes([0.9, 0.1, 0.04, 0.8])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('OPD in nm', rotation=270, labelpad = 10)

#pl.savefig(str(fpath_pdf), tight=True, transparent=True)



#%%
"""
### Radial intensity profiles of the images
"""
fname_pdf = 'intensity_profiles_w-stop.pdf'
fpath_pdf = fdir_pdf / fname_pdf

if nImg2d%2 == 0:
    xi2d = xi2d_ctr*1

pl.figure(10, (8, 4.5))
pl.clf()
pl.title('Radial intensity profiles of the images')

pl.semilogy(xi2d,int_DD[nImg2d//2,nImg2d//2:],label='w/  corono (ZWFS camera)')
pl.semilogy(xi2d,int_D0[nImg2d//2,nImg2d//2:],label='w/o corono (Sci. camera)', ls ='--')
#pl.semilogy(xi2d,int_D[nImg2d//2,nImg2d//2:],label='w/  corono (Sci. camera)')

#pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.05, 10.05)
pl.ylim(2e-6, 2e0)

pl.axvspan(-1, rMask, alpha=0.25, color='b')

pl.grid(True, which='both')

pl.legend()
pl.tight_layout()
#pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
"""
### Image comparison
"""

vmin1 = -6
vmax1 = 0

fname_pdf = 'PSF_images_w-stop.pdf'
fpath_pdf = fdir_pdf / fname_pdf

pl.figure(11, (4, 8))
pl.clf()

pl.subplot(211)
pl.imshow(np.log10(int_D0), 
          cmap = 'inferno', vmin=vmin1, vmax=vmax1)
pl.title('w/o corono (Sci. camera)')

pl.subplot(212)
pl.imshow(np.log10(int_DD), 
          cmap = 'inferno', vmin=vmin1, vmax=vmax1)
pl.title('w/o corono (WFS camera)')


pl.tight_layout()
#pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
pl.show()