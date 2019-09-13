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

try:
    import pyfftw
    shift = pyfftw.interfaces.numpy_fft.fftshift
    fft   = pyfftw.interfaces.numpy_fft.fft2
    ifft  = pyfftw.interfaces.numpy_fft.ifft2
    print("using pyfftw library!")
except:
    shift = np.fft.fftshift # short-hand for FFTs
    fft   = np.fft.fft2
    ifft  = np.fft.ifft2

#
#from astropy.io import fits

#%% parameters
"""
Parameters
"""
# discretization parameters
nPup = 300
nFPM = 100

# mask radius in lam0/D units
rMask = 1.062/2
mB = 2*rMask

# vortex charge
ll = 2

# Pixel centering
CtrBtwnPix  = False
CtrBtwnPix2 = False

# spectral bandwidth
bw   = 0.1
nlam = 1
lam0 = 1.65e-6

# save fits
do_fits = True

# scale in nm for aberration plots
vmin0 = -100
vmax0 = 100

# Size factor
k_big  = 10
mB_big = k_big*mB
nArr   = nPup*k_big
nImg2d = nArr*1

# Zernike mode
iZern = 4
coeff = 0.02*lam0

# Pupil extraction in padded array
idx_ini = np.int(nPup*((k_big-1)/2))
idx_end = idx_ini+nPup

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
Pupil2d_pad = coro.utils.uniform_disk(nArr, nPup/2., CtrBtwnPix=CtrBtwnPix)
#
LyotStop2d_pad = coro.utils.uniform_disk(nArr, nPup/2., CtrBtwnPix=CtrBtwnPix)
LyotStop2d_big = coro.utils.uniform_disk(nArr, nPup/2.*k_big, CtrBtwnPix=CtrBtwnPix)
LyotStop2d = coro.utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)

#Pupil2d    = ztools.aperture.disc(nPup, nPup/2., cpix=True)
#Pupil2d_pad = ztools.aperture.disc(nPup_big, nPup/2., cpix=True)
#
#LyotStop2d = ztools.aperture.disc(nPup, nPup/2., cpix=True)
#LyotStop2d_pad = ztools.aperture.disc(nPup_big, nPup/2.*k_big, cpix=True)



"""
### Focal plane masks
"""
# phase shift for the Zernike mask
eps_Z = np.exp(1j*np.pi/2)

# focal plane mask
mask2d     = coro.utils.uniform_disk(nFPM, nFPM/2., CtrBtwnPix=CtrBtwnPix)
#mask2d     = ztools.aperture.disc(nFPM, nFPM/2., cpix=True)

# vortex focal plane mask
xx, yy = np.meshgrid(np.arange(nArr)-nArr//2, np.arange(nArr)-nArr//2)

theta2d = np.arctan(yy/xx)
theta2d[nArr//2, nArr//2] = 0.
exp_V  = np.exp(1j*ll*theta2d)
exp_V2 = np.exp(-1j*ll*theta2d)

"""
### Introduced aberration
"""
# introduced OPD map
opd_ini = coeff*zernike.zernike1(iZern, npix = nPup, outside = 0)
# Corresponding phase map
phi = 2.*np.pi*opd_ini/lam0
phi_pad = np.asarray(np.pad(phi, np.int(nPup*((k_big-1)/2)), mode='constant'))
# Corresponding phasor
phasor     = np.exp(1j*phi)
phasor_pad = np.exp(1j*phi_pad)

"""
### Centering
"""
val = 0
if nImg2d%2 == 0:
    val = 1/2        
xi2d     = (np.arange(nImg2d//2+1))* nPup/nArr
xi2d_ctr = (np.arange(nImg2d//2)+val)* nPup/nArr
 
#%%
"""
### Compute direct field
"""
fld_A0 = Pupil2d_pad*1.
fld_B0 = shift(fft(shift(fld_A0)))
fld_C0 = shift(ifft(shift(fld_B0)))
fld_L0 = fld_C0*LyotStop2d_pad
fld_D0 = shift(fft(shift(fld_L0)))

int_D0tmp = np.abs(fld_D0)**2
int_D0 = int_D0tmp/np.max(int_D0tmp)


#%%
"""
### Compute coronagraphic field
"""
fld_A = Pupil2d_pad*1.
fld_B = shift(fft(shift(fld_A)))*exp_V
fld_C = shift(ifft(shift(fld_B)))
fld_L = fld_C*LyotStop2d_pad
fld_D = shift(fft(shift(fld_L)))

int_Dtmp = np.abs(fld_D)**2
int_D    = int_Dtmp/np.max(int_D0tmp)

int_C = np.abs(fld_C)**2

#%%
pl.figure(10)
pl.clf()
pl.subplot(121)
pl.imshow(np.log10(int_D0), vmin=-8, vmax=0)
pl.subplot(122)
pl.imshow(np.log10(int_D), vmin=-8, vmax=0)

pl.show()

xlam = (nPup/nArr)*np.arange(nArr//2)

pl.figure(30, (8, 4.5))
pl.clf()
pl.semilogy(xlam, int_D0[nArr//2, nArr//2:], label='no vortex')
pl.semilogy(xlam, int_D[nArr//2, nArr//2:], label='vortex')
pl.xlim(0, 20)
pl.ylim(3e-11,3e0)
pl.xlabel(r'Angular separation in $\lambda$/D')
pl.ylabel(r'Normalized intensity')


#%%
"""
### Compute coronagraphic field with an aberration
"""
fld_Awfe = Pupil2d_pad*phasor_pad
fld_Bwfe = shift(fft(shift(fld_Awfe)))*exp_V
fld_Cwfe = shift(ifft(shift(fld_Bwfe)))
fld_Lwfe = fld_Cwfe*LyotStop2d_pad
fld_Dwfe = shift(fft(shift(fld_Lwfe)))

int_Cwfe = np.abs(fld_Cwfe)**2

#%%
pl.figure(31)
pl.clf()
pl.subplot(121)
pl.imshow(int_C[idx_ini:idx_end,idx_ini:idx_end]**0.25)
pl.title('No aberration')
pl.subplot(122)
pl.imshow(int_Cwfe[idx_ini:idx_end,idx_ini:idx_end]**0.25)
pl.title('With aberration')
pl.show()

#%%
"""
### post-corono ZWFS (no aberration)
"""
fld_AA = Pupil2d_pad*1
fld_BB = shift(fft(shift(fld_AA)))*exp_V
fld_CC = shift(ifft(shift(fld_BB)))
fld_LL = fld_CC*LyotStop2d_big
fld_DD = shift(fft(shift(fld_LL)))


fld_YY = mask2d*coro.utils.sft(fld_LL, nFPM, mB_big, CtrBtwnPix=CtrBtwnPix)

fld_ZZ_tmp = fld_LL - (1.-eps_Z)\
               *coro.utils.isft(fld_YY, nArr, mB_big, CtrBtwnPix=CtrBtwnPix)

#int_CC = np.abs(fld_CC)**2
#int_LL = np.abs(fld_LL)**2
int_DDtmp = np.abs(fld_DD)**2
int_DD    = int_DDtmp/np.max(int_D0tmp)
               
int_ZZ_tmp = np.abs(fld_ZZ_tmp)**2
int_ZZ = int_ZZ_tmp[idx_ini:idx_end,idx_ini:idx_end]*LyotStop2d       

#%%
"""
### post-corono ZWFS (with aberration)
"""

fld_AAwfe = Pupil2d_pad*phasor_pad
fld_BBwfe = shift(fft(shift(fld_AAwfe)))*exp_V
fld_CCwfe = shift(ifft(shift(fld_BBwfe)))
fld_LLwfe = fld_CCwfe*LyotStop2d_big
fld_DDwfe = shift(fft(shift(fld_LLwfe)))

fld_YYwfe = mask2d*coro.utils.sft(fld_LLwfe, nFPM, mB_big, CtrBtwnPix=CtrBtwnPix)

fld_ZZwfe_tmp = fld_LLwfe - (1.-eps_Z)\
               *coro.utils.isft(fld_YYwfe, nArr, mB_big, CtrBtwnPix=CtrBtwnPix)

#int_CCwfe = np.abs(fld_CCwfe)**2
int_LLwfe_tmp = np.abs(fld_LLwfe)**2
int_LLwfe     = int_LLwfe_tmp[idx_ini:idx_end,idx_ini:idx_end]

int_ZZwfe_tmp = np.abs(fld_ZZwfe_tmp)**2  
int_ZZwfe = int_ZZwfe_tmp[idx_ini:idx_end,idx_ini:idx_end]*LyotStop2d

#%%
pl.figure(32)
pl.clf()
pl.subplot(141)
pl.imshow(int_C[idx_ini:idx_end,idx_ini:idx_end]**0.25)
pl.title('No aberration (Science camera)')
pl.subplot(142)
pl.imshow(int_ZZ**0.25)
pl.title('No aberration (ZWFS camera)')
pl.subplot(143)
pl.imshow(int_ZZwfe**0.25)
pl.title('w/ aberration (ZWFS camera)')
pl.subplot(144)
pl.imshow(np.abs(int_ZZwfe-int_ZZ)**0.25)
pl.title('Difference')
pl.show()

#%%
"""
### Using opposite vortex mask to produce reference image
"""
fld_ZZref2_tmp = shift(fft(shift(fld_DDwfe*exp_V2)))
int_ZZref2_tmp = np.abs(fld_ZZref2_tmp)**2  
int_ZZref2 = int_ZZref2_tmp[idx_ini:idx_end,idx_ini:idx_end]*LyotStop2d

#%%
"""
### post-corono ZWFS (with aberration and reference)
"""
fld_AAref = Pupil2d_pad*phasor_pad
fld_BBref = shift(fft(shift(fld_AAref)))
fld_CCref = shift(ifft(shift(fld_BBref)))
fld_LLref = fld_CCref*LyotStop2d_pad

int_LLref_tmp = np.abs(fld_LLref)**2
int_LLref     = int_LLref_tmp[idx_ini:idx_end,idx_ini:idx_end]

#%%
"""
### zelda analysis
"""
z = zelda.Sensor('MISTIGRI')

#clear_pupil = int_ZZwfe2#int_LLwfe
clear_pupil = [int_LLref, int_ZZref2]
zelda_pupil = [int_ZZwfe, int_ZZwfe]

clear_pupil = np.asarray(clear_pupil)
zelda_pupil = np.asarray(zelda_pupil)

wave = lam0*1

#%%
# OPD map extraction
opd_map = z.analyze(clear_pupil, zelda_pupil, wave, ratio_limit=100)

#%%
fname_pdf = 'ZWFS_images_before-after_aberr-vortex_l={0}.pdf'.format(ll)
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

pl.savefig(str(fpath_pdf), tight=True, transparent=True)


#%%
"""
### Tests with different reference images
"""

fname_pdf = 'ZWFS_images_before-after_ZWFSmask_ref=00_vortex_l={0}.pdf'.format(ll)
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
pl.imshow(int_LLref, cmap = 'inferno', vmin=0, vmax=1.5)
pl.title(r'Reference image (no mask)')

pl.subplot(144)
pl.imshow(-opd_map[0], cmap = 'inferno', vmin=vmin0, vmax=vmax0)
pl.title(r'Reconstructed OPD')

pl.tight_layout()
pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
fname_pdf = 'ZWFS_images_before-after_ZWFSmask_ref=VV_l={0}.pdf'.format(ll)
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
pl.imshow(int_ZZref2, cmap = 'inferno', vmin=0, vmax=1.5)
pl.title(r'Reference image (Vortex)')

pl.subplot(144)
pl.imshow(-opd_map[1], cmap = 'inferno', vmin=vmin0, vmax=vmax0)
pl.title(r'Reconstructed OPD')

pl.tight_layout()
pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
"""
### Reconstrcution error (reference, no mask)
"""
k_scl= 2

fname_pdf = 'ZWFS_reconstruction_ref=00_vortex_l={0}.pdf'.format(ll)
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

pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
"""
### Reconstrcution error (reference, R&R)
"""
k_scl= 2

fname_pdf = 'ZWFS_reconstruction_ref=VV_l={0}.pdf'.format(ll)
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

pl.savefig(str(fpath_pdf), tight=True, transparent=True)



#%%
"""
### Radial intensity profiles of the images
"""
fname_pdf = 'intensity_profiles_vortex_l={0}.pdf'.format(ll)
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

#pl.axvspan(-1, rMask, alpha=0.25, color='b')

pl.grid(True, which='both')

pl.legend()
pl.tight_layout()
pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
vmin1 = -6
vmax1 = 0

idx_ini2 = nArr//2 - 225//2
idx_end2 = idx_ini2 + 225

fname_pdf = 'PSF_images_vortex_l={0}.pdf'.format(ll)
fpath_pdf = fdir_pdf / fname_pdf

pl.figure(11, (4, 8))
pl.clf()

pl.subplot(211)
pl.imshow(np.log10(int_D0[idx_ini2:idx_end2,idx_ini2:idx_end2]), 
          cmap = 'inferno', vmin=vmin1, vmax=vmax1)
pl.title('w/o corono (Sci. camera)')

pl.subplot(212)
pl.imshow(np.log10(int_DD[idx_ini2:idx_end2,idx_ini2:idx_end2]), 
          cmap = 'inferno', vmin=vmin1, vmax=vmax1)
pl.title('w/o corono (WFS camera)')


pl.tight_layout()
pl.savefig(str(fpath_pdf), tight=True, transparent=True)

#%%
pl.show()
