#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:14:22 2020

@author: mndiaye
"""

"""
### Initialization
"""

import numpy as np
import pylab as pl
import corono as coro
import os

from pathlib import Path
from astropy.io import fits

from matplotlib.colors import LogNorm

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

#%%
"""
### Parameters
"""
nPup = 1024

nImg0 = 2048
nImg = 50
nBeg = (nImg0-nPup)//2
nEnd = (nImg0+nPup)//2

lam0_m = 575e-9

#%%
"""
### Directories & paths
"""

fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/Proposals/CNES/2019/WFIRST corono/phaseb_data/hlc_20190210/').resolve()

# Telescope Aperture
fname_tel = 'run461_pupil.fits'
fpath_tel = fdir / fname_tel 

# DM1 WFE
fname_dm1 = 'run461_dm1wfe.fits'
fpath_dm1 = fdir / fname_dm1

# DM2 WFE
fname_dm2 = 'run461_dm2wfe.fits'
fpath_dm2 = fdir / fname_dm2 

# Focal Plane Mask
fname_fpm = 'run461_occ_lam5.75e-07theta6.69polp'
fname_fpm_re = fname_fpm + '_real' + '_rotated' + '.fits'
fname_fpm_im = fname_fpm + '_imag' + '_rotated' + '.fits'

fpath_fpm_re = fdir / fname_fpm_re 
fpath_fpm_im = fdir / fname_fpm_im

# Lyot Stop
fname_lys = 'run461_lyot_rotated.fits'
fpath_lys = fdir / fname_lys 

#%%
"""
### File reading
"""
# Telescope Aperture
Pupil = fits.getdata(fpath_tel)

# DM1 WFE
dm1wfe = fits.getdata(fpath_dm1)

# DM2 WFE
dm2wfe = fits.getdata(fpath_dm2)

# FPM
fpm_re0 = fits.getdata(fpath_fpm_re)
fpm_im0 = fits.getdata(fpath_fpm_im)

# Lyot Stop
LyotStop = fits.getdata(fpath_lys)

#%%
"""
### Focal Plane Mask
"""
fpm_re = fpm_re0[nBeg:nEnd,nBeg:nEnd]
fpm_im = fpm_im0[nBeg:nEnd,nBeg:nEnd]

Mask = fpm_re + 1j*fpm_im


#%%
"""
### Electric Field Computation
"""
# Non coronagraphic field
Fld_A0 = Pupil*1
Fld_B0 = shift(fft(shift(Fld_A0)))
Fld_C0 = shift(ifft(shift(Fld_B0)))
Fld_C0 *= LyotStop
Fld_D0 = shift(fft(shift(Fld_C0)))

# Coronagraphic field
Fld_A = Pupil*np.exp(1j*2*np.pi*dm1wfe/lam0_m)
Fld_B = shift(fft(shift(Fld_A)))
Fld_B *= Mask
Fld_C = shift(ifft(shift(Fld_B)))
Fld_C *= LyotStop
Fld_D = shift(fft(shift(Fld_C)))

#%%
"""
### intensity Computation
"""
# Non coronagraphic image
Int_D0 = np.abs(Fld_D0)**2
peak = np.max(Int_D0)
Int_D0 /= peak

# Coronagraphic image
Int_D = np.abs(Fld_D)**2
Int_D /= peak

#%%
tmp = shift(ifft(shift(Fld_B)))

Int_CC = np.abs(tmp)**2
Int_C = np.abs(Fld_C)**2


#%%
"""
### Coronagraphic part display
"""
pl.figure(0)
pl.clf()
pl.imshow(Pupil)
pl.title('WFRIST pupil')

pl.figure(1)
pl.clf()
pl.imshow(fpm_re)
pl.title('FPM real part')

pl.figure(2)
pl.clf()
pl.imshow(fpm_im)
pl.title('FPM imag part')

pl.figure(3)
pl.clf()
pl.imshow(LyotStop)
pl.title('LyotStop')

#%%
"""
### Image display
"""
vmin0 = -10
vmax0 = 0

nSub = nPup
n0 = (nPup - nSub)//2
n1 = (nPup + nSub)//2 

Im0 = np.log10(Int_D0[n0:n1,n0:n1])
Im1 = np.log10(Int_D[n0:n1,n0:n1])

pl.figure(4)
pl.clf()
pl.imshow(Im0, vmin=vmin0, vmax=vmax0)
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')
pl.title('non coronagraphic image')

pl.figure(5)
pl.clf()
pl.imshow(Im1, vmin=vmin0, vmax=vmax0)
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')
pl.title('coronagraphic image')

#%%
"""
### Pupil plane display before and after Lyot Stop
"""
pl.figure(6)
pl.clf()
pl.subplot(121)
pl.imshow(Int_CC)
pl.title('Intensity (before Lyot stop)')
pl.subplot(122)
pl.imshow(Int_C)
pl.title('Intensity (after Lyot stop)')

pl.show()

