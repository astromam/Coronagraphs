#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:42:29 2020

@author: mndiaye
"""

#%%
"""
*** Initialization
"""
import numpy as np
import time
import os

from pathlib import Path
import corono as coro
from corono.utils import sft, isft, besselJ0

import pylab as pl
import pandas as pd

from astropy.io import fits

#%% parameters
"""
*** Parameters
"""
# coronagraph type
corono_name  = 'APLC' # 'APLC' or 'SP'

# sampling
nPup = 300
nFPM = 70
nImg = 256
Fmax = 50
R    = 1

nPup2d = 2*nPup
nImg2d = 2*nImg
Fmax2d = 100

# spectral sampling
bw   = 0.2
nlam = 11

# coronagraph configuration
PupilID    = 0.14
rMask      = 2.8
LyotStopID = 0.28
LyotStopOD = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 5.
rho1 = 40.0

# contrast in the dark region
cDarkHole = 8.0

# pupil definitions
r   = np.arange(nPup)*R/nPup + R/(2*nPup)
xi  = (Fmax/nImg)*np.arange(nImg+1)

Pupil1d      = (r>PupilID)*1.0
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

# centering aspects
CtrBtwnPix = False
CtrBtwnPix2 = False

# dictionary parameters
params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 corono_name = corono_name,
                 CtrBtwnPix = CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nImg2d = nImg2d, Fmax2d = Fmax2d)

#%%
"""
*** Working directory
"""
fdir = Path('../../results/1D/dat_pyth').resolve()
if not os.path.exists(fdir):
    os.makedirs(fdir)      

#%%
"""
*** GPI File directory - 1D
"""
# generic name for GPI coronagraph
fgen = 'obs=14_m=2800_bw=20_rI=3400_rO=20_C=08_b=005'


# file directory for GPI coronagraph
fdir_gpi_1d = Path('/Users/mndiaye/OneDrive - Université Nice Sophia \
Antipolis/Work/Code/Routines/apodizer linear programming/APLC \
paper/GPI/1D/').resolve()


# filename for the GPI parts
fname_pup_1d = fgen + '_pupIII.csv'
fname_apo_1d = fgen + '_apoIII.csv'
fname_stp_1d = fgen + '_stpIII.csv'

# filename for the GPI PSFs
fname_psfmono_1d = fgen + '_psfmonoIII.csv'
fname_psfpoly_1d = fgen + '_psfpolyIII.csv'
fname_cormono_1d = fgen + '_cormonoIII.csv'
fname_corpoly_1d = fgen + '_corpolyIII.csv'

# filepath for the GPI parts
fpath_pup_1d  = fdir_gpi_1d / fname_pup_1d
fpath_apo_1d  = fdir_gpi_1d / fname_apo_1d
fpath_stp_1d  = fdir_gpi_1d / fname_stp_1d

# filepath for the GPI PSFs
fpath_psfmono_1d  = fdir_gpi_1d / fname_psfmono_1d
fpath_psfpoly_1d  = fdir_gpi_1d / fname_psfpoly_1d
fpath_cormono_1d  = fdir_gpi_1d / fname_cormono_1d
fpath_corpoly_1d  = fdir_gpi_1d / fname_corpoly_1d

#%%
"""
*** GPI File directory - 2D
"""
# file directory for GPI coronagraph
fdir_gpi_2d = Path('/Users/mndiaye/OneDrive - Université Nice Sophia \
Antipolis/Work/Code/Routines/apodizer linear programming/APLC \
paper/GPI/2D/').resolve()

# filename for the GPI parts
fname_pup_2d = fgen + '_pupil2D.fits'
fname_apo_2d = fgen + '_apoIII2D.fits'
fname_stp_2d = fgen + '_LyotStop2D.fits'

# filename for the GPI PSFs
fname_psfmono_2d = fgen + '_psfmonoIII2D.fits'
fname_psfpoly_2d = fgen + '_psfpolyIII2D.fits'
fname_cormono_2d = fgen + '_cormonoIII2D.fits'
fname_corpoly_2d = fgen + '_corpolyIII2D.fits'

fname_cormono_2d_PupRe = fgen + '_cormonoIII2DPupRe.fits'
fname_cormono_2d_PupIm = fgen + '_cormonoIII2DPupIm.fits'

fname_fieldBRe = fgen + '_fieldBRe.fits'
fname_fieldBIm = fgen + '_fieldBIm.fits'
fname_fieldCRe = fgen + '_fieldCRe.fits'
fname_fieldCIm = fgen + '_fieldCIm.fits'

fname_Mask = fgen + '_Mask.fits'

# filepath for the GPI parts
fpath_pup_2d  = fdir_gpi_2d / fname_pup_2d
fpath_apo_2d  = fdir_gpi_2d / fname_apo_2d
fpath_stp_2d  = fdir_gpi_2d / fname_stp_2d

# filepath for the GPI PSFs
fpath_psfmono_2d  = fdir_gpi_2d / fname_psfmono_2d
fpath_psfpoly_2d  = fdir_gpi_2d / fname_psfpoly_2d
fpath_cormono_2d  = fdir_gpi_2d / fname_cormono_2d
fpath_corpoly_2d  = fdir_gpi_2d / fname_corpoly_2d

#filepath for the pupils
fpath_cormono_2d_PupRe  = fdir_gpi_2d / fname_cormono_2d_PupRe
fpath_cormono_2d_PupIm  = fdir_gpi_2d / fname_cormono_2d_PupIm

# filepath for the electric field
fpath_fieldBRe  = fdir_gpi_2d / fname_fieldBRe
fpath_fieldBIm  = fdir_gpi_2d / fname_fieldBIm
fpath_fieldCRe  = fdir_gpi_2d / fname_fieldCRe
fpath_fieldCIm  = fdir_gpi_2d / fname_fieldCIm

fpath_Mask = fdir_gpi_2d / fname_Mask

#%%
"""
*** Import GPI 1D files and convert them into numpy array
"""
Pupil1d = pd.read_csv(fpath_pup_1d, header = None).to_numpy().flatten()
Apod1d = pd.read_csv(fpath_apo_1d, header = None).to_numpy().flatten()
LyotStop1d = pd.read_csv(fpath_stp_1d, header = None).to_numpy().flatten()

Psfmono_1d = pd.read_csv(fpath_psfmono_1d, header = None).to_numpy().flatten()
Psfpoly_1d = pd.read_csv(fpath_psfpoly_1d, header = None).to_numpy().flatten()
Cormono_1d = pd.read_csv(fpath_cormono_1d, header = None).to_numpy().flatten()
Corpoly_1d = pd.read_csv(fpath_corpoly_1d, header = None).to_numpy().flatten()

#%%
"""
*** Import GPI 2D files and convert them into numpy array
"""
Pupil2d = fits.getdata(fpath_pup_2d)
Apod2d = fits.getdata(fpath_apo_2d)
LyotStop2d = fits.getdata(fpath_stp_2d)

Psfmono_2d = fits.getdata(fpath_psfmono_2d)
Psfpoly_2d = fits.getdata(fpath_psfpoly_2d)
Cormono_2d = fits.getdata(fpath_cormono_2d)
Corpoly_2d = fits.getdata(fpath_corpoly_2d)

Cormono_2d_PupRe = fits.getdata(fpath_cormono_2d_PupRe)
Cormono_2d_PupIm = fits.getdata(fpath_cormono_2d_PupIm)

mathfield_B_re = fits.getdata(fpath_fieldBRe)
mathfield_B_im = fits.getdata(fpath_fieldBIm)
mathfield_C_re = fits.getdata(fpath_fieldCRe)
mathfield_C_im = fits.getdata(fpath_fieldCIm)

mathMask2d = fits.getdata(fpath_Mask)

#%%
"""
*** Define 1d coronagraph
"""
params1d    = coro.update_params(params, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d) 

if corono_name == 'APLC':
    corono1d = coro.design.APLC1d(**params1d)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))


#%%
"""
*** Define 2d coronagraph
"""
params2d    = coro.update_params(params, 
                                Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                                nPup=nPup2d, nFPM=nFPM*2) 

if corono_name == 'APLC':
    corono2d = coro.design.APLC2d(**params2d)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
*** Computation of the direct and coronagraphic images (1d)
"""
# monochromatic images
mono_direct_image0_1d = corono1d.compute_direct_intensity_1d(Apod1d, poly=False)
mono_corono_image0_1d = corono1d.compute_corono_intensity_1d(Apod1d, poly=False)

mono_peak_1d = mono_direct_image0_1d[nlam//2].max()
mono_direct_image0_1d /= mono_peak_1d
mono_corono_image0_1d /= mono_peak_1d

# broadband images
poly_direct_image0_1d = corono1d.compute_direct_intensity_1d(Apod1d)
poly_corono_image0_1d = corono1d.compute_corono_intensity_1d(Apod1d)

poly_peak_1d = poly_direct_image0_1d.max()
poly_direct_image0_1d /= poly_peak_1d
poly_corono_image0_1d /= poly_peak_1d


#%%
"""
*** Computation of the direct and coronagraphic images (2d)
"""
# monochromatic images
mono_direct_image0 = corono2d.compute_direct_intensity_2d(Apod2d, poly=False)
mono_corono_image0 = corono2d.compute_corono_intensity_2d(Apod2d, poly=False)

mono_peak = mono_direct_image0[nlam//2].max()
mono_direct_image0 /= mono_peak
mono_corono_image0 /= mono_peak

# broadband images
poly_direct_image0 = corono2d.compute_direct_intensity_2d(Apod2d)
poly_corono_image0 = corono2d.compute_corono_intensity_2d(Apod2d)

poly_peak = poly_direct_image0.max()
poly_direct_image0 /= poly_peak
poly_corono_image0 /= poly_peak

# pupil images
poly_corono_pupil0 = corono2d.compute_corono_lyot_field_2d(Apod2d)

#%%
"""
### Mask diffracted wave - 1d
"""

nFPMi = (nFPM*corono1d.ilam_t).astype(int)
nFPMi_max = np.max(nFPMi)

FPM_t = np.arange(nFPMi_max+1)                           
mi   = corono1d.dmi*FPM_t*\
    (FPM_t[None,:] <= nFPMi[:,None])           
        
# variable inside the Bessel function for Hankel transform to the FPM plane
HK_B_var = np.pi*corono1d.mi[:,:,None]*corono1d.r[None,None,:]
# Hankel kernel (direct transform) to the FPM plane
HK_B = np.pi*besselJ0(HK_B_var)*corono1d.r[None, None,:]*corono1d.dr
# Hankel kernel (inverse transform) to the FPM plane
iHK_B = np.pi*besselJ0(HK_B_var.transpose(0,2,1))*mi[:, None,:]*corono1d.dmi 


E_field = Apod1d*Pupil1d
FPM_field_1d = np.einsum("ijk,k->ij", HK_B, E_field)


#%%
"""
### Mask diffracted wave - 2d
"""
field_A    = Apod2d*Pupil2d
FPM_field_2d = sft(field_A, corono2d.nFPM, corono2d.mB_t[nlam//2], 
                                                CtrBtwnPix=corono2d.CtrBtwnPix)

FPM_field_2d_re = FPM_field_2d.real
FPM_field_2d_im = FPM_field_2d.imag


#%%
"""
### Display mask diffracted wave
"""

pl.figure(20)
pl.clf()
pl.subplot(121)
pl.imshow(FPM_field_2d_re, cmap = 'inferno')
pl.subplot(122)
pl.imshow(FPM_field_2d_im, cmap = 'inferno')
pl.show()

#%%
xiFPM1d = (rMask/nFPM)*np.arange(nFPMi_max+1)
FPM_field_1d_mono = FPM_field_1d[nlam//2]

xiFPM2d = np.arange(corono2d.nFPM//2)*(2.*rMask)/corono2d.nFPM
FPM_field_2d_re_vec = FPM_field_2d_re[corono2d.nFPM//2,corono2d.nFPM//2:]



pl.figure(21)
pl.clf()
pl.semilogy(xiFPM1d, np.abs(FPM_field_1d_mono), label='1d')
pl.semilogy(xiFPM2d, np.abs(FPM_field_2d_re_vec), label='2d')
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.5, 5.5)
pl.ylim(10**(-8.2), 10**(1.8))
pl.legend()

#%%
# pl.figure(22)
# pl.clf()
# for ilam in range(nlam):
#     pl.semilogy(xiFPM1d, np.abs(FPM_field_1d[ilam]), label='{0}'.format(ilam))
# pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='k', linestyle='--')    

#%%
"""
*** Display images
"""
## Pupil

# pl.figure(0, (8, 4.5))
# pl.clf()
# pl.subplot(131)
# pl.imshow(Pupil2d, cmap = 'inferno')
# pl.title('Pupil')
# pl.subplot(132)
# pl.imshow(Apod2d, cmap = 'inferno')
# pl.title('Apodizer')
# pl.subplot(133)
# pl.imshow(LyotStop2d, cmap = 'inferno')
# pl.title('Lyot Stop')


#%%
# mask2d  = corono2d.mask2d
# field_B = coro.utils.sft(Pupil2d*Apod2d, corono2d.nFPM, corono2d.mB_t[nlam//2], CtrBtwnPix=corono2d.CtrBtwnPix)

# pythfield_B_re = field_B.real
# pythfield_B_im = field_B.imag

# pl.figure(11, (8,6))
# pl.clf()
# pl.subplot(231)
# pl.imshow(pythfield_B_re, cmap= 'inferno')
# pl.title(r'pyth - $Re[\Psi_B]$')
# pl.subplot(232)
# pl.imshow(mathfield_B_re, cmap= 'inferno')
# pl.title(r'math - $Re[\Psi_B]$')
# pl.subplot(233)
# pl.imshow(mathfield_B_re-pythfield_B_re, cmap= 'inferno')
# pl.title(r'diff - $Re[\Psi_B]$')
# pl.subplot(234)
# pl.imshow(pythfield_B_im, cmap= 'inferno')
# pl.title(r'pyth - $Im[\Psi_B]$')
# pl.subplot(235)
# pl.imshow(mathfield_B_im, cmap= 'inferno')
# pl.title(r'math - $Im[\Psi_B]$')
# pl.subplot(236)
# pl.imshow(mathfield_B_im-pythfield_B_im, cmap= 'inferno')
# pl.title(r'diff - $Im[\Psi_B]$')


#%%

# field_C = coro.utils.isft(mask2d*field_B, corono2d.nPup, corono2d.mB_t[nlam//2], CtrBtwnPix=corono2d.CtrBtwnPix)

# pythfield_C_re = field_C.real
# pythfield_C_im = field_C.imag

# pl.figure(12, (8,6))
# pl.clf()
# pl.subplot(231)
# pl.imshow(pythfield_C_re, cmap= 'inferno')
# pl.title(r'pyth - $Re[TF\Psi_B]$')
# pl.subplot(232)
# pl.imshow(mathfield_C_re, cmap= 'inferno')
# pl.title(r'math - $Re[TF\Psi_B]$')
# pl.subplot(233)
# pl.imshow(mathfield_C_re - pythfield_C_re, cmap= 'inferno')
# pl.title(r'diff - $Re[TF\Psi_B]$')
# pl.subplot(234)
# pl.imshow(pythfield_C_im, cmap= 'inferno')
# pl.title(r'pyth - $Im[TF\Psi_B]$')
# pl.subplot(235)
# pl.imshow(mathfield_C_im, cmap= 'inferno')
# pl.title(r'math - $Im[TF\Psi_B]$')
# pl.subplot(236)
# pl.imshow(mathfield_C_im - pythfield_C_im, cmap= 'inferno')
# pl.title(r'diff - $Im[TF\Psi_B]$')



#%%

# pyth_pup_re = (LyotStop2d*poly_corono_pupil0[nlam//2]).real
# pyth_pup_im = (LyotStop2d*poly_corono_pupil0[nlam//2]).imag

# math_pup_re = Cormono_2d_PupRe[nlam//2]
# math_pup_im = Cormono_2d_PupIm[nlam//2]

# diff_pup_re = math_pup_re - pyth_pup_re
# diff_pup_im = math_pup_im - pyth_pup_im

# pl.figure(10, (8, 6))
# pl.clf()
# pl.subplot(231)
# pl.imshow(pyth_pup_re, cmap = 'inferno')
# pl.title(r'pyth - $Re[\Psi_C]$')
# pl.subplot(232)
# pl.imshow(math_pup_re, cmap = 'inferno')
# pl.title(r'math - $Re[\Psi_C]$')
# pl.subplot(233)
# pl.imshow(diff_pup_re, cmap = 'inferno')
# pl.title(r'diff - $Re[\Psi_C]$')

# pl.subplot(234)
# pl.imshow(pyth_pup_im, cmap = 'inferno')
# pl.title(r'pyth - $Im[\Psi_C]$')
# pl.subplot(235)
# pl.imshow(math_pup_im, cmap = 'inferno')
# pl.title(r'math - $Im[\Psi_C]$')
# pl.subplot(236)
# pl.imshow(diff_pup_im, cmap = 'inferno')
# pl.title(r'diff - $Im[\Psi_C]$')

#%%
# pl.figure(14, (8,6))
# pl.clf()
# pl.subplot(131)
# pl.imshow(mask2d)
# pl.title('pyth - FPM')
# pl.subplot(132)
# pl.imshow(mathMask2d)
# pl.title('math - FPM')
# pl.subplot(133)
# pl.imshow(mask2d-mathMask2d)
# pl.title('diff - FPM')
# pl.show()

#%%
## Monochromatic images

vmin0 = -14
vmax0 = -4

# pl.figure(1, (8, 4.5))
# pl.clf()
# pl.subplot(131)
# pl.imshow(np.log10(Cormono_2d), cmap = 'inferno', vmin=vmin0,vmax=vmax0)
# pl.title('math - mono')
# pl.subplot(132)
# pl.imshow(np.log10(mono_corono_image0[nlam//2]), cmap = 'inferno', vmin=vmin0,vmax=vmax0)
# pl.title('python - mono')
# pl.subplot(133)
# pl.imshow(np.log10(np.abs(mono_corono_image0[nlam//2]-Cormono_2d)), cmap = 'inferno', vmin=vmin0,vmax=vmax0)
# pl.title('difference in abs value')

#%%
## Broadband images

# pl.figure(2, (8, 4.5))
# pl.clf()
# pl.subplot(131)
# pl.imshow(np.log10(Corpoly_2d), cmap = 'inferno')
# pl.title('math - poly')
# pl.subplot(132)
# pl.imshow(np.log10(poly_corono_image0), cmap = 'inferno')
# pl.title('python - poly')
# pl.subplot(133)
# pl.imshow(np.log10(np.abs(poly_corono_image0-Corpoly_2d)), cmap = 'inferno')
# pl.title('difference in abs value')
# pl.show()

#%%
"""
*** Display plot for coronagraphic images
"""

### Monochromatic image profiles

xi2d = corono2d.xi2d[:nImg2d//2]
Cormono_2d_vec = Cormono_2d[nImg2d//2,nImg2d//2:]
mono_corono_image0_vec = mono_corono_image0[nlam//2,nImg2d//2,nImg2d//2:]

pl.figure(3, (8, 4.5))
pl.clf()
pl.semilogy(xi, Cormono_1d, label='Math 1d - mono')
#pl.semilogy(xi2d, Cormono_2d_vec, label='Math 2d - mono')
pl.semilogy(xi2d, mono_corono_image0_vec, label='Python 2d - mono')
# pl.semilogy(xi2d, Cormono_2d_vec, label='Math 2d - mono')
# pl.loglog(xi[1:], Cormono_1d[1:], label='Math 1d - mono')
# pl.loglog(xi2d[1:], mono_corono_image0_vec[1:], label='Python 2d - mono')
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.5, 50.5)
pl.ylim(10**(-12.2), 10**(-3.8))
pl.legend()
pl.tight_layout()

#%%
### Broadband image profiles

# xi2d = corono2d.xi2d[:nImg2d//2]
Corpoly_2d_vec = Corpoly_2d[nImg2d//2,nImg2d//2:]
poly_corono_image0_vec = poly_corono_image0[nImg2d//2,nImg2d//2:]

pl.figure(4, (8, 4.5))
pl.clf()
pl.semilogy(xi, Corpoly_1d, label='Math 1d - poly')
#pl.semilogy(xi2d, Corpoly_2d_vec, label='Math 2d - poly')
pl.semilogy(xi2d, poly_corono_image0_vec, label='Python 2d - poly')
# pl.semilogy(xi2d, Corpoly_2d_vec, label='Math 2d - poly')
# pl.loglog(xi[1:], Corpoly_1d[1:], label='Math 1d - poly')
# pl.loglog(xi2d[1:], poly_corono_image0_vec[1:], label='Python 2d - poly')
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.5, 50.5)
pl.ylim(10**(-12.2), 10**(-3.8))
pl.legend()
pl.tight_layout()

#%%
"""
*** Display plot for direct images
"""

### Monochromatic image profiles

xi2d = corono2d.xi2d[:nImg2d//2]
#Cormono_2d_vec = Cormono_2d[nImg2d//2,nImg2d//2:]
mono_direct_image0_vec = mono_direct_image0[nlam//2,nImg2d//2,nImg2d//2:]

pl.figure(5, (8, 4.5))
pl.clf()
pl.semilogy(xi, Psfmono_1d, label='Math 1d - mono')
#pl.semilogy(xi2d, Cormono_2d_vec, label='Math 2d - mono')
pl.semilogy(xi2d, mono_direct_image0_vec, label='Python 2d - mono')
# pl.semilogy(xi2d, Cormono_2d_vec, label='Math 2d - mono')
# pl.loglog(xi[1:], Cormono_1d[1:], label='Math 1d - mono')
# pl.loglog(xi2d[1:], mono_corono_image0_vec[1:], label='Python 2d - mono')
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.5, 50.5)
pl.ylim(10**(-8.2), 10**(0.8))
pl.legend()
pl.tight_layout()

#%%
### Broadband image profiles

# xi2d = corono2d.xi2d[:nImg2d//2]
# Psfpoly_2d_vec = Psfpoly_2d[nImg2d//2,nImg2d//2:]
poly_direct_image0_vec = poly_direct_image0[nImg2d//2,nImg2d//2:]

pl.figure(6, (8, 4.5))
pl.clf()
pl.semilogy(xi, Psfpoly_1d, label='Math 1d - poly')
#pl.semilogy(xi2d, Corpoly_2d_vec, label='Math 2d - poly')
pl.semilogy(xi2d, poly_direct_image0_vec, label='Python 2d - poly')
# pl.semilogy(xi2d, Corpoly_2d_vec, label='Math 2d - poly')
# pl.loglog(xi[1:], Corpoly_1d[1:], label='Math 1d - poly')
# pl.loglog(xi2d[1:], poly_corono_image0_vec[1:], label='Python 2d - poly')
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.5, 50.5)
pl.ylim(10**(-8.2), 10**(0.8))
pl.legend()
pl.tight_layout()

