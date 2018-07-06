#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

@author: mndiaye
"""

import pylab as pl
from pathlib import Path

from corono import corono_design as cd
#from corono import corono_optim_2d as co2d

from corono.utils import to_dict

from matplotlib import cm

from astropy.io import fits

import numpy as np

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

#%% parameters
"""
Parameters
"""
# Telescope name
pupil_name = 'sbr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxTau' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
solver       = 'stdgrb' # 'gurobipy' #  'gurobipy', 'scipy.linprog'

#nPup = corono0.params['nPup']
nPup = 206

Fmax2d = nPup/2 
nImg2d = 512

# mask radius in lam0/D unit
rMask = 4.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  4.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 4

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
corono_name   = 'SP' # 'SP' or 'APLC'
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True

#nlam
nlam = 5
bw   = 0.2

do_fits = True

#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = Path('./pupils/2D/').resolve()
if pupil_name == 'lvr':
    fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
    fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
else:
    fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)

fpath_pup = fdir / fname_pup
fpath_lys = fdir / fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)


params = to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2=CtrBtwnPix2, 
                 nlam=nlam, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, solver = solver)

#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'SP':
    corono0 = cd.SP2d(**params)
elif corono_name == 'APLC':
    corono0 = cd.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Read files
"""
fdir = Path('./results/2D/dat_pyth').resolve()

if corono_name == 'SP':
    fname_gen  = 'SP00_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}_{problem_name}_{solver}.fits'
else:
    fname_gen  = 'APLC_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}_{problem_name}_{solver}.fits'

fpath = fdir / pupil_name / fname_gen.format(**{key: corono0.params[key] for key in corono0.params})

print('{0}'.format(fpath))

Apod1_2d = fits.getdata(fpath,)

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
pl.figure(4)
pl.clf()
pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil transmission')

fdir_pdf = Path('./results/2D/plots/').resolve()
fname = '{0}_apodisation.pdf'.format(pupil_name)
fpath = fdir_pdf / fname
pl.figure(5)
pl.clf()
pl.imshow(Apod1_2d*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 transmission - MaxTau problem')
pl.savefig(str(fpath))

Pupil2d_bis = np.zeros((nImg2d, nImg2d))
Pupil2d_bis[nImg2d//2-nPup//2:nImg2d//2+nPup//2,nImg2d//2-nPup//2:nImg2d//2+nPup//2] = corono0.Pupil2d 

Apod1_2dbis = np.zeros((nImg2d, nImg2d))
Apod1_2dbis[nImg2d//2-nPup//2:nImg2d//2+nPup//2,nImg2d//2-nPup//2:nImg2d//2+nPup//2] = Apod1_2d 

#fpath = '/users/mndiaye/Desktop/apod1.pdf'
pl.figure(20)
pl.imshow(Apod1_2dbis)
#pl.savefig(fpath)

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""

nlam = 1

params = to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2=CtrBtwnPix2, 
                 nlam=nlam, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, solver = solver)

if corono_name == 'SP':
    corono0 = cd.SP2d(**params)
else:
    corono0 = cd.APLC2d(**params)

#if corono_name == 'APLC':
#    poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod1_2d)
#else:
#    poly_direct_image1 = corono0.compute_direct_intensity_2d(corono0.Pupil2d)
#poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod1_2d)

poly_direct_image1 = np.abs(shift(fft(shift(Pupil2d_bis))))**2
poly_direct_image1 /= np.max(poly_direct_image1) 

poly_corono_image1 = np.abs(shift(fft(shift(Apod1_2dbis))))**2
poly_corono_image1 /= np.max(poly_corono_image1) 

#%% image plot
"""
Display direct and coronagraphic images
"""
fname = '{0}_direct_image.pdf'.format(pupil_name)
fpath = fdir_pdf / fname
pl.figure(10)
pl.clf()
pl.imshow(poly_direct_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - direct image')
pl.savefig(str(fpath))

fname  = '{0}_apodized_image.pdf'.format(pupil_name)
fpath = fdir_pdf / fname
pl.figure(11)
pl.clf()
pl.imshow(poly_corono_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - apodized image')
pl.savefig(str(fpath))

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

nImg2d = corono0.params['nImg2d']
fname = '{0}_intensity_profiles.pdf'.format(pupil_name)
fpath = fdir_pdf / fname

pl.figure(8)
pl.clf()
pl.title('Radial intensity profiles of the images')
pl.semilogy(corono0.xi2d[:nImg2d//2],poly_direct_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
if corono_name == 'SP':
    pl.semilogy(corono0.xi2d[:nImg2d//2],poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_corono_image1.max(),label='Apod')
else:
    pl.semilogy(corono0.xi2d[:nImg2d//2],poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='MaxTau')
    
#pl.semilogy(corono0.xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
#pl.semilogy(corono0.xi2d,poly_corono_image3[nImg2d//2,nImg2d//2:]/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
#pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()
pl.savefig(str(fpath))

#%%
pl.show()
