#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

@author: mndiaye
"""

import pylab as pl
from pathlib import Path
from matplotlib import cm
from astropy.io import fits
import corono as coro

#%% parameters
"""
Parameters
"""
# Telescope name
pupil_name = 'lvr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxContrastLinf' # 'MaxContrastL1' # 'MaxTau' # ,'MaxContrastL1' # #  
solver       = 'stdgrb' #,'gurobipy' #  'gurobipy', 'scipy.linprog'

import os

#nPup = corono0.params['nPup']
nPup = 600

Fmax2d = 44 
nImg2d = 220

# mask radius in lam0/D unit
rMask = 4.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  3.5
rho1 = 10.0

# contrast in the dark region
cDarkHole = 10

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
corono_name   = 'APLC' # 'SP' or 'APLC'
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True

#nlam
nlam = 3
nlambis = 11
bw   = 0.1

do_fits = True

#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = Path('../../pupils/2D/').resolve()
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

params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name)

#%%
"""
Working directories
"""
fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name

fdir_pdf = Path('../../results/2D/plots/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

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

#%%
"""
Read files
"""
fname_gen = problem1.get_filename()
fname     = fname_gen + '.fits'
fpath     = fdir / fname

Apod1_2d = fits.getdata(fpath,)

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
pl.figure(4)
pl.clf()
pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil transmission')

fname = fname_gen + '_apodisation.pdf'
fpath = fdir_pdf / fname

pl.figure(5)
pl.clf()
pl.imshow(Apod1_2d*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 transmission - MaxTau problem')
pl.savefig(str(fpath))

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis) 

if corono_name == 'APLC':
    poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod1_2d)
else:
    poly_direct_image1 = corono0.compute_direct_intensity_2d(corono0.Pupil2d)
poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod1_2d)

#%% image plot
"""
Display direct and coronagraphic images
"""
fname = fname_gen + '_direct_image.pdf'
fpath = fdir_pdf / fname

pl.figure(10)
pl.clf()
pl.imshow(poly_direct_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - direct image')
pl.savefig(str(fpath))

fname  = fname_gen + '_apodized_image.pdf'
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
fname = fname_gen + '_intensity_profiles.pdf'
fpath = fdir_pdf / fname

pl.figure(8)
pl.clf()
pl.title('Radial intensity profiles of the images')
pl.semilogy(corono0.xi2d,poly_direct_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
if corono_name == 'SP':
    pl.semilogy(corono0.xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_corono_image1.max(),label='Apod')
else:
    pl.semilogy(corono0.xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='MaxTau')
    
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
