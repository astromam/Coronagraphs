#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:12:00 2019

@author: mndiaye
"""

import numpy as np
import pylab as pl
from pathlib import Path
from pyzelda.utils import imutils

import os
from matplotlib import cm
from astropy.io import fits
import corono as coro

from scipy.misc import imresize

import time

#%% parameters
"""
### Parameters
"""
pl.close('all')
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'vlt' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxContrastL1' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
    solver       = 'stdgrb' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    FirstDerGlobalLim = 1.
    
    #nPup = corono0.params['nPup']
    nPup = 384
    nFPM = 50
    Fmax2d = 50
    nImg2d = 500

    # central wavelength (H2 filter)
    wv = 1.593e-6
    
    # mask radius in lam0/D unit
    dAper     = 8
    mas2rad   = np.pi/(180.*3600)
    rMask_m   = 287e-6/2.#372e-6/2.#
    Fratio    = 40
    rMask  = rMask_m/(wv*Fratio)
    print('Mask radius: {0:.3f} lambda_0/D at {1:.3f}um'.format(rMask, wv*1e6))
    rMask_mas = 1000.*rMask * (wv/dAper)/mas2rad
    print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  2.0
    rho1 = 20.0
    
    # contrast in the dark region
    cDarkHole = 6.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.756
    
    # CtrBtwnPix2
    CtrBtwnPix  = True
    CtrBtwnPix2 = True
    Pupil2dSym  = False # set it True only for optimization
    
    #nlam
    bw   = 0.2
    nlam = 5

    
    do_fits = True


nlambis = 11  
nImg2dbis = 200  
Fmax2dbis = (nImg2dbis/2)*(950e-9/wv)
   

# total number of existing maps
qmap = 1000
# total number of used maps
nmap = 1

# plot parameters
vmin0 = -8
vmax0 = 0

# PSD number
iPSD = 0

#%%  
""" 
### Coronagraph defintion
"""
params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
#                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)

if corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
### Problem defintion
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
### File reading path for Pupil, Apodizer, Lyot stop and AO residuals
"""
# Pupil
if True:
    fdir0 = Path('../../../data/2D/pupils/').resolve()
    if pupil_name == 'vlt':
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
        fname_lys = 'SPHERE/sphere_stop_ST_ALC2.fits' 
    else:
        raise ValueError('pupil_name should be vlt instead of {0}'.format(pupil_name))
    
    fpath_pup = fdir0 / fname_pup
    fpath_lys = fdir0 / fname_lys
    Pupil2d    = fits.getdata(fpath_pup)

# Lyot stop    
    LyotStop2dtmp = fits.getdata(fpath_lys)
    if nPup != 384:
        LyotStop2d = imresize(LyotStop2dtmp, (nPup, nPup))
    else:
        LyotStop2d = LyotStop2dtmp*1
    
    if solver != 'gurobipy' and solver != 'stdgrb':
        solver = 'scipy'

# Apodizer 
fdir_apod  = Path('../../../data/2D/pupils/SPHERE/').resolve()
fname      = 'SPHERE_APO1_field_transmission_map.fits'
fpath_apod = fdir_apod / fname

# AO residuals
fdir_opd = Path('../../../data/2D/AO_tests/').resolve()
fname = 'AOres_opd_nPup={0}_iPSD={1:04d}_nmap={2:04d}.fits'.format(nPup,iPSD,qmap)
fpath_opd = fdir_opd / fname 

#%%
"""
### File saving path for the generated data
"""
fdir_pdf = Path('../../../results/2D/plots/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

fdir_data = Path('../../../results/2D/data/AO_tests/aplc1/').resolve()
if not os.path.exists(fdir_data):
    os.makedirs(fdir_data)

fname_direct = 'aplc1_direct_nPup={0}_nImg={1}_img.fits'.format(nPup, nImg2dbis)
fname_corono = 'aplc1_corono_nPup={0}_nImg={1}_img.fits'.format(nPup, nImg2dbis)
fpath_direct = fdir_data / fname_direct
fpath_corono = fdir_data / fname_corono

fname_direct_AO = 'aplc1_direct_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_img.fits'.format(nPup, nImg2dbis, iPSD, nmap)
fname_corono_AO = 'aplc1_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_img.fits'.format(nPup, nImg2dbis, iPSD, nmap)
fpath_direct_AO = fdir_data / fname_direct_AO
fpath_corono_AO = fdir_data / fname_corono_AO

# with no aberrations
fname_direct_avg = 'aplc1_direct_nPup={0}_nImg={1}_avg.fits'.format(nPup, nImg2dbis)
fname_corono_avg = 'aplc1_corono_nPup={0}_nImg={1}_avg.fits'.format(nPup, nImg2dbis)
fname_direct_std = 'aplc1_direct_nPup={0}_nImg={1}_std.fits'.format(nPup, nImg2dbis)
fname_corono_std = 'aplc1_corono_nPup={0}_nImg={1}_std.fits'.format(nPup, nImg2dbis)

fpath_direct_avg = fdir_data / fname_direct_avg
fpath_corono_avg = fdir_data / fname_corono_avg
fpath_direct_std = fdir_data / fname_direct_std
fpath_corono_std = fdir_data / fname_corono_std

# with AO residuals
fname_direct_avg_AO = 'aplc1_direct_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_avg.fits'.format(nPup, nImg2dbis, iPSD, nmap)
fname_corono_avg_AO = 'aplc1_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_avg.fits'.format(nPup, nImg2dbis, iPSD, nmap)
fname_direct_std_AO = 'aplc1_direct_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_std.fits'.format(nPup, nImg2dbis, iPSD, nmap)
fname_corono_std_AO = 'aplc1_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_std.fits'.format(nPup, nImg2dbis, iPSD, nmap)

fpath_direct_avg_AO = fdir_data / fname_direct_avg_AO
fpath_corono_avg_AO = fdir_data / fname_corono_avg_AO
fpath_direct_std_AO = fdir_data / fname_direct_std_AO
fpath_corono_std_AO = fdir_data / fname_corono_std_AO

#%%
"""
### Read Apodizer
"""
Apod2d = fits.getdata(fpath_apod,)

#%% Signal in intensity
"""
### Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                                nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis) 

if corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

direct_poly_img = corono0.compute_direct_intensity_2d(Apod2d)
corono_poly_img = corono0.compute_corono_intensity_2d(Apod2d)

pk = np.max(direct_poly_img)
direct_poly_img /= pk
corono_poly_img /= pk

#%%
"""
### Read phase screens
"""
opd_arr = fits.getdata(fpath_opd)

#%%
"""
### Computation of the direct and coronagraphic images with AO screens
"""
direct_poly_img_AO = np.zeros((nImg2dbis, nImg2dbis))
corono_poly_img_AO = np.zeros((nImg2dbis, nImg2dbis))

paramsAO = coro.update_params(params2, Pupil2d = corono0.Pupil2d, 
                              LyotStop2d = corono0.LyotStop2d)
coronoAO = coro.design.APLC2d(**paramsAO)

t0 = time.time()
for imap in range(nmap):
    if (imap+1) % 10 == 0:
        print('imap {0:04d}/{1:04d}'.format(imap+1,nmap))
    direct_poly_img_AO += coronoAO.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d = opd_arr[imap])
    corono_poly_img_AO += coronoAO.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d = opd_arr[imap])
t1 = time.time()
print('time: {0:.2f}s'.format(t1-t0))
print('iteration average time: {0:.2f}s'.format((t1-t0)/nmap))
    
# Normalization
pk_AO = np.max(direct_poly_img_AO)
direct_poly_img_AO /= pk_AO
corono_poly_img_AO /= pk_AO

#%%
"""
### Computation contrast curves with no aberration
"""
direct_poly_avg, rad_direct = imutils.profile(direct_poly_img, type='mean')
corono_poly_avg, rad_corono = imutils.profile(corono_poly_img, type='mean')
direct_poly_std, rad_direct = imutils.profile(direct_poly_img, type='std')
corono_poly_std, rad_corono = imutils.profile(corono_poly_img, type='std')

#%%
"""
### Compute contrast curves with AO residuals
"""
direct_poly_avg_AO, rad_direct = imutils.profile(direct_poly_img_AO, type='mean')
corono_poly_avg_AO, rad_corono = imutils.profile(corono_poly_img_AO, type='mean')
direct_poly_std_AO, rad_direct = imutils.profile(direct_poly_img_AO, type='std')
corono_poly_std_AO, rad_corono = imutils.profile(corono_poly_img_AO, type='std')

#%%
"""
### Save image (no aberration)
"""
fits.writeto(fpath_direct, direct_poly_img, overwrite=True)
fits.writeto(fpath_corono, corono_poly_img, overwrite=True)

#%%
"""
### Save image (with AO residuals)
"""
fits.writeto(fpath_direct_AO, direct_poly_img_AO, overwrite=True)
fits.writeto(fpath_corono_AO, corono_poly_img_AO, overwrite=True)

#%%
"""
### Save profiles (with AO residuals)
"""
# with no aberrations
fits.writeto(fpath_direct_avg, direct_poly_avg, overwrite=True)
fits.writeto(fpath_corono_avg, corono_poly_avg, overwrite=True)
fits.writeto(fpath_direct_std, direct_poly_std, overwrite=True)
fits.writeto(fpath_corono_std, corono_poly_std, overwrite=True)

# with AO residuals
fits.writeto(fpath_direct_avg_AO, direct_poly_avg_AO, overwrite=True)
fits.writeto(fpath_corono_avg_AO, corono_poly_avg_AO, overwrite=True)
fits.writeto(fpath_direct_std_AO, direct_poly_std_AO, overwrite=True)
fits.writeto(fpath_corono_std_AO, corono_poly_std_AO, overwrite=True)

#%% Display of the apodizer
"""
### Display pupil and apodizer
"""
pl.figure(0)
pl.clf()
pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil transmission')

fname = fname_gen + '_apodisation_ampl_nPup={0}.pdf'.format(nPup)
fpath = fdir_pdf / fname

#pl.figure(1)
#pl.clf()
#pl.imshow(Apod2d*corono0.Pupil2d, cmap = cm.Greys_r)
#pl.title('Apod 1 transmission - MaxTau problem - '+ solver)
#pl.savefig(str(fpath))

pl.figure(1)
pl.clf()
pl.imshow(Apod2d*corono0.Pupil2d, cmap = 'inferno')
pl.title('Apodized entrance pupil')
pl.tight_layout()
#pl.savefig(str(fpath), transparent=True)


#%%
"""
### Image display (perfect)
"""
pl.figure(2)
pl.clf()
pl.imshow(np.log10(direct_poly_img), 
          vmin=vmin0, vmax=vmax0, cmap = 'inferno')
pl.title('Direct image (no aberration)')

pl.figure(3)
pl.clf()
pl.imshow(np.log10(corono_poly_img), 
          vmin=vmin0, vmax=vmax0, cmap = 'inferno')
pl.title('Corono image (no aberration)')


#%%
"""
### Image display (AO residuals)
"""
pl.figure(10)
pl.clf()
pl.imshow(np.log10(direct_poly_img_AO), vmin=vmin0, vmax=vmax0, cmap = 'inferno')
pl.title('Direct image with AO residuals')

pl.figure(11)
pl.clf()
pl.imshow(np.log10(corono_poly_img_AO), vmin=vmin0, vmax=vmax0, cmap = 'inferno')
pl.title('Corono image with AO residuals')


#%%
"""
### Display contrast curves
"""
fname_image_plane_plot = 'aplc1_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_plt.pdf'.format(nPup, nImg2dbis, iPSD, nmap)
fpath_image_plane_plot = fdir_pdf / fname_image_plane_plot

rad_corono = np.arange(nImg2dbis//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,2))

pl.figure(12)
pl.clf()
pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_std/direct_poly_img.max(),
        label='no turbulence', color = colors_cor[0])

pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_std_AO,
        label='AO residuals', color = colors_cor[1])

pl.axvspan(-1, rMask, alpha=0.25, color='b')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'5$\sigma$ normalized intensity in log scale')
pl.xlim(-0.5, 30.5)
pl.ylim(3e-8, 3e-4)
pl.grid(True, which='both')

pl.legend()
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
pl.tight_layout()
pl.savefig(str(fpath_image_plane_plot), transparent=True)

pl.show()
