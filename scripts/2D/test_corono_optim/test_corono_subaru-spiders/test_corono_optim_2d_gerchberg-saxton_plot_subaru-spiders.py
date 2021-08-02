#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

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
    nPup = 200
    nFPM = 50
    Fmax2d = 50
    nImg2d = 500

    # telescope parameters
    pdiam, odiam = 7.92, 2.3  # tel. and obst. diameters (meters)
    thick = 0.25              # adopted spider thickness (meters)
    offset = 1.278            # spider intersection offset (meters)
    beta = 51.75              # spider angle beta
    
    fac = 2.0
    odiam2 = fac*odiam
    thick2 = fac*thick
    Fratio    = 64

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
width_Y = 1889.08e-10
bw_Y  = width_Y/wv0_Y
rMask_Y = rMask_m/(wv0_Y*Fratio) 

wv0_J   = 12317.58e-10
if fac == 1.0:
    wv1_J = 1.2385776e-06
elif fac == 1.5:
    wv1_J = 1.2385776e-06
elif fac == 2.0:
    wv1_J = 1.2385776e-06
else:
    wv1_J   = (1.72/1.65)*wv0_J
width_J = 2273.20e-10
bw_J  = width_J/wv0_J
rMask_J = rMask_m/(wv0_J*Fratio) 

wv0_H   = 16444.09e-10
if fac == 1.0:
    wv1_H   = 1.61754562e-06
elif fac == 1.5:
    wv1_H   = 1.63843936e-06
elif fac == 2.0:
    wv1_H   = 1.6891813e-06
else:
    (1.72/1.65)*wv0_H
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
    wv1 = wv1_J
    width = width_J
elif band == 'GPI_H':
    wv0 = wv0_H
    wv1 = wv1_H
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

rMask1    = rMask_m/(wv1*Fratio)  # mask size in lam1/D


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

    if pupil_name == 'lvr':
        fname_pup = f'ATLAST_Aperture_nPup={nPup}.fits'
        fname_lys = f'ATLAST_LyotStop_nPup={nPup}.fits'
    elif pupil_name == 'sbr':
        fname_pup = f'pupil=sbr_nPup={nPup}_odiam={int(odiam*100)}_thick={int(thick*100):03d}.fits'
        fname_lys = f'pupil=sbr_nPup={nPup}_odiam={int(odiam2*100)}_thick={int(thick2*100):03d}.fits'
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
Working directories
"""
#fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
    elif syst == 'linux':
        fdir = Path('/scratch/mndiaye/data/Coronagraphs/results/2D/dat_pyth/').resolve()
    else:
        raise ValueError('Unknown operating system {0}'.format(syst))
else:
    raise ValueError('Unknown user {0}'.format(user))

fdir_pdf = Path('../../results/2D/plots/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

#%%  
""" 
Coronagraph defintion
"""
params1    = coro.update_params(params, rMask=rMask1) 

if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params)
    corono1 = coro.design.SP2d(**params1)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
    corono1 = coro.design.APLC2d(**params1)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_2d.MaxTau(corono=corono1, **params1)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono1, Lnorm='L1',**params1)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono1, Lnorm='Linf',**params1)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#%%
"""
Read files
"""
fname_gen = problem1.get_filename()
fname     = fname_gen + f'_{band}band_gbsx.fits'
fpath_apod= fdir / fname

Apod_pyth = fits.getdata(fpath_apod,)

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
# pl.figure(4)
# pl.clf()
# pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
# pl.title('Pupil transmission')

fname = fname_gen + '_apodisation_ampl_gbsx.pdf'
fpath = fdir_pdf / fname

pl.figure(5, (12,4))
pl.clf()
pl.subplot(131)
pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil')
pl.subplot(132)
pl.imshow(Apod_pyth*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 transmission \n Gerchberg-Saxton')
pl.subplot(133)
pl.imshow(corono0.LyotStop2d, cmap = cm.Greys_r)
pl.title('Lyot Stop')
pl.savefig(str(fpath))

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis) 

if corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params2)
elif corono_name == 'SP':
    corono0 = coro.design.SP2d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

if corono_name == 'APLC':
    poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod_pyth)
else:
    poly_direct_image1 = corono0.compute_direct_intensity_2d(corono0.Pupil2d)
poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod_pyth)

#%%

params_z    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, 
                                 nImg2d = nImg2dbis, bw=bw_z, rMask = rMask_z) 
params_Y    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, 
                                 nImg2d = nImg2dbis, bw=bw_Y, rMask = rMask_Y)
params_J    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, 
                                 nImg2d = nImg2dbis, bw=bw_J, rMask = rMask_J) 
params_H    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, 
                                 nImg2d = nImg2dbis, bw=bw_H, rMask = rMask_H) 

if corono_name == 'APLC':
    corono0_z = coro.design.APLC2d(**params_z)
    corono0_Y = coro.design.APLC2d(**params_Y)
    corono0_J = coro.design.APLC2d(**params_J)
    corono0_H = coro.design.APLC2d(**params_H)
elif corono_name == 'SP':
    corono0_z = coro.design.SP2d(**params_z)
    corono0_Y = coro.design.SP2d(**params_Y)
    corono0_J = coro.design.SP2d(**params_J)
    corono0_H = coro.design.SP2d(**params_H)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

if corono_name == 'APLC':
    poly_direct_image1_z = corono0_z.compute_direct_intensity_2d(Apod_pyth)
    poly_direct_image1_Y = corono0_Y.compute_direct_intensity_2d(Apod_pyth)
    poly_direct_image1_J = corono0_J.compute_direct_intensity_2d(Apod_pyth)
    poly_direct_image1_H = corono0_H.compute_direct_intensity_2d(Apod_pyth)
else:
    poly_direct_image1_z = corono0_z.compute_direct_intensity_2d(corono0_z.Pupil2d)
    poly_direct_image1_Y = corono0_Y.compute_direct_intensity_2d(corono0_Y.Pupil2d)
    poly_direct_image1_J = corono0_J.compute_direct_intensity_2d(corono0_J.Pupil2d)
    poly_direct_image1_H = corono0_H.compute_direct_intensity_2d(corono0_H.Pupil2d)
poly_corono_image1_z = corono0_z.compute_corono_intensity_2d(Apod_pyth)
poly_corono_image1_Y = corono0_Y.compute_corono_intensity_2d(Apod_pyth)
poly_corono_image1_J = corono0_J.compute_corono_intensity_2d(Apod_pyth)
poly_corono_image1_H = corono0_H.compute_corono_intensity_2d(Apod_pyth)

# normalization
poly_direct_image1_z_pk = poly_direct_image1_z.max()
poly_direct_image1_Y_pk = poly_direct_image1_Y.max()
poly_direct_image1_J_pk = poly_direct_image1_J.max()
poly_direct_image1_H_pk = poly_direct_image1_H.max()

poly_direct_image1_z /= poly_direct_image1_z_pk
poly_direct_image1_Y /= poly_direct_image1_Y_pk
poly_direct_image1_J /= poly_direct_image1_J_pk
poly_direct_image1_H /= poly_direct_image1_H_pk

poly_corono_image1_z /= poly_direct_image1_z_pk
poly_corono_image1_Y /= poly_direct_image1_Y_pk
poly_corono_image1_J /= poly_direct_image1_J_pk
poly_corono_image1_H /= poly_direct_image1_H_pk

# azithally averaged intensity profiles
poly_corono_prf_avg_z, rad_corono_z = imutils.profile(poly_corono_image1_z, type='mean')
poly_corono_prf_avg_Y, rad_corono_Y = imutils.profile(poly_corono_image1_Y, type='mean')
poly_corono_prf_avg_J, rad_corono_J = imutils.profile(poly_corono_image1_J, type='mean')
poly_corono_prf_avg_H, rad_corono_H = imutils.profile(poly_corono_image1_H, type='mean')

poly_corono_prf_std_z, rad_corono_z = imutils.profile(poly_corono_image1_z, type='std')
poly_corono_prf_std_Y, rad_corono_Y = imutils.profile(poly_corono_image1_Y, type='std')
poly_corono_prf_std_J, rad_corono_J = imutils.profile(poly_corono_image1_J, type='std')
poly_corono_prf_std_H, rad_corono_H = imutils.profile(poly_corono_image1_H, type='std')


#%% image plot
"""
Display direct and coronagraphic images
"""
fname = fname_gen + '_direct_image_gbsx.pdf'
fpath = fdir_pdf / fname

pl.figure(10)
pl.clf()
pl.imshow(poly_direct_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - direct image')
pl.savefig(str(fpath))

bands = ['z', 'Y', 'J', 'H']
nband = len(bands)

vmin0 = -8
vmax0 = -3

fname = fname_gen + '_apodized_image_gbsx.pdf'
fpath = fdir_pdf / fname

f1 = pl.figure(11, (16, 4.5))
pl.clf()

ax1 = f1.add_subplot(141)
im = ax1.imshow(np.log10(poly_corono_image1_z), cmap = cm.inferno, vmin=vmin0,vmax=vmax0)
ax1.set_title(f'{corono_name}, {bands[0]} band')

ax2 = f1.add_subplot(142)
im = ax2.imshow(np.log10(poly_corono_image1_Y), cmap = cm.inferno, vmin=vmin0,vmax=vmax0)
ax2.set_title(f'{corono_name}, {bands[1]} band')

ax3 = f1.add_subplot(143)
im = ax3.imshow(np.log10(poly_corono_image1_J), cmap = cm.inferno, vmin=vmin0,vmax=vmax0)
ax3.set_title(f'{corono_name}, {bands[2]} band')

ax4 = f1.add_subplot(144)
im = ax4.imshow(np.log10(poly_corono_image1_H), cmap = cm.inferno, vmin=vmin0,vmax=vmax0)
ax4.set_title(f'{corono_name}, {bands[3]} band')

cbar_ax = f1.add_axes([0.92, 0.15, 0.03, 0.7])
cbar = f1.colorbar(im, cax = cbar_ax)
cbar.ax.set_ylabel("intensity in log scale", rotation=270, labelpad = 10)
pl.savefig(str(fpath))

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

#val = 0
#if CtrBtwnPix2 is True:
#    val = 1/2

#val = 0
#if nImg2d%2 == 0:
#    val = 1/2
#
#xx,yy  = np.meshgrid(np.arange(nImg2d)-nImg2d//2+val, np.arange(nImg2d)-nImg2d//2+val)
#mydist = (Fmax2d/nImg2d)*np.hypot(yy,xx)
#xi2d = mydist[nImg2d//2,nImg2d//2:]

xi2d = corono0.xi2d
if nImg2dbis%2 == 0:
    xi2d = corono0.xi2d_ctr

nImg2d = corono0.params['nImg2d']




poly_direct_image1_arr = [poly_direct_image1_z, poly_direct_image1_Y, 
              poly_direct_image1_J, poly_direct_image1_H]
poly_corono_image1_arr = [poly_corono_image1_z, poly_corono_image1_Y, 
              poly_corono_image1_J, poly_corono_image1_H]

rad_corono_arr = np.asarray([rad_corono_z, rad_corono_Y,
                  rad_corono_J, rad_corono_H])*Fmax2dbis/nImg2dbis

poly_corono_prf_avg_arr = [poly_corono_prf_avg_z, poly_corono_prf_avg_Y,
                           poly_corono_prf_avg_J, poly_corono_prf_avg_H]

poly_corono_prf_std_arr = [poly_corono_prf_std_z, poly_corono_prf_std_Y,
                           poly_corono_prf_std_J, poly_corono_prf_std_H]
poly_corono_prf_stdpl1_arr = np.asarray(poly_corono_prf_avg_arr)+np.asarray(poly_corono_prf_std_arr)
poly_corono_prf_stdpl3_arr = np.abs(np.asarray(poly_corono_prf_avg_arr)+3*np.asarray(poly_corono_prf_std_arr))


rMask_arr = [corono0_z.rMask, corono0_Y.rMask, corono0_J.rMask, corono0_H.rMask]

for iband in range(nband):
    
    fname = fname_gen + f'_intensity_profiles_{bands[iband]}_gbsx.pdf'
    fpath = fdir_pdf / fname
    pl.figure(21+iband)
    pl.clf()
    pl.title(f'Azimith averaged intensity profiles of the images in {bands[iband]} band')
    pl.semilogy(rad_corono_arr[iband], poly_corono_prf_avg_arr[iband], ls='-', label='avg')
    pl.semilogy(rad_corono_arr[iband], poly_corono_prf_stdpl1_arr[iband], ls='--', label=r'avg+1$\sigma$')
    pl.semilogy(rad_corono_arr[iband], poly_corono_prf_stdpl3_arr[iband], ls='-.', label=r'avg+3$\sigma$')
    
    pl.axvline(x=rMask_arr[iband], ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
    pl.xlabel(r'Angular separation in $\lambda_0$/D')
    pl.ylabel('Normalized intensity in log scale')
    pl.ylim(1e-8, 1e-2)
    pl.legend()
    pl.tight_layout()
    pl.savefig(str(fpath))

#%%
pl.show()

values = range(nlambis)
colors = pl.cm.rainbow(np.linspace(0,1,nlambis))

mono_direct_image1 = corono0.compute_direct_intensity_2d(Apod_pyth, poly=False)
mono_corono_image1 = corono0.compute_corono_intensity_2d(Apod_pyth, poly=False)



pl.figure(41)
pl.clf()
pl.title('Radial intensity profiles of the images')
#pl.semilogy(corono0.xi2d,poly_direct_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
for i in range(corono0.nlam):
    if corono_name == 'SP':
        pl.semilogy(xi2d,mono_corono_image1[i, nImg2dbis//2,nImg2dbis//2:]/mono_corono_image1.max(), '-',
                    label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[i]),
                    color = colors[i])
    else:
        pl.semilogy(xi2d,mono_corono_image1[i, nImg2dbis//2,nImg2dbis//2:]/mono_direct_image1[(corono0.nlam+1)//2].max(), 
                    '-',
                    label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[i]),  color = colors[i])
#pl.semilogy(corono0.xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
#pl.semilogy(corono0.xi2d,poly_corono_image3[nImg2d//2,nImg2d//2:]/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-9, 2e0)
pl.legend()
pl.tight_layout()


#%%
"""
Robustness to spectral bandwidth
"""   
nlam_ter = 101
bw_ter   = 0.92
   
fname_gen  = problem1.get_filename(nlam=nlambis)
params3    = coro.update_params(params, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis, 
                                nlam = nlam_ter, bw = bw_ter)

if corono_name == 'SP':
    corono3 = coro.design.SP2d(**params3)
elif corono_name == 'APLC':
    corono3 = coro.design.APLC2d(**params3)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

if corono_name == 'APLC':
    direct_poly_img_f3 = corono3.compute_direct_intensity_2d(Apod_pyth)
    direct_mono_img_t3 = corono3.compute_direct_intensity_2d(Apod_pyth, poly=False)
elif corono_name == 'SP':
    direct_poly_img_f3 = corono3.compute_direct_intensity_2d(corono0.Pupil2d)
    direct_mono_img_t3 = corono3.compute_direct_intensity_2d(corono0.Pupil2d, poly=False)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
corono_poly_img_f3 = corono3.compute_corono_intensity_2d(Apod_pyth)
corono_mono_img_t3 = corono3.compute_corono_intensity_2d(Apod_pyth, poly=False)


#%%
val = 0
if nImg2dbis%2 == 0:
    val = 1/2

sepbis=3.0
septer=6.0

# array of angular distances in the final image plane
xx,yy  = np.meshgrid(np.arange(nImg2dbis)-nImg2dbis//2+val, np.arange(nImg2dbis)-nImg2dbis//2+val)
mydist = (Fmax2dbis/nImg2dbis)*np.hypot(yy,xx)        
resbis = (mydist <= sepbis +0.5)*(mydist >= sepbis -0.5)
rester = (mydist <= septer +0.5)*(mydist >= septer -0.5)

#%%
corono_poly_avg_resbis_wv_t = []
corono_poly_avg_rester_wv_t = []

for i in range(nlam_ter):
    corono_poly_avg_resbis_wv_t.append(np.mean(corono_mono_img_t3[i, resbis != 0])/direct_mono_img_t3[(nlam_ter-1)//2].max())
    corono_poly_avg_rester_wv_t.append(np.mean(corono_mono_img_t3[i, rester != 0])/direct_mono_img_t3[(nlam_ter-1)//2].max())
    


#%%
colors_shifts = pl.cm.rainbow(np.linspace(0,1,2))
ls_shifts = ["-", "--"]

fname_bw_plot = 'corono_poly_bw_sensitivity_plot_nPup={0}_gbsx.pdf'.format(nPup)
fpath_bw_plot = fdir_pdf / fname_bw_plot

plot_lines = []

wv_ind = (wv0*corono3.lam_t >= (wv0_z - width_z/2))*(wv0*corono3.lam_t <= (wv0_H + width_H/2))

pl.figure(51)
pl.clf()
l1, = pl.semilogy(corono3.lam_t[wv_ind]*wv0*1e6, np.asarray(corono_poly_avg_resbis_wv_t)[wv_ind],
            color = colors_shifts[0], marker='x', ls ='-')
l2, = pl.semilogy(corono3.lam_t[wv_ind]*wv0*1e6, np.asarray(corono_poly_avg_rester_wv_t)[wv_ind],
            color = colors_shifts[1], marker='x', ls ='-')

#l5, = pl.semilogy([], [], color = "k", ls='-')
#l6, = pl.semilogy([], [], color = "k", ls='--')

pl.xlabel(r'Wavelength in $\mu$m ($\lambda_0={0:.3f}\mu$m)'.format(wv0*1e6))
pl.ylabel(r'Averaged normalized intensity')
pl.axvline(x=(corono3.lam0-bw/2)*wv0*1e6, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=(corono3.lam0+bw/2)*wv0*1e6, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole+2), xmin=0, xmax=1,
           linewidth=1, color='k', linestyle='--')    
pl.axhline(10**(-cDarkHole), xmin=0, xmax=1, 
           linewidth=1, color='k', linestyle='--')    
#pl.xlim((corono3.lam0-corono3.bw/2)*wv0*1e6, (corono3.lam0+corono3.bw/2)*wv0*1e6)
pl.xlim(corono3.lam_t[wv_ind].min()*wv0*1e6, corono3.lam_t[wv_ind].max()*wv0*1e6)
pl.ylim(1e-9, 1e-3)  
pl.title(r'Averaged intensity in monochromatic light')
pl.grid(True,which="both",ls="--")

#legend1 = pl.legend([l5,l6], ["x-axis", "y-axis"], loc=3)
#pl.gca().add_artist(legend1)
pl.legend([l1,l2], [r'{0:.1f} $\lambda_0/D$'.format(sepbis), r'{0:.1f} $\lambda_0/D$'.format(septer)], loc=4)

pl.tight_layout()
if do_plot is True:
    pl.savefig(str(fpath_bw_plot), transparent=True)

#%%
"""
Contour plot for spectral bandwidth robustness
""" 
    
direct_mono_prf_avg_t3 = np.zeros((nlam_ter, nImg2dbis//2))
corono_mono_prf_avg_t3 = np.zeros((nlam_ter, nImg2dbis//2))

direct_mono_prf_std_t3 = np.zeros((nlam_ter, nImg2dbis//2))
corono_mono_prf_std_t3 = np.zeros((nlam_ter, nImg2dbis//2))

#%%
for i in range(corono3.nlam):
    direct_mono_prf_avg_t3[i], rad_direct = imutils.profile(direct_mono_img_t3[i], type='mean')
    corono_mono_prf_avg_t3[i], rad_corono = imutils.profile(corono_mono_img_t3[i], type='mean')
    direct_mono_prf_std_t3[i], rad_direct = imutils.profile(direct_mono_img_t3[i], type='std')
    corono_mono_prf_std_t3[i], rad_corono = imutils.profile(corono_mono_img_t3[i], type='std')    

#%%
# normalization term    
direct_mono_img_f3_peak = direct_mono_img_t3[nlam_ter//2].max()

lam0D_corono = rad_corono*Fmax2dbis/nImg2dbis

wv_ind = (wv0*corono3.lam_t >= (wv0_z - width_z/2))*(wv0*corono3.lam_t <= (wv0_H + width_H/2))
lam0D_ind = lam0D_corono >= 1.0

# bounds for the separations
lam0D_min = lam0D_corono[lam0D_ind].min()
lam0D_max = lam0D_corono[lam0D_ind].max()

# bounds for the wavelengths
lam_min0 = corono3.lam_t.min()
lam_min = corono3.lam_t[wv_ind].min()
lam_max = corono3.lam_t[wv_ind].max()

# bounds for the optimization wavelength 
lam0 = 1.
lam_opt_min = lam0 - 0.5*bw#corono0.lam_t.min()
lam_opt_max = lam0 + 0.5*bw#corono0.lam_t.max()

#%%
fname_image_plane_f_disp = 'corono_poly_bw_sensitivity_contour_nPup={0}_gbsx.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_pdf / fname_image_plane_f_disp

# line width parameter
lw0 = 2.5

lam02um = wv0*1e6 
lam0D2mas = (wv0/pdiam)*(180.*3600*1000/np.pi)

tmp = corono_mono_prf_std_t3[wv_ind]

Z0 = np.log10(tmp[:, lam0D_ind]/direct_mono_img_f3_peak)
extent0 = [lam0D_min, lam0D_max, lam_min, lam_max]

f2 = pl.figure(52, figsize=(12,4.5))
pl.clf()
ax0 = f2.add_subplot(111)
im = ax0.imshow(Z0, 
                cmap = "inferno", 
                vmin=-9.0, vmax=-3.0,
                extent = extent0,
                origin = 'lower',
                )

cs = ax0.contour(Z0, [-7., -6., -5.], colors = 'white',
            extent = extent0,linestyles = '-')
ax0.clabel(cs, inline=1, fontsize=ftsz, fmt = '%1.1f')

ax0.set_xlabel(r'Angular separation in $\lambda_0$/D')
ax0.set_ylabel(r'Wavelength $\lambda$ in $\lambda_0$')
ax0.set_aspect('auto')
#ax0.set_title(r'New APLC design')
ax0.text(10**(np.log10(lam0D_max)/2), 1.4, "New APLC design", fontsize=ftsz, 
         horizontalalignment="center", color = "white")
#exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(1,))
#exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(1,))

ax0.axvline(x=rMask, ymin=-12, ymax =2, linewidth=lw0, color='r', linestyle='--')
ax0.axvline(x=rho0, ymin=-12, ymax =2, linewidth=lw0, color='b', linestyle='--')
ax0.axvline(x=rho1, ymin=-12, ymax =2, linewidth=lw0, color='b', linestyle='--')

ax0.axhline(y=1.0, xmin=0., xmax =lam0D_max, 
            linewidth=lw0, color='g', linestyle=':')
ax0.axhline(y=lam_opt_min, xmin=0., xmax =lam0D_max, 
            linewidth=lw0, color='k', linestyle=':')
ax0.axhline(y=lam_opt_max, xmin=0., xmax =lam0D_max, 
            linewidth=lw0, color='k', linestyle=':')

ax0.autoscale(False)
ax0.set_xscale("log")

ax2 = ax0.twinx()
ax2.set_ylim(lam_min*lam02um, lam_max*lam02um)
ax2.set_ylabel(r'$\lambda$ in $\mu$m ($\lambda_0={0:.3f}\mu$m)'.format(wv0*1e6), 
               rotation=270, labelpad = ftsz)
#
ax3 = ax0.twiny()
ax3.set_xlim(lam0D_min*lam0D2mas, lam0D_max*lam0D2mas)
##ax3.set_yticks()
ax3.set_xlabel(r'Angular separation in mas')
ax3.set_xscale("log")



f2.subplots_adjust(bottom=0.13, top=0.87, left=0.1, right=0.75,
                    wspace=0.02, hspace=0.02)
#
#f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.86, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('1$\sigma$ intensity in log scale', rotation=270, labelpad = 16)
if do_plot is True:
    pl.savefig(str(fpath_image_plane_f_disp), transparent=True)
pl.tight_layout()
pl.show()




#%%
pl.show()
