#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""
import numpy as np
import pylab as pl
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
    problem_name = 'MaxContrastLinf' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
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
    
    odiam2 = 1.5*odiam
    thick2 = 1.5*thick
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
    nlam = 3

    
    do_fits = False

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
width_J = 2273.20e-10
bw_J  = width_J/wv0_J
rMask_J = rMask_m/(wv0_J*Fratio) 

wv0_H   = 16444.09e-10
width_H = 2984.82e-10
bw_H  = width_H/wv0_H
rMask_H = rMask_m/(wv0_H*Fratio)            

if band == 'HSC_z':
    wv0 = wv0_z
    width = width_z 
elif band == 'GPI_Y':
    wv0 = wv0_Y
    width = width_Y
elif band == 'GPI_J':
    wv0 = wv0_J
    width = width_J
elif band == 'GPI_H':
    wv0 = wv0_H
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
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
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


if test_gurobi is True:
    idx_pup = problem1.idx_pup
    npp = problem1.npp
    
    sol = []
    import csv
    with open('/Users/mndiaye/Desktop/apod2.sol', newline='\n') as csvfile:
        reader = csv.reader((line.replace('  ', ' ') for line in csvfile), delimiter=' ')
        next(reader)
        next(reader)
        for var, value in reader:
            sol.append(float(value))

    Apod1 = np.zeros((corono0.nPup**2))
    Apod1[idx_pup] = sol[:npp]
    
    Apod_pyth = np.reshape(Apod1, (corono0.nPup, corono0.nPup))
    
    if Pupil2dSym == True:
        Apod1_2dtmp =  Apod_pyth[corono0.nPup//2:, corono0.nPup//2:]
        Apod_pyth[:corono0.nPup//2, corono0.nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
        Apod_pyth[:, :corono0.nPup//2]          = np.flip(Apod_pyth[:, corono0.nPup//2:], axis=1)
        
else:    
    Apod_pyth = fits.getdata(fpath,)

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
# pl.figure(4)
# pl.clf()
# pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
# pl.title('Pupil transmission')

fname = fname_gen + '_apodisation_ampl.pdf'
fpath = fdir_pdf / fname

pl.figure(5, (12,4))
pl.clf()
pl.subplot(131)
pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil')
pl.subplot(132)
pl.imshow(Apod_pyth*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title(f'Apod 1 transmission \n {problem_name} problem - {solver}')
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
fname = fname_gen + '_direct_image.pdf'
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

fname = fname_gen + '_apodized_image.pdf'
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
fname = fname_gen + '_intensity_profiles.pdf'
fpath = fdir_pdf / fname



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
    pl.figure(21+iband)
    pl.clf()
    pl.title(f'Radial intensity profiles of the images in {bands[iband]} band')
    pl.semilogy(rad_corono_arr[iband], poly_corono_prf_avg_arr[iband], ls='-', label='avg')
    pl.semilogy(rad_corono_arr[iband], poly_corono_prf_stdpl1_arr[iband], ls='--', label=r'avg+1$\sigma$')
    pl.semilogy(rad_corono_arr[iband], poly_corono_prf_stdpl3_arr[iband], ls='-.', label=r'avg+3$\sigma$')
    
    pl.axvline(x=rMask_arr[iband], ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
    pl.xlabel(r'Angular separation in $\lambda_0$/D')
    pl.ylabel('Normalized intensity in log scale')
    pl.ylim(1e-9, 2e0)
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
pl.show()




##%% Intensity profiles of the direct and coronagraphic images
#"""
#Display of the intensity profiles of the coronagraphic images
#"""
#
#nImg2d = corono0.params['nImg2d']
#fname = fname_gen + '_intensity_profiles.pdf'
#fpath = fdir_pdf / fname
#
#pl.figure(8)
#pl.clf()
#pl.title('Radial intensity profiles of the images')
#pl.semilogy(corono0.xi2d,poly_direct_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='Direct')
##pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
##pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
#if corono_name == 'SP':
#    pl.semilogy(corono0.xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_corono_image1.max(),label='Apod')
#else:
#    pl.semilogy(corono0.xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='MaxTau')
#    
##pl.semilogy(corono0.xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
##pl.semilogy(corono0.xi2d,poly_corono_image3[nImg2d//2,nImg2d//2:]/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
##pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#pl.xlabel(r'Angular separation in $\lambda_0$/D')
#pl.ylabel('Normalized intensity in log scale')
#pl.legend()
#pl.savefig(str(fpath))

#%%
pl.show()
