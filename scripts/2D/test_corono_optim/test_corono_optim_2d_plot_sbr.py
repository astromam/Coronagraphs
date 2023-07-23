#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""
import numpy as np
import matplotlib.pyplot as plt
#import pylab as plt
from pathlib import Path

import os
from matplotlib import cm
from astropy.io import fits
import corono as coro

import sys
import pwd

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

#%% parameters
"""
### Parameters
"""
plt.close('all')

test_gurobi = False
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxTau' #'MaxContrastLinf' #'MaxContrastL1' #  , 'MaxContrastL1' # 'MaxContrastLinf' # #  
    solver       = 'gurobipy' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    FirstDerGlobalLim = 1.
    
    do_margin = True

    str_margin=''
    if do_margin:
        str_margin = '_v2'

    # telescope parameters
    pdiam, odiam = 7.92, 2.3 # tel. and obst. diameters (meters)
    if do_margin:
        pdiam, odiam = 7.92*0.99, 2.3+7.92*0.01 # tel. and obst. diameters (meters)
    thick = 0.25                # adopted spider thickness (meters)
    offset = 1.278            # spider intersection offset (meters)
    beta = 51.75              # spider angle beta

    Fratio    = 64

    kpdiam0 = pdiam/(7.92)
    kodiam0 = odiam/(2.3)
    if do_margin:
        kpdiam0 = pdiam/(7.92*0.99)
        kodiam0 = odiam/(2.3+7.92*0.01)
    kthick0 = thick/0.25
    
    #nPup = corono0.params['nPup']
    nPup = 200
    nFPM = 50
    Fmax2d = 50
    nImg2d = 500
    
    # mask radius in lam0/D unit
    rMask = 5.740/2 # ALC1 at 1.593um (145mas) 
    # rMask = 2.252 # ALC2 at 1.593um (185mas)
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  4.5
    rho1 = 20.0
    
    # contrast in the dark region
    cDarkHole = 9.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.400
    
    # CtrBtwnPix2

    CtrBtwnPix  = True
    CtrBtwnPix2 = True
    Pupil2dSym  = True # set it True only for optimization
    LSRobustness = False
    test_shift = True
    shift_x = 0
    shift_y = 0
    test_flip_x = False
    test_flip_y = True
    
    #nlam
    bw   = 0.01
    nlam = 3

    # Lyot stop with dead actuators
    do_dead_act = False
    str_dead_act = ''
    if do_dead_act:
        str_dead_act = '_deadact'
    
    do_fits = False

nlambis = 11    
Fmax2dbis = 50
nImg2dbis = 500    

#%%
"""
### File reading for Pupil and Lyot stop
"""
if True:
#    fdir = Path('../../../data/2D/pupils/').resolve()
    fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()

    if user == 'mndiaye':
        if syst == 'darwin':
            fdir = Path('~/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').expanduser()
            sim_case = 'test' # 'test' or 'server'
        elif syst == 'linux':
            fdir = Path('/home/mndiaye/python/Coronagraphs/data/2D/pupils').resolve()
            sim_case = 'server' # 'test' or 'server'            
        else:
            raise ValueError('Unknown operating system {0}'.format(user))
    else:
        raise ValueError('Unknown user {0}'.format(user))

    if pupil_name == 'sbr':
        folder_tel = ''
        fname_pup = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam0*100)):03d}_kodiam{int(np.round(kodiam0*100)):03d}_kthick{int(np.round(kthick0*100)):03d}{str_margin}.fits' 
        fname_lys = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam0*100)):03d}_kodiam{int(np.round(kodiam0*100)):03d}_kthick{int(np.round(kthick0*100)):03d}{str_margin}.fits' 
    elif pupil_name == 'lvr':
        fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
        fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
    else:
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
        fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
        if do_dead_act:
            fname_lys = f'sphere_stop_ST_ALC2_nPup{nPup:04d}.fits'
    
    fpath_pup = fdir / fname_pup
    fpath_lys = fdir / fname_lys
    Pupil2d    = fits.getdata(fpath_pup)
    LyotStop2d = fits.getdata(fpath_lys)
    
    if test_shift:
        LyotStop2d = np.roll(np.roll(LyotStop2d, shift_x, axis=0), shift_y, axis=1)
        
    if test_flip_x:
        LyotStop2d = np.fliplr(LyotStop2d)
        
    if test_flip_y:
        LyotStop2d = np.flipud(LyotStop2d)
    
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
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 LSRobustness = LSRobustness)

#%%
"""
### Working directories
"""
#fdir = Path('../../../results/2D/dat_pyth').resolve() / pupil_name
fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name

fdir_pdf = Path('../../results/2D/plots/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

#%%  
""" 
### Coronagraph defintion
"""
if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params)
elif corono_name == 'APLC':
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
### Read files
"""
fname_gen = problem1.get_filename()
fname     = fname_gen + f'{str_dead_act}.fits'
fpath     = fdir / fname
print(fpath)

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
### Display of the pupil
"""
plt.figure(0)
plt.clf()
plt.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
plt.title('Pupil transmission')

#%% Display of the apodizer
"""
### Display of the apodizer
"""
fname = fname_gen + '_apodisation_ampl.pdf'
fpath = fdir_pdf / fname

fig = plt.figure(1)
plt.clf()
im = plt.imshow(Apod_pyth*corono0.Pupil2d, vmin=0, vmax=1, cmap = cm.Greys_r)
plt.title(f'Apod 1 transmission - {problem_name} problem - {solver}')
plt.savefig(str(fpath))

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
fig.colorbar(im, cax=cbar_ax, label='Normalized amplitude')
plt.savefig(str(fpath))

#%% Signal in intensity
"""
### Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis) 

if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params2)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

if corono_name == 'APLC':
    poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod_pyth)
else:
    poly_direct_image1 = corono0.compute_direct_intensity_2d(corono0.Pupil2d)
poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod_pyth)

#%%
"""
### normalization of the direct and coronagraphic image in broadband light
"""
poly_direct_image1_peak = np.max(poly_direct_image1)

poly_direct_image1 /= poly_direct_image1_peak
poly_corono_image1 /= poly_direct_image1_peak

#%%
"""
### normalization of the direct and coronagraphic image in monochromatic light
"""
mono_direct_image1 = corono0.compute_direct_intensity_2d(Apod_pyth, poly=False)
mono_corono_image1 = corono0.compute_corono_intensity_2d(Apod_pyth, poly=False)

#%%
"""
### normalization of the direct and coronagraphic image in monocrhomatic light
"""

for ilam in range(corono0.nlam):
    mono_direct_image1_peak = np.max(mono_direct_image1[ilam])
       
    mono_direct_image1[ilam] /= mono_direct_image1_peak
    mono_corono_image1[ilam] /= mono_direct_image1_peak


#%%
"""
### computation of the averaged intensity profiles in broadband light
"""
# radius coordinate in the image
r  = np.linspace(0,nImg2dbis//2-1,num=nImg2dbis//2)
r_lamD = (Fmax2dbis/nImg2dbis)*r 

# definition of the array profiles for the direct image
poly_direct_prf_avg = np.zeros((nImg2dbis//2))
poly_direct_prf_std = np.zeros((nImg2dbis//2))

# definition of the array profiles for the coronagraphic image
poly_corono_prf_avg = np.zeros((nImg2dbis//2))
poly_corono_prf_std = np.zeros((nImg2dbis//2))

# image center
xc = nImg2dbis//2
yc = nImg2dbis//2

# grid in x and y
xx,yy = np.meshgrid(np.arange(nImg2dbis),np.arange(nImg2dbis))

# grid in radius R
R = np.sqrt((xx-xc)**2+(yy-yc)**2)

# computation of the average and standard deviation intensity profiles for direct image 
f_poly_dir_avg = lambda r : poly_direct_image1[(R >= r -.5) & (R < r +.5)].mean()
f_poly_dir_std = lambda r : poly_direct_image1[(R >= r -.5) & (R < r +.5)].std()
poly_direct_prf_avg = np.vectorize(f_poly_dir_avg)(r)
poly_direct_prf_std = np.vectorize(f_poly_dir_std)(r)

# computation of the average and standard deviation intensity profiles for coronagraphic image 
f_poly_cor_avg = lambda r : poly_corono_image1[(R >= r -.5) & (R < r +.5)].mean()
f_poly_cor_std = lambda r : poly_corono_image1[(R >= r -.5) & (R < r +.5)].std()
poly_corono_prf_avg = np.vectorize(f_poly_cor_avg)(r)
poly_corono_prf_std = np.vectorize(f_poly_cor_std)(r)

#%%
"""
### computation of the averaged intensity profiles in monochromatic light
"""
# definition of the array profiles for the direct image
mono_direct_prf_avg = np.zeros((corono0.nlam, nImg2dbis//2))
mono_direct_prf_std = np.zeros((corono0.nlam, nImg2dbis//2))

# definition of the array profiles for the coronagraphic image
mono_corono_prf_avg = np.zeros((corono0.nlam, nImg2dbis//2))
mono_corono_prf_std = np.zeros((corono0.nlam, nImg2dbis//2))

# computation of the average and standard deviation intensity profiles for direct image 
for ilam in range(corono0.nlam):
    f_mono_dir_avg = lambda r : mono_direct_image1[ilam, (R >= r -.5) & (R < r +.5)].mean()
    f_mono_dir_std = lambda r : mono_direct_image1[ilam, (R >= r -.5) & (R < r +.5)].std()
    mono_direct_prf_avg[ilam] = np.vectorize(f_mono_dir_avg)(r)
    mono_direct_prf_std[ilam] = np.vectorize(f_mono_dir_std)(r)

# computation of the average and standard deviation intensity profiles for coronagraphic image
for ilam in range(corono0.nlam): 
    f_mono_cor_avg = lambda r : mono_corono_image1[ilam, (R >= r -.5) & (R < r +.5)].mean()
    f_mono_cor_std = lambda r : mono_corono_image1[ilam, (R >= r -.5) & (R < r +.5)].std()
    mono_corono_prf_avg[ilam] = np.vectorize(f_mono_cor_avg)(r)
    mono_corono_prf_std[ilam] = np.vectorize(f_mono_cor_std)(r)

#%% image plot
"""
### Display of the direct images
"""
fname = fname_gen + '_direct_image.pdf'
fpath = fdir_pdf / fname

fig = plt.figure(10)
plt.clf()
im = plt.imshow(np.log10(poly_direct_image1), vmin = -10, vmax = 0, cmap = cm.inferno)
plt.title('Apod1 - direct image')
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
fig.colorbar(im, cax=cbar_ax, label='Normalized intensity in log scale')
plt.savefig(str(fpath))

#%% image plot
"""
### Display of the coronagraphic images
"""
fname = fname_gen + '_apodized_image.pdf'
fpath = fdir_pdf / fname

fig = plt.figure(11)
plt.clf()
im = plt.imshow(np.log10(poly_corono_image1), vmin = -10, vmax = 0, cmap = cm.inferno)
plt.title('Apod1 - apodized image')
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
fig.colorbar(im, cax=cbar_ax, label='Normalized intensity in log scale')
plt.savefig(str(fpath))

#%% Intensity profiles of the direct and coronagraphic images
"""
### Display of the intensity profiles of the coronagraphic images in broadband light
"""
xi2d = corono0.xi2d
if nImg2dbis%2 == 0:
    xi2d = corono0.xi2d_ctr

nImg2d = corono0.params['nImg2d']
fname = fname_gen + '_intensity_profiles_broadband.pdf'
fpath = fdir_pdf / fname

plt.figure(20)
plt.clf()
plt.title('Radial intensity profiles of the images')

if corono_name == 'SP':
    plt.semilogy(xi2d,poly_corono_image1[nImg2dbis//2,nImg2dbis//2:]/poly_corono_image1.max(),label=solver)
else:
    plt.semilogy(xi2d,poly_corono_image1[nImg2dbis//2,nImg2dbis//2:]/poly_direct_image1.max(),label=solver) 
    
plt.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
plt.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
plt.xlabel(r'Angular separation in $\lambda_0$/D')
plt.ylabel('Normalized intensity in log scale')
plt.ylim(1e-13, 2e0)
plt.legend()
plt.tight_layout()
plt.savefig(str(fpath))


#%%
"""
### Display of the intensity profiles of the coronagraphic images in monochromatic light
"""
values = range(nlambis)
colors = plt.cm.rainbow(np.linspace(0,1,nlambis))

plt.figure(21)
plt.clf()
plt.title('Radial intensity profiles of the images')

for ilam in range(corono0.nlam):
    if corono_name == 'SP':
        plt.semilogy(xi2d,mono_corono_image1[ilam, nImg2dbis//2,nImg2dbis//2:]/mono_corono_image1.max(), '-',
                    label=r'{0:.3f}$\lambda_0$'.format(corono0.lam_t[ilam]),
                    color = colors[ilam])
    else:
        plt.semilogy(xi2d,mono_corono_image1[ilam, nImg2dbis//2,nImg2dbis//2:], 
                    '-',
                    label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[ilam]),  color = colors[ilam])
plt.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
plt.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
plt.xlabel(r'Angular separation in $\lambda_0$/D')
plt.ylabel('Normalized intensity in log scale')
plt.ylim(1e-13, 2e0)
plt.legend()
plt.tight_layout()

#%%
"""
### Display of the averaged intensity profiles of the coronagraphic images in broadband light
"""
xi2d = corono0.xi2d
if nImg2dbis%2 == 0:
    xi2d = corono0.xi2d_ctr

nImg2d = corono0.params['nImg2d']
fname = fname_gen + '_intensity_profiles_broadband_avg.pdf'
fpath = fdir_pdf / fname

plt.figure(30)
plt.clf()
plt.title('Averaged intensity profiles of the images')

plt.semilogy(r_lamD,poly_corono_prf_avg,label=solver)
plt.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
plt.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
plt.xlabel(r'Angular separation in $\lambda_0$/D')
plt.ylabel('Normalized averaged intensity in log scale')
plt.ylim(1e-13, 2e0)
plt.legend()
plt.tight_layout()
plt.savefig(str(fpath))

#%%
"""
### Display of the averaged intensity profiles of the coronagraphic images in monochromatic light
"""
xi2d = corono0.xi2d
if nImg2dbis%2 == 0:
    xi2d = corono0.xi2d_ctr

nImg2d = corono0.params['nImg2d']
fname = fname_gen + '_intensity_profiles_monochromatic_avg.pdf'
fpath = fdir_pdf / fname

plt.figure(31)
plt.clf()
plt.title('Averaged intensity profiles of the images')

for ilam in range(corono0.nlam):
    plt.semilogy(r_lamD, mono_corono_prf_avg[ilam],
                 label=r'{0:.3f}$\lambda_0$'.format(corono0.lam_t[ilam]),  color = colors[ilam])
plt.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
plt.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
plt.xlabel(r'Angular separation in $\lambda_0$/D')
plt.ylabel('Normalized averaged intensity in log scale')
plt.ylim(1e-13, 2e0)
plt.legend()
plt.tight_layout()
plt.savefig(str(fpath))

#%%
plt.show()
