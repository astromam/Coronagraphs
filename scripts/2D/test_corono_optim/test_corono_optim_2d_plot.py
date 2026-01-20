#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
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
    pupil_name   = 'vlt_btw_compass' #'vlt_btw' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxContrastL1' #'MaxContrastLinf' # 'MaxTau' # ,  'MaxContrastLinf' # #  
    solver       = 'gurobipy' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    FirstDerGlobalLim = 1.
    
    #nPup = corono0.params['nPup']
    red_factor = 1.0 #0.995
    nPup0 = 400 # 506
    nExt = 0 # 50 #int(0.05*nPup0)
    nPup = nPup0 + nExt
    nFPM = 50
    Fmax2d = 75 # 
    nImg2d = 750 # 
    
    LS_OD = 1.0#0.96
    nPupLS = int(LS_OD*nPup0)
    
    # mask radius in lam0/D unit
    # rMask = 1.766 # ALC1 at 1.593um (145mas) 
    rMask0 = 2.252 # ALC2 at 1.593um (185mas)
    # rMask0 = 2.133 # ALC3 at 2.182µm (240mas diameter)
    rMask = rMask0*1.# 1.766
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  0.0
    rho1 = 30.0
    
    # contrast in the dark region
    cDarkHole = 10.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.500 # 0.700 #  0.756/0.954 #0.756
    
    # CtrBtwnPix2

    CtrBtwnPix  = True
    CtrBtwnPix2 = True
    Pupil2dSym  = False # set it True only for optimization
    LSRobustness = True

    test_shift = True
    shift_tot0 = 1.0 #(np.sqrt(2.)/0.5)*int(0.005*nPup0) #1.0 #int(0.005*nPup0) #np.sqrt(shift_x**2+shift_y**2)
    alpha0 = 0 # np.pi/3 # np.arctan2(shift_x, shift_y)
    shift_y0 = shift_tot0*np.cos(alpha0)
    shift_x0 = shift_tot0*np.sin(alpha0)

    # shift_x0 = 0
    # shift_y0 = 1
    test_flip_x = False
    test_flip_y = False
    
    nshift = 9
    shift_max = 1. #(np.sqrt(2.)/0.5)*int(0.005*nPup0) #1.
    shift_xy_t = (shift_max/(nshift//2))*(np.arange(nshift)-nshift//2)
    shift_x_t = shift_xy_t*1 #int(0.005*nPup0)*(np.arange(nshift)-nshift//2)
    shift_y_t = shift_xy_t*1 #int(0.005*nPup0)*(np.arange(nshift)-nshift//2)
    
    ishift_all0 = nshift*np.where(shift_x_t == shift_x0)[0][0] + np.where(shift_x_t == shift_y0)[0][0]
    
    #nlam
    bw   = 0.2
    nlam = 3

    # Lyot stop with dead actuators
    do_dead_act = True
    str_dead_act = ''
    if do_dead_act:
        str_dead_act = '_deadact'
    
    LSRobustness_pre = False
    LSRobustness_bis = False
    LSRobustness_qua = False
    LSRobustness_qua2 = False
    
    LSRobustness_sev = False
    LSRobustness_v8 = False
    LSRobustness_v9 = True

    LSRobustness_coeff_pre = 10 #0.05
    LSRobustness_coeff_bis = 1.0
    LSRobustness_coeff_qua = 0.3    
    LSRobustness_coeff_qua2 = 0.5
    
    LSRobustness_coeff_sev = 24.
    
    LSRobustness_coeff_v8 = 0.1*np.sqrt(2.)/nPup
    LSRobustness_coeff_v9 = (np.sqrt(2.)/0.5)*int(0.005*nPup0) # int(0.005*nPup0)# 4*np.sqrt(2.)*int(0.005*nPup0/2) # np.sqrt(2.)*int(0.005*nPup0) #   #4 #



    str_LSRcoeff_pre = ''
    str_LSRcoeff_bis = ''
    str_LSRcoeff_qua = ''
    str_LSRcoeff_qua2 = ''
    str_LSRcoeff_sev = ''
    str_LSRcoeff_v8 = ''
    str_LSRcoeff_v9 = ''
    str_red_factor = ''
    
    if LSRobustness:
        if LSRobustness_pre:
            str_LSRcoeff_pre = f'LSRcoeff={int(np.round(LSRobustness_coeff_pre*1e3)):05d}'
        if LSRobustness_bis:
            str_LSRcoeff_bis = f'LSRcoeff_bis={int(np.round(LSRobustness_coeff_bis*1e3)):05d}'
        if LSRobustness_qua:
            str_LSRcoeff_qua = f'LSRcoeff_qua={int(np.round(LSRobustness_coeff_qua*1e3)):05d}'
        if LSRobustness_qua2:
            str_LSRcoeff_qua2 = f'LSRcoeff_qua2={int(np.round(LSRobustness_coeff_qua2*1e3)):05d}'
        if LSRobustness_sev:
            str_LSRcoeff_sev = f'LSRcoeff_sev={int(np.round(LSRobustness_coeff_sev*1e3)):05d}'
        if LSRobustness_v8:
            str_LSRcoeff_v8 = f'LSRcoeff_v8={int(np.round(LSRobustness_coeff_v8*1e3)):05d}'
        if LSRobustness_v9:
            str_LSRcoeff_v9 = f'LSRcoeff_v9={int(np.round(LSRobustness_coeff_v9*1e3)):05d}'

    if red_factor != 1.:
        str_red_factor = f'_nPup={int(np.round(red_factor*nPup0))}'
    
    do_fits = False

nlambis = 5
Fmax2dbis = 75 #  
nImg2dbis = 750 # 

ylim_min0 = 1e-8
ylim_max0 = 1e-3


#%%
"""
### File reading for Pupil and Lyot stop
"""
if True:
#    fdir = Path('../../../data/2D/pupils/').resolve()
#    fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()

    if user == 'mndiaye':
        if syst == 'darwin':
            fdir_dat = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').expanduser()
            fdir_res = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
            fdir_pdf = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/plots/').resolve()
            sim_case = 'test' # 'test' or 'server'
        elif syst == 'linux':
            fdir_dat = Path('/scratch/mndiaye/data/Coronagraphs/data/2D/pupils/').resolve()
            fdir_res = Path('/scratch/mndiaye/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
            fdir_pdf = Path('/scratch/mndiaye/data/Coronagraphs/results/2D/plots/').resolve()
            sim_case = 'server' # 'test' or 'server'            
        else:
            raise ValueError('Unknown operating system {0}'.format(user))
    else:
        raise ValueError('Unknown user {0}'.format(user))

    if pupil_name == 'lvr':
        fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup0,)
        fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup0,)
    else:
        # fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup0,)
        # fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup0,)
        fname_pup = f'pupil={pupil_name}_nArr={nPup0}_nPup={int(np.round(red_factor*nPup0))}.fits' #'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup0,)
        fname_lys = f'pupil={pupil_name}_nArr={nPup0}_nPup={int(np.round(red_factor*nPup0))}.fits' #'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup0,)
        if do_dead_act:
            fname_lys = f'sphere_stop_ST_ALC2_nPup{nPup0:04d}.fits'
    
    fpath_pup = fdir_dat / fname_pup
    fpath_lys = fdir_dat / fname_lys
    Pupil2d0    = fits.getdata(fpath_pup)
    LyotStop2d0 = fits.getdata(fpath_lys)

    if nPup0 != nPup:
        Pupil2d = np.zeros((nPup, nPup))
        LyotStop2d = np.zeros((nPup, nPup))
        ini = (nPup-nPup0)//2
        end = (nPup+nPup0)//2
        Pupil2d[ini:end, ini:end] = Pupil2d0
        LyotStop2d[ini:end, ini:end] = LyotStop2d0
    else:
        Pupil2d = Pupil2d0*1.
        LyotStop2d = LyotStop2d0*1. 

    
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
                 Pupil2dSym = Pupil2dSym, rMask=rMask0,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 LSRobustness = LSRobustness)

#%%
"""
### Working directories
"""
# fdir_res = Path('../../../results/2D/dat_pyth').resolve() / pupil_name
#fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name

#fdir_pdf = Path('../../results/2D/plots/').resolve()
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
fname     = fname_gen + f'{str_dead_act}' + str_LSRcoeff_pre + str_LSRcoeff_bis + str_LSRcoeff_qua + str_LSRcoeff_qua2 + str_LSRcoeff_sev + str_LSRcoeff_v8 + str_LSRcoeff_v9 + str_red_factor + '.fits'
fname     = fname.replace(f'N={nPup:04d}', f'N={nPup0:04d}') 
#fname     = fname.replace(f'N={nPup:04d}', f'N={nPup:04d}') 
fpath     = fdir_res / fname
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
    
if nPup0 != nPup:
    Apod2d = np.zeros((nPup, nPup))
    ini = (nPup-nPup0)//2
    end = (nPup+nPup0)//2
    Apod2d[ini:end, ini:end] = Apod_pyth
else:
    Apod2d = Apod_pyth*1. 
    

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
im = plt.imshow(Apod2d*corono0.Pupil2d, vmin=0, vmax=1, cmap = cm.Greys_r)
plt.title(f'Apod 1 transmission - {problem_name} problem - {solver}')
#plt.savefig(str(fpath))

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
fig.colorbar(im, cax=cbar_ax, label='Normalized amplitude')
plt.savefig(str(fpath))

# #%%
# fname = fname_gen + '_apodisation_ampl_flip_ud.pdf'
# fpath = fdir_pdf / fname

# fig = plt.figure(2)
# plt.clf()
# im = plt.imshow((Apod2d-np.flipud(Apod2d))*corono0.Pupil2d, vmin=0, vmax=1, cmap = cm.Greys_r)
# plt.title(f'Apod 1 transmission - {problem_name} problem - {solver}')
# plt.savefig(str(fpath))

# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
# fig.colorbar(im, cax=cbar_ax, label='Normalized amplitude')
# plt.savefig(str(fpath))

# #%%
# fname = fname_gen + '_apodisation_ampl_flip_lr.pdf'
# fpath = fdir_pdf / fname

# fig = plt.figure(3)
# plt.clf()
# im = plt.imshow((Apod2d-np.fliplr(Apod2d))*corono0.Pupil2d, vmin=0, vmax=1, cmap = cm.Greys_r)
# plt.title(f'Apod 1 transmission - {problem_name} problem - {solver}')
# plt.savefig(str(fpath))

# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
# fig.colorbar(im, cax=cbar_ax, label='Normalized amplitude')
# plt.savefig(str(fpath))

#%% Display of the apodizer
"""
### Display of the apodizer
"""
fname = fname_gen + '_lyotstop_ampl.png' # pdf is not save properly and I don't know why...
fpath = fdir_pdf / fname

fig = plt.figure(2)
plt.clf()
im = plt.imshow(LyotStop2d, vmin=0, vmax=1, cmap = cm.Greys_r)
plt.title(f'Lyot stop transmission - {problem_name} problem - {solver}')
#plt.savefig(str(fpath))

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
fig.colorbar(im, cax=cbar_ax, label='Normalized amplitude')
plt.savefig(str(fpath))


#%% Signal in intensity
"""
### Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, rMask= rMask*(nPup/nPup0), nlam=nlambis, Fmax2d = Fmax2dbis*(nPup/nPupLS), nImg2d = nImg2dbis) 

if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params2)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

if corono_name == 'APLC':
    poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod2d)
else:
    poly_direct_image1 = corono0.compute_direct_intensity_2d(corono0.Pupil2d)
poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod2d)

#%%
"""
### normalization of the direct and coronagraphic image in broadband light
"""
poly_direct_image1_peak = np.max(poly_direct_image1)

poly_direct_image1 /= poly_direct_image1_peak
poly_corono_image1 /= poly_direct_image1_peak


#%%
"""
### Computation of the direct and coronagraphic images with shifted Lyot stop
"""
if test_shift:
    
    poly_direct_image1_shift = np.zeros((nshift**2, nImg2dbis, nImg2dbis))
    poly_corono_image1_shift = np.zeros((nshift**2, nImg2dbis, nImg2dbis))
    
    xidr = np.zeros((2*nPup,2*nPup), dtype='float64')
    xyp = (nPup/(2*nPup))*(np.arange(2*nPup)-2*nPup//2+1/2)
    xxp, yyp  = np.meshgrid(xyp, xyp)
    
    for ishift_x, shift_x in enumerate(shift_x_t):
        for ishift_y, shift_y in enumerate(shift_y_t):
            
            ishift_all = nshift*ishift_x + ishift_y
    
#            LyotStop2d_shift = np.roll(np.roll(LyotStop2d, shift_x, axis=0), shift_y, axis=1)
            shift_tot = np.sqrt(shift_x**2+shift_y**2)
            alpha = np.arctan2(shift_x, shift_y)

            # dot product term insde the complex exponential to represent the shift in spatial domain
            xidr = (2.*np.pi)*(yyp*(shift_tot*np.cos(alpha)/nPup) + xxp*(shift_tot*np.sin(alpha)/nPup))
    
            # direct Fourier transform of the Lyot stop 
            FT_LyotStop2d = coro.utils.sft(LyotStop2d, 2*nPup, nPup, 
                      CtrBtwnPix=True)
            
            # FT of the Lyot stop multiplied by the exponential term to represent the shift in spatial domain
            weighted_FT_LyotStop2d = FT_LyotStop2d*np.exp(-1j*xidr)
            
            # shifted Lyot stop using FTs
            LyotStop2d_shift_ana = coro.utils.isft(weighted_FT_LyotStop2d, nPup, nPup, 
                      CtrBtwnPix=True)

            
            params_shift    = coro.update_params(params, rMask = rMask*(nPup/nPup0), nlam=nlambis, Fmax2d = Fmax2dbis*(nPup/nPupLS), nImg2d = nImg2dbis, LyotStop2d = LyotStop2d_shift_ana)
            if corono_name == 'SP':
                corono0_shift = coro.design.SP2d(**params_shift)
            elif corono_name == 'APLC':
                corono0_shift = coro.design.APLC2d(**params_shift)
            else:
                raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
        
            if corono_name == 'APLC':
                poly_direct_image1_shift[ishift_all] = corono0_shift.compute_direct_intensity_2d(Apod2d)
            else:
                poly_direct_image1_shift[ishift_all] = corono0_shift.compute_direct_intensity_2d(corono0.Pupil2d)
            poly_corono_image1_shift[ishift_all] = corono0_shift.compute_corono_intensity_2d(Apod2d)
           
            poly_direct_image1_shift_peak = np.max(poly_direct_image1_shift[ishift_all])
        
            poly_direct_image1_shift[ishift_all] /= poly_direct_image1_shift_peak
            poly_corono_image1_shift[ishift_all] /= poly_direct_image1_shift_peak

#%%
"""
### normalization of the direct and coronagraphic image in monochromatic light
"""
mono_direct_image1 = corono0.compute_direct_intensity_2d(Apod2d, poly=False)
mono_corono_image1 = corono0.compute_corono_intensity_2d(Apod2d, poly=False)

# if test_shift:
#     mono_direct_image1_shift = corono0_shift.compute_direct_intensity_2d(Apod2d, poly=False)
#     mono_corono_image1_shift = corono0_shift.compute_corono_intensity_2d(Apod2d, poly=False)

#%%
"""
### normalization of the direct and coronagraphic image in monocrhomatic light
"""

for ilam in range(corono0.nlam):
    mono_direct_image1_peak = np.max(mono_direct_image1[ilam])
       
    mono_direct_image1[ilam] /= mono_direct_image1_peak
    mono_corono_image1[ilam] /= mono_direct_image1_peak
    
# if test_shift:    
#     for ilam in range(corono0_shift.nlam):
#         mono_direct_image1_shift_peak = np.max(mono_direct_image1_shift[ilam])
           
#         mono_direct_image1_shift[ilam] /= mono_direct_image1_shift_peak
#         mono_corono_image1_shift[ilam] /= mono_direct_image1_shift_peak

#%%
"""
### computation of the averaged intensity profiles in broadband light
"""
# radius coordinate in the image
r  = np.linspace(0,nImg2dbis//2-1,num=nImg2dbis//2)
r_lamD = (Fmax2dbis/nImg2dbis)*r 

# image center
xc = nImg2dbis//2
yc = nImg2dbis//2

# grid in x and y
xx,yy = np.meshgrid(np.arange(nImg2dbis),np.arange(nImg2dbis))

# grid in radius R
R = np.sqrt((xx-xc)**2+(yy-yc)**2)

# definition of the array profiles for the direct image
poly_direct_prf_avg = np.zeros((nImg2dbis//2))
poly_direct_prf_std = np.zeros((nImg2dbis//2))

# definition of the array profiles for the coronagraphic image
poly_corono_prf_avg = np.zeros((nImg2dbis//2))
poly_corono_prf_std = np.zeros((nImg2dbis//2))

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

if test_shift:
    
    
    
    # definition of the array profiles for the direct image
    poly_direct_shift_prf_avg = np.zeros((nshift**2, nImg2dbis//2))
    poly_direct_shift_prf_std = np.zeros((nshift**2, nImg2dbis//2))

    # definition of the array profiles for the coronagraphic image
    poly_corono_shift_prf_avg = np.zeros((nshift**2, nImg2dbis//2))
    poly_corono_shift_prf_std = np.zeros((nshift**2, nImg2dbis//2))

    for ishift_x, shift_x in enumerate(shift_x_t):
        for ishift_y, shift_y in enumerate(shift_y_t):
            
            ishift_all = nshift*ishift_x + ishift_y

        
            # computation of the average and standard deviation intensity profiles for direct image 
            f_poly_dir_shift_avg = lambda r : poly_direct_image1_shift[ishift_all, (R >= r -.5) & (R < r +.5)].mean()
            f_poly_dir_shift_std = lambda r : poly_direct_image1_shift[ishift_all, (R >= r -.5) & (R < r +.5)].std()
            poly_direct_shift_prf_avg[ishift_all] = np.vectorize(f_poly_dir_shift_avg)(r)
            poly_direct_shift_prf_std[ishift_all] = np.vectorize(f_poly_dir_shift_std)(r)
        
            # computation of the average and standard deviation intensity profiles for coronagraphic image 
            f_poly_cor_shift_avg = lambda r : poly_corono_image1_shift[ishift_all, (R >= r -.5) & (R < r +.5)].mean()
            f_poly_cor_shift_std = lambda r : poly_corono_image1_shift[ishift_all, (R >= r -.5) & (R < r +.5)].std()
            poly_corono_shift_prf_avg[ishift_all] = np.vectorize(f_poly_cor_shift_avg)(r)
            poly_corono_shift_prf_std[ishift_all] = np.vectorize(f_poly_cor_shift_std)(r)

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
    
# if test_shift:
#     # definition of the array profiles for the direct image
#     mono_direct_shift_prf_avg = np.zeros((corono0.nlam, nImg2dbis//2))
#     mono_direct_shift_prf_std = np.zeros((corono0.nlam, nImg2dbis//2))

#     # definition of the array profiles for the coronagraphic image
#     mono_corono_shift_prf_avg = np.zeros((corono0.nlam, nImg2dbis//2))
#     mono_corono_shift_prf_std = np.zeros((corono0.nlam, nImg2dbis//2))
    
#     # computation of the average and standard deviation intensity profiles for direct image 
#     for ilam in range(corono0_shift.nlam):
#         f_mono_dir_shift_avg = lambda r : mono_direct_image1_shift[ilam, (R >= r -.5) & (R < r +.5)].mean()
#         f_mono_dir_shift_std = lambda r : mono_direct_image1_shift[ilam, (R >= r -.5) & (R < r +.5)].std()
#         mono_direct_shift_prf_avg[ilam] = np.vectorize(f_mono_dir_shift_avg)(r)
#         mono_direct_shift_prf_std[ilam] = np.vectorize(f_mono_dir_shift_std)(r)

#     # computation of the average and standard deviation intensity profiles for coronagraphic image
#     for ilam in range(corono0_shift.nlam): 
#         f_mono_cor_shift_avg = lambda r : mono_corono_image1_shift[ilam, (R >= r -.5) & (R < r +.5)].mean()
#         f_mono_cor_shift_std = lambda r : mono_corono_image1_shift[ilam, (R >= r -.5) & (R < r +.5)].std()
#         mono_corono_shift_prf_avg[ilam] = np.vectorize(f_mono_cor_shift_avg)(r)
#         mono_corono_shift_prf_std[ilam] = np.vectorize(f_mono_cor_shift_std)(r)

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

# if test_shift:
#     fig = plt.figure(12)
#     plt.clf()
#     im = plt.imshow(np.log10(poly_direct_image1_shift), vmin = -10, vmax = 0, cmap = cm.inferno)
#     plt.title('Apod1 - direct image (shift)')
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
#     fig.colorbar(im, cax=cbar_ax, label='Normalized intensity in log scale')
    

#%% image plot
"""
### Display of the coronagraphic images
"""
fname = fname_gen + '_apodized_image.pdf'
fpath = fdir_pdf / fname

fig = plt.figure(11)
plt.clf()
im = plt.imshow(np.log10(poly_corono_image1), vmin = -8, vmax = -3, cmap = cm.inferno)
plt.title('Apod1 - apodized image')
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
fig.colorbar(im, cax=cbar_ax, label='Normalized intensity in log scale')
plt.savefig(str(fpath))

if test_shift:
    fname = fname_gen + '_apodized_image_shift.pdf'
    fpath = fdir_pdf / fname
    
    
    fig = plt.figure(13)
    plt.clf()
    im = plt.imshow(np.log10(poly_corono_image1_shift[ishift_all0]), vmin = -8, vmax = -3, cmap = cm.inferno)
    plt.title('Apod1 - apodized image (shift)')
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
    
if test_shift:
    if corono_name == 'SP':
        plt.semilogy(xi2d,poly_corono_image1_shift[ishift_all0, nImg2dbis//2,nImg2dbis//2:]/poly_corono_image1.max(),label=solver+ ' shift', ls=':')
    else:
        plt.semilogy(xi2d,poly_corono_image1_shift[ishift_all0, nImg2dbis//2,nImg2dbis//2:]/poly_direct_image1.max(),label=solver +' shift', ls=':') 
    
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
                    label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[ilam]),
                    color = colors[ilam])
    else:
        plt.semilogy(xi2d,mono_corono_image1[ilam, nImg2dbis//2,nImg2dbis//2:], 
                    '-',
                    label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[ilam]),  
                    color = colors[ilam])

# if test_shift:
#     for ilam in range(corono0.nlam):
#         if corono_name == 'SP':
#             plt.semilogy(xi2d,mono_corono_image1_shift[ilam, nImg2dbis//2,nImg2dbis//2:]/mono_corono_image1_shift.max(), ':',
#                         label=r'{0:.2f}$\lambda_0$'.format(corono0_shift.lam_t[ilam]),
#                         color = colors[ilam])
#         else:
#             plt.semilogy(xi2d,mono_corono_image1_shift[ilam, nImg2dbis//2,nImg2dbis//2:], 
#                         ':',
#                         label=r'{0:.2f}$\lambda_0$'.format(corono0_shift.lam_t[ilam]) + ' shift',  
#                         color = colors[ilam])    

plt.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
plt.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
plt.xlabel(r'Angular separation in $\lambda_0$/D')
plt.ylabel('Normalized intensity in log scale')
plt.ylim(ylim_min0, ylim_max0)
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
fname_png = fname_gen + '_intensity_profiles_broadband_avg.png'
fpath_png = fdir_pdf / fname_png

plt.figure(30)
plt.clf()
plt.title('Averaged intensity profiles of the images')

colortest = 'C1'
if np.abs(rMask-rMask0) > 0.001:
    colortest = 'C2'


plt.semilogy(r_lamD,poly_corono_prf_avg,label=f'rMask={rMask:.3f}$\lambda_0$/D', color=colortest)

if test_shift:
    plt.semilogy(r_lamD,poly_corono_shift_prf_avg[ishift_all0],label=solver + ' shift', ls=':')
    
plt.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, linestyle='--', color=colortest)
plt.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
plt.xlabel(r'Angular separation in $\lambda_0$/D')
plt.ylabel('Normalized intensity in log scale')
plt.ylim(ylim_min0, ylim_max0)
plt.legend()
plt.tight_layout()
plt.savefig(str(fpath))
plt.savefig(str(fpath_png), transparent=True)

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
                 label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[ilam]),  color = colors[ilam])
    
# if test_shift:
#     for ilam in range(corono0.nlam):
#         plt.semilogy(r_lamD, mono_corono_shift_prf_avg[ilam],
#                      label=r'{0:.2f}$\lambda_0$'.format(corono0_shift.lam_t[ilam]) + ' shift',  color = colors[ilam], ls=':')
    
plt.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
plt.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
plt.xlabel(r'Angular separation in $\lambda_0$/D')
plt.ylabel('Normalized averaged intensity in log scale')
plt.ylim(ylim_min0, ylim_max0)
plt.legend()
plt.tight_layout()
plt.savefig(str(fpath))

#%%
"""
### Display of the coronagraphic images for all the Lyot stop shift
"""
if test_shift:
    #poly_corono_image1_shift_all = poly_corono_image1_shift.reshape((nImg2dbis*nshift, nImg2dbis*nshift))
    poly_corono_image1_shift_all = np.zeros((nImg2dbis*nshift, nImg2dbis*nshift))
    
    for ishift_x, shift_x in enumerate(shift_x_t):
        for ishift_y, shift_y in enumerate(shift_y_t):
            
            ishift_all = nshift*ishift_x + ishift_y
            
            beg_x = ishift_x*nImg2dbis
            end_x = (ishift_x+1)*nImg2dbis
            beg_y = ishift_y*nImg2dbis
            end_y = (ishift_y+1)*nImg2dbis
            poly_corono_image1_shift_all[beg_x:end_x, beg_y:end_y]= poly_corono_image1_shift[ishift_all]
    
    
    fname = fname_gen + '_apodized_image_shift_all.pdf'
    fpath = fdir_pdf / fname
    
    
    shift_x_t_str = [f'{i:.2f}' for i in shift_x_t]
    shift_y_t_str = [f'{i:.2f}' for i in shift_y_t]
    
    fig = plt.figure(40, (14, 9))
    plt.clf()
    im = plt.imshow(np.log10(poly_corono_image1_shift_all), vmin = -8, vmax = -3, cmap = cm.inferno, origin='lower')
    plt.xticks(nImg2dbis//2+nImg2dbis*np.arange(len(shift_x_t)), labels=shift_x_t_str)
    plt.yticks(nImg2dbis//2+nImg2dbis*np.arange(len(shift_y_t)), labels=shift_y_t_str)
    plt.xlabel('Lyot stop shift in % of pupil diameter along x-axis')
    plt.ylabel('Lyot stop shift in % of pupil diameter along y-axis')
    plt.vlines(x=nImg2dbis*np.arange(len(shift_x_t)), ymin=0, ymax=poly_corono_image1_shift_all.shape[0]-1, colors='w') 
    plt.hlines(y=nImg2dbis*np.arange(len(shift_x_t)), xmin=0, xmax=poly_corono_image1_shift_all.shape[0]-1, colors='w') 
    cbar_ax = fig.add_axes([0.82, 0.15, 0.03, 0.70])
    fig.colorbar(im, cax=cbar_ax, label='Normalized intensity in log scale')
    plt.tight_layout()
    plt.savefig(str(fpath), bbox_inches='tight')



#%%
plt.show()
