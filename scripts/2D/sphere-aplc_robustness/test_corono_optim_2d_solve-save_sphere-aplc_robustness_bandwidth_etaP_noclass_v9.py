#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 16:05:49 2025

@author: mndiaye

License: MIT license

"""

import numpy as np
import matplotlib.pyplot as plt
ftsz = 16 
plt.rcParams.update({'font.size': ftsz})
from matplotlib.patches import Circle

from pathlib import Path
from pyzelda.utils import imutils

import os
from astropy.io import fits
import corono as coro

#from scipy.misc import imresize
from skimage.transform import resize as imresize

import time

import sys
import pwd

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform


#%% parameters
"""
### Parameters
"""
plt.close('all')

wv_min = 1.92e-6
wv_max = 2.35e-6

asym_ratio = 0.5
wv = 1.593e-6#wv_min*(1 - asym_ratio) + wv_max*asym_ratio


dAper     = 8
mas2rad   = np.pi/(180.*3600)


test_gurobi = False
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'vlt_btw' #'vlt_btw' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxContrastL1' #'MaxContrastLinf' # 'MaxTau' # ,  'MaxContrastLinf' # #  
    solver       = 'gurobipy' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    FirstDerGlobalLim = 1.
    
    #nPup = corono0.params['nPup']
    nPup0 = 506
    nExt = 0 #int(0.05*nPup0)
    nPup = nPup0 + nExt
    nFPM = 50
    Fmax2d = 50
    nImg2d = 500
    
    LS_OD = 1.0#0.96
    nPupLS = int(LS_OD*nPup0)
    
    # mask radius in lam0/D unit
    # rMask = 1.766 # ALC1 at 1.593um (145mas) 
    rMask = 2.252 # ALC2 at 1.593um (185mas)
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  0.0
    rho1 = 20.0
    
    # contrast in the dark region
    cDarkHole = 10.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.756 #0.756
    
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
    LSRobustness_coeff_v9 = 4*np.sqrt(2.)*int(0.005*nPup0/2) # np.sqrt(2.)*int(0.005*nPup0) #  int(0.005*nPup0)#  #4 #



    str_LSRcoeff_pre = ''
    str_LSRcoeff_bis = ''
    str_LSRcoeff_qua = ''
    str_LSRcoeff_qua2 = ''
    str_LSRcoeff_sev = ''
    str_LSRcoeff_v8 = ''
    str_LSRcoeff_v9 = ''
    
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


    
    do_fits = False

nlambis = 5
Fmax2dbis = 50
nImg2dbis = 500    

ylim_min0 = 1e-8
ylim_max0 = 1e-3

do_plot = True    

nImg2dbis = 256
Fmax2dbis = nImg2dbis/(2*(wv/950e-9))
nlambis   = 11

Dtel = 8
lam0 = 1
lam0D2mas = (wv/Dtel)*(180.*3600*1000/np.pi)

lam02um = wv*1e6 

val = 0
if nImg2dbis%2 == 0:
    val = 1/2

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
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup0,)
        fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup0,)
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
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 LSRobustness = LSRobustness)

#%%
"""
File reading for Pupil and Lyot stop
"""
# if True:
# #    fdir = Path('../../data/2D/pupils/').resolve()
#     fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').resolve()
#     if pupil_name == 'lvr':
#         fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
#         fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
#     elif pupil_name == 'vlt':
#         fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
#         fname_lys = 'SPHERE/sphere_stop_ST_ALC2.fits' 
#     else:
#         fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
#         fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
#         if do_dead_act:
#             fname_lys = f'sphere_stop_ST_ALC2_nPup{nPup:04d}.fits'

    
#     fpath_pup = fdir / fname_pup
#     fpath_lys = fdir / fname_lys
#     Pupil2d    = fits.getdata(fpath_pup)
#     LyotStop2d = fits.getdata(fpath_lys)

#     if test_shift:
#         LyotStop2d = np.roll(np.roll(LyotStop2d, shift_x, axis=0), shift_y, axis=1)
        
#     if test_flip_x:
#         LyotStop2d = np.fliplr(LyotStop2d)
        
#     if test_flip_y:
#         LyotStop2d = np.flipud(LyotStop2d)
    
#     if solver != 'gurobipy' and solver != 'stdgrb':
#         solver = 'scipy'

#%%
"""
Working directories
"""
#fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name

fdir_plots = Path('../../results/2D/plots/').resolve()
if not os.path.exists(fdir_plots):
    os.makedirs(fdir_plots)

#%%
"""
### Filenames to be read
"""

fname_direct_poly_img_f     = 'direct_poly_img_f.fits'
fname_corono_poly_img_f     = 'corono_poly_img_f.fits'
fpath_direct_poly_img_f     = fdir_plots / fname_direct_poly_img_f
fpath_corono_poly_img_f     = fdir_plots / fname_corono_poly_img_f

fname_direct_poly_prf_avg_f = 'direct_poly_prf_avg_f.fits'
fname_corono_poly_prf_avg_f = 'corono_poly_prf_avg_f.fits'
fname_direct_poly_prf_std_f = 'direct_poly_prf_std_f.fits'
fname_corono_poly_prf_std_f = 'corono_poly_prf_std_f.fits'
fpath_direct_poly_prf_avg_f = fdir_plots / fname_direct_poly_prf_avg_f
fpath_corono_poly_prf_avg_f = fdir_plots / fname_corono_poly_prf_avg_f
fpath_direct_poly_prf_std_f = fdir_plots / fname_direct_poly_prf_std_f
fpath_corono_poly_prf_std_f = fdir_plots / fname_corono_poly_prf_std_f

fname_direct_mono_img_t     = 'direct_mono_img_t.fits'
fname_corono_mono_img_t     = 'corono_mono_img_t.fits'
fpath_direct_mono_img_t     = fdir_plots / fname_direct_mono_img_t
fpath_corono_mono_img_t     = fdir_plots / fname_corono_mono_img_t

fname_direct_mono_prf_avg_t = 'direct_mono_prf_avg_t.fits'
fname_corono_mono_prf_avg_t = 'corono_mono_prf_avg_t.fits'
fname_direct_mono_prf_std_t = 'direct_mono_prf_std_t.fits'
fname_corono_mono_prf_std_t = 'corono_mono_prf_std_t.fits'
fpath_direct_mono_prf_avg_t = fdir_plots / fname_direct_mono_prf_avg_t
fpath_corono_mono_prf_avg_t = fdir_plots / fname_corono_mono_prf_avg_t
fpath_direct_mono_prf_std_t = fdir_plots / fname_direct_mono_prf_std_t
fpath_corono_mono_prf_std_t = fdir_plots / fname_corono_mono_prf_std_t


#%%  
""" 
Coronagraph defintion
"""
# params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
#                  rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
#                  CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
#                  nlam=nlam, bw=bw,
#                  Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
#                  Pupil2dSym = Pupil2dSym, rMask=rMask,
#                  problem_name = problem_name, 
#                  solver = solver, 
#                  corono_name = corono_name, pupil_name = pupil_name,
#                  MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
#                  LSRobustness = LSRobustness)


if corono_name == 'SP':
    corono00 = coro.design.SP2d(**params)
elif corono_name == 'APLC':
    corono00 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_2d.MaxTau(corono=corono00, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono00, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono00, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#%%
"""
Read files
"""
fname_gen = problem1.get_filename()
fname     = fname_gen + f'{str_dead_act}' + str_LSRcoeff_pre + str_LSRcoeff_bis + str_LSRcoeff_qua + str_LSRcoeff_qua2 + str_LSRcoeff_sev + str_LSRcoeff_v8 + str_LSRcoeff_v9 + '.fits'
fname     = fname.replace(f'N={nPup:04d}', f'N={nPup0:04d}') 
#fname     = fname.replace(f'N={nPup:04d}', f'N={nPup:04d}') 
fpath     = fdir_res / fname
print(fpath)


Apod2d = fits.getdata(fpath,)

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
plt.figure(4)
plt.clf()
plt.imshow(corono00.Pupil2d, cmap = 'inferno')
plt.title('Pupil transmission')

#fname = fname_gen + '_apodisation_ampl_nPup={0}.pdf'.format(nPup)
fname = 'vlt_newAPLC_apod_nPup={0:04d}.pdf'.format(nPup)
fpath = fdir_plots / fname

#pl.figure(5)
#pl.clf()
#pl.imshow(Apod2d*corono0.Pupil2d, cmap = cm.Greys_r)
#pl.title('Apod 1 transmission - MaxTau problem - '+ solver)
#pl.savefig(str(fpath))

plt.figure(2, figsize=(5,5))
plt.clf()
plt.imshow(Apod2d*corono00.Pupil2d, cmap = 'inferno')
plt.title('Apodization')
plt.tight_layout()
if do_plot is True:
    plt.savefig(str(fpath), transparent=True)



#%%
"""
Robustness to spectral bandwidth
"""
# Parameters   
nlam_ter = 11
bw_ter   = 0.20#0.92

#%%
"""
### Computation of the normalization factor
"""
# Parameters for the monochromatic PSF for a pupil without apodizer nor Lyot stop   
params_N = coro.update_params(params, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis, 
                                nlam = nlam_ter, bw = bw_ter, Pupil2d = Pupil2d, 
                                LyotStop2d = Pupil2d)

# Coronagraph object definition including previous parameters
if corono_name != 'APLC':
    raise NameError('{0}: Not an APLC!'.format(corono_name))
corono_N = coro.design.APLC2d(**params_N)

# Computation of the monochromatic psf of the telescope aperture
direct_poly_img_N = corono_N.compute_direct_intensity_2d(Pupil2d, poly=True)

# normalization term wrt to the PSF of the pupil without apodizer nor Lyot stop  
direct_poly_img_N_peak = direct_poly_img_N.max()

#%%
"""
### Computation of the monochromatic direct and coronagraphic images
"""
# filepaths of the monochromatic direct and monochromatic images
fname_imgs_dir = 'new_aplc_poly_bw={0}_nlam={1}_nPup={2}_mono_direct.fits'.format(
        bw_ter, nlam_ter, nPup)
fname_imgs_cor = 'new_aplc_poly_bw={0}_nlam={1}_nPup={2}_mono_corono.fits'.format(
        bw_ter, nlam_ter, nPup)
fpath_imgs_dir = fdir_plots / fname_imgs_dir
fpath_imgs_cor = fdir_plots / fname_imgs_cor

# Parameters for the coronagraph object 
params_S    = coro.update_params(params, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis, 
                                nlam = nlam_ter, bw = bw_ter, Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,)

# coronagraph object
if corono_name != 'APLC':
    raise NameError('{0}: Not an APLC!'.format(corono_name))
corono_S = coro.design.APLC2d(**params_S)


#%%
"""
### Planet image in monochormatic light
"""   
# vector of angular separation for the computation of planet transmission
sep_min  = 0.0
sep_max  = Fmax2dbis//2
sep_stp  = Fmax2dbis/nImg2dbis #0.5

# array of angular separations
nsep     = int(round(1+(sep_max-sep_min)/sep_stp))
sep_arr  = sep_min + sep_stp*np.arange(nsep)
print('{0:5d} sep'.format(nsep))

# array of angular distances in the final image plane
xx,yy   = np.meshgrid(np.arange(nPup)-nPup/2, np.arange(nPup)-nPup/2)
rr      = (2./float(nPup))*np.hypot(yy,xx)
theta   = np.arctan2(yy,xx)
Z       = 2.*rr*np.cos(theta)*Pupil2d   

# OPD map to generate a planet
opd_arr = [sep*wv*(1./4.) * Z for l, sep in enumerate(sep_arr)]
        
#%%
# computation of the coronagraphic image of the planet
corono_poly_img_P = np.zeros((nsep, nImg2dbis, nImg2dbis))
print('computation of the planet coronagraphic image')
t0 = time.time()
for l in range(nsep):
    params_P = coro.update_params(params_S, OPDmap2d = opd_arr[l]) 
    corono_P = coro.design.APLC2d(**params_P)
    corono_poly_img_P[l]  = corono_P.compute_corono_intensity_2d(Apod2d, poly = True)
t1 = time.time()
print('computation time: {0:.2f}s'.format(t1-t0))

# normalization of the planet coronagraphic image
corono_poly_img_P /= direct_poly_img_N_peak


#%%
"""
### Eta_S and Eta_P computation
"""
# photometric aperture circular or ring radius in lam/D
phot_rad = 0.7

# array of angular distances in the final image plane
xxi,yyi  = np.meshgrid(np.arange(nImg2dbis)-nImg2dbis//2+val, np.arange(nImg2dbis)-nImg2dbis//2+val)
mydist = (Fmax2dbis/nImg2dbis)*np.hypot(yyi,xxi)        

eta_P = np.zeros((nsep))

circ_aper = np.zeros((nImg2dbis, nImg2dbis))

sum_circ_aper = np.zeros((nsep))

print('eta_P computation')
t0 = time.time()
for i in range(nsep):    
    # compute averaged intensity of the planet at its location within a photmetric aperture of 0.7lam/D
    mydist_P = (Fmax2dbis/nImg2dbis)*np.hypot(yyi,xxi-sep_arr[i]*nImg2dbis/Fmax2dbis)        
    ind_P = (mydist_P <= phot_rad)
    circ_aper[ind_P] = 1.
    sum_circ_aper[i] = np.sum(circ_aper)
    eta_P[i] = np.sum(circ_aper*corono_poly_img_P[i])/np.sum(circ_aper)
    circ_aper[ind_P] = 0.
t1 = time.time()
print('computation time: {0}s'.format(t1-t0))

#%%
mydist_N = (Fmax2dbis/nImg2dbis)*np.hypot(yyi,xxi)        
ind_N = (mydist_N <= phot_rad)
circ_aper[ind_N] = 1.
sum_circ_aper[i] = np.sum(circ_aper)
circ_aper[ind_N] = 1.
eta_N = np.sum(circ_aper*direct_poly_img_N/direct_poly_img_N_peak)/np.sum(circ_aper)
 
#%%
fname_image_plane_f_disp = f'corono_poly_bw_throughput_nPup={nPup}_tau={tau:.3f}_disp.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_plots / fname_image_plane_f_disp

plt.figure(40, (8, 4.5))
plt.clf()
plt.plot(sep_arr, eta_P, color='C1', label='new APLC')
plt.xlabel(f'Angular separation in $\lambda_0$/D ($\lambda_0={wv*1e6}\mu$m)')
plt.ylabel(r'Planet throughput $\eta_P$')
plt.xlim(-1, 31)
plt.ylim(-0.01, 0.41)
plt.axvline(x=rMask, ymin=0, ymax =1, linewidth=1, color='r', linestyle='--')
plt.grid()
plt.legend(loc=4)
plt.tight_layout()
if do_plot is True:
    plt.savefig(str(fpath_image_plane_f_disp), transparent=True)


#%%
"""
### Apodizer throughput
"""
TT_apod = np.sum((Apod2d*Pupil2d)**2)/np.sum(Pupil2d**2)
TT_coro = np.sum((Apod2d*Pupil2d*LyotStop2d)**2)/np.sum(Pupil2d**2)

print(f'TT_apod = {TT_apod*100:.1f}%')
print(f'TT_coro = {TT_coro*100:.1f}%')

#%%