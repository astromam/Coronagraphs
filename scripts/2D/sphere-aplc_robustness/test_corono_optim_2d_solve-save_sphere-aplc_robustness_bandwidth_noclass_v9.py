#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 15:12:45 2025

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt
ftsz = 16 
plt.rcParams.update({'font.size': ftsz})
from matplotlib.patches import Circle
from matplotlib import cm

from pathlib import Path
from pyzelda.utils import imutils

import os
from astropy.io import fits
import corono as coro

#from scipy.misc import imresize
from skimage.transform import resize as imresize

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
    red_factor=1.#0.99
    nPup0 = 400  #506
    nExt = 0 #int(0.05*nPup0)
    nPup = nPup0 + nExt
    nFPM = 50
    Fmax2d = 50
    nImg2d = 500
    
    LS_OD = 1.0#0.96
    nPupLS = int(LS_OD*nPup0)
    
    # mask radius in lam0/D unit
    # rMask = 1.766 # ALC1 at 1.593um (145mas) 
    # rMask0 = 2.252 # ALC2 at 1.593um (185mas)
    rMask0 = 2.116 # ALC3 at 2.2µm (240mas diameter)
    SPHERE_mask = 'ALC3'
    if SPHERE_mask == 'ALC1':
        rMask = 1.766
        colortest = 'C2'
        loctest = 1
    elif SPHERE_mask == 'ALC2':
        rMask = 2.252
        colortest = 'C1' 
        loctest = 7 #4
    elif SPHERE_mask == 'ALC3':
        rMask = 2.922
        colortest = 'C0' 
        loctest = 4
    else:
        rMask = rMask0
        colortest = 'C5'
        loctest = 1
        
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  0.0
    rho1 = 30.0
    
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
    LSRobustness_coeff_v9 = (np.sqrt(2.)/0.5)*int(0.005*nPup0) #4*np.sqrt(2.)*int(0.005*nPup0/2) # np.sqrt(2.)*int(0.005*nPup0) #  int(0.005*nPup0)#  #4 #



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

    str_red_factor = ''    
    if red_factor != 1.:
        str_red_factor = f'_nPup={int(np.round(red_factor*nPup0))}'


    
    do_fits = False

nlambis = 5
Fmax2dbis = 50
nImg2dbis = 500    

ylim_min0 = 1e-8
ylim_max0 = 1e-3

do_plot = True
wv0 = 1.593e-6
wv = wv0 #2.2e-6

nImg2dbis = 256
Fmax2dbis = nImg2dbis/(2*(wv/950e-9))
nlambis   = 11

Dtel = 8
lam0D2mas = (wv0/Dtel)*(180.*3600*1000/np.pi)

lam02um = wv0*1e6 


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
#fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name

fdir_plots = Path('../../results/2D/plots/').resolve()
if not os.path.exists(fdir_plots):
    os.makedirs(fdir_plots)

#%%
"""
### filename for exporting files
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
# fname_gen = problem1.get_filename()
# fname     = fname_gen + f'{str_dead_act}.fits'
# fpath     = fdir / fname

fname_gen = problem1.get_filename()
fname     = fname_gen + f'{str_dead_act}' + str_LSRcoeff_pre + str_LSRcoeff_bis + str_LSRcoeff_qua + str_LSRcoeff_qua2 + str_LSRcoeff_sev + str_LSRcoeff_v8 + str_LSRcoeff_v9 + str_red_factor + '.fits'
fname     = fname.replace(f'N={nPup:04d}', f'N={nPup0:04d}') 
#fname     = fname.replace(f'N={nPup:04d}', f'N={nPup:04d}') 
fpath     = fdir_res / fname
print(fpath)


Apod_pyth = fits.getdata(fpath,)

#%% Display of the apodizer
"""
### Plot display of the pupil
"""
plt.figure(4)
plt.clf()
plt.imshow(corono0.Pupil2d, cmap = 'inferno')
plt.title('Pupil transmission')

#%%
#fname = fname_gen + '_apodisation_ampl_nPup={0}.pdf'.format(nPup)
fname = f'{pupil_name}_newAPLC_apod_nPup={0:04d}.pdf'.format(nPup)
fpath = fdir_plots / fname

#pl.figure(5)
#pl.clf()
#pl.imshow(Apod_pyth*corono0.Pupil2d, cmap = cm.Greys_r)
#pl.title('Apod 1 transmission - MaxTau problem - '+ solver)
#pl.savefig(str(fpath))

"""
### Plot display of the apodizer
"""
# plt.figure(2, figsize=(5,5))
# plt.clf()
# plt.imshow(Apod_pyth*corono0.Pupil2d, cmap = 'inferno')
# plt.title('Apodization')
# plt.tight_layout()
# if do_plot is True:
#     plt.savefig(str(fpath), transparent=True)


extent_pup = [-0.5, 0.5, -0.5, 0.5]

f1 = plt.figure(2)
plt.clf()
ax0 = f1.add_subplot(111)
im = ax0.imshow(Apod_pyth*corono0.Pupil2d, cmap = cm.inferno, 
          extent = extent_pup, origin = 'lower', vmin=0.0, vmax=1.0)
ax0.set_xlabel('Pupil diameter [D unit]')
#pl.imshow(Apod2d*Pupil2d, cmap = cm.Greys_r)
#ax0.title('Apodized entrance pupil')

f1.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

#f1.subplots_adjust(right=0.8)
cbar_ax = f1.add_axes([0.835, 0.20, 0.05, 0.70])
cbar    = f1.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Normalized amplitude', rotation=270, labelpad = 20)

plt.tight_layout()
plt.savefig(str(fpath), transparent=True)


#%% Signal in intensity
"""
### Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis, rMask=rMask) 

if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params2)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

if corono_name == 'APLC':
    direct_poly_img_f = corono0.compute_direct_intensity_2d(Apod_pyth)
    direct_mono_img_t = corono0.compute_direct_intensity_2d(Apod_pyth, poly=False)
else:
    direct_poly_img_f = corono0.compute_direct_intensity_2d(corono0.Pupil2d)
    direct_mono_img_t = corono0.compute_direct_intensity_2d(corono0.Pupil2d, poly=False)
    
corono_poly_img_f = corono0.compute_corono_intensity_2d(Apod_pyth)
corono_mono_img_t = corono0.compute_corono_intensity_2d(Apod_pyth, poly=False)

#%%
"""
### Normalization of the images
"""
# direct_poly_img_f_peak = np.max(direct_poly_img_f)

# direct_poly_img_f /= direct_poly_img_f_peak
# corono_poly_img_f /= direct_poly_img_f_peak

#%%
"""
### Computatation of the profiles
"""
direct_poly_prf_avg_f, rad_direct = imutils.profile(direct_poly_img_f, type='mean')
corono_poly_prf_avg_f, rad_corono = imutils.profile(corono_poly_img_f, type='mean')
direct_poly_prf_std_f, rad_direct = imutils.profile(direct_poly_img_f, type='std')
corono_poly_prf_std_f, rad_corono = imutils.profile(corono_poly_img_f, type='std')

#%%
"""
### definition of the averaged profiles for the direct and coronagraphic images 
"""
direct_mono_prf_avg_t = np.zeros((nlambis, nImg2dbis//2))
corono_mono_prf_avg_t = np.zeros((nlambis, nImg2dbis//2))

direct_mono_prf_std_t = np.zeros((nlambis, nImg2dbis//2))
corono_mono_prf_std_t = np.zeros((nlambis, nImg2dbis//2))

#%%
"""
### computation of the averaged profiles for the direct and coronagraphic images
"""
for i in range(corono0.nlam):
    direct_mono_prf_avg_t[i], rad_direct = imutils.profile(direct_mono_img_t[i], type='mean')
    corono_mono_prf_avg_t[i], rad_corono = imutils.profile(corono_mono_img_t[i], type='mean')
    direct_mono_prf_std_t[i], rad_direct = imutils.profile(direct_mono_img_t[i], type='std')
    corono_mono_prf_std_t[i], rad_corono = imutils.profile(corono_mono_img_t[i], type='std')    

#%% Intensity profiles of the direct and coronagraphic images
"""
### Display of the intensity profiles of the coronagraphic images
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

#%%


fname_image_plane_disp = 'corono_poly_img_t_disp.pdf'
fpath_image_plane_disp = fdir_plots / fname_image_plane_disp

nSub = 150
nIni = (nImg2dbis-nSub)//2
nEnd = (nImg2dbis+nSub)//2

FhalfFOV = Fmax2dbis*nSub/nImg2dbis*lam0D2mas
extent_img = [-0.5*FhalfFOV, 0.5*FhalfFOV, -0.5*FhalfFOV, 0.5*FhalfFOV]



f2 = plt.figure(22)#, figsize=(7,5))
plt.clf()
ax0 = f2.add_subplot(111)
 
im = ax0.imshow(np.log10(corono_poly_img_f[nIni:nEnd,nIni:nEnd]/direct_poly_img_f.max()), 
                cmap = "inferno", vmin=-7.5, vmax=-3.5,
                extent=extent_img)
ax0.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
ax0.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")
ax0.set_xlabel('Angular separation [mas]')

f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

#f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.835, 0.20, 0.05, 0.70])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Intensity in log scale', rotation=270, labelpad = 20)
plt.tight_layout()
plt.savefig(str(fpath_image_plane_disp), transparent=True)
plt.show()




#%%
"""
Display of the coronagraphic image
"""
fname_image_plane_f_disp = 'corono_poly_img_f_nPup={0:04d}_disp.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_plots / fname_image_plane_f_disp

lw0 = 2.5

# parameters for the circle definition
cx0 = nImg2dbis/2
cy0 = nImg2dbis/2
crm = rMask*nImg2dbis/Fmax2dbis
cr0 = rho0*nImg2dbis/Fmax2dbis
cr1 = rho1*nImg2dbis/Fmax2dbis
# definition of a circle for the pupil
circle_mask = Circle((cx0, cy0), crm, color='red', fill = False, ls = '--',
                     linewidth = lw0)
circle_rho0 = Circle((cx0, cy0), cr0, color='blue', fill = False, ls = '--',
                     linewidth = lw0)
circle_rho1 = Circle((cx0, cy0), cr1, color='blue', fill = False, ls = '--',
                     linewidth = lw0)
# in the plot




f2 = plt.figure(23, figsize=(7,5))
plt.clf()
ax0 = f2.add_subplot(111)
im = ax0.imshow(np.log10(corono_poly_img_f/direct_poly_img_f.max()), cmap = "inferno", vmin=-7.5, vmax=-3.5)
#exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "nmap={1:05d}", fontsize=16, horizontalalignment="center", color = "white")'.format(1,1))
ax0.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
ax0.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")
ax0.add_artist(circle_mask)
#ax0.add_artist(circle_rho0)
ax0.add_artist(circle_rho1)

f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.825, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Intensity in log scale', rotation=270, labelpad = 16)
if do_plot is True:
    plt.savefig(str(fpath_image_plane_f_disp), transparent=True)
plt.tight_layout()
plt.show()


#%%
"""
Display of the apodization and coronagraphic image
"""
fname_image_plane_f_disp = 'corono_poly_img_f_nPup={0:04d}_disp_all.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_plots / fname_image_plane_f_disp

lw0 = 2.5

lam00D_corono = (Fmax2dbis/nImg2dbis)*np.linspace(-nImg2dbis/2-0.5, nImg2dbis/2-0.5, nImg2dbis)

# bounds for the separations
lam00D_min = lam00D_corono.min()
lam00D_max = lam00D_corono.max()

# parameters for the circle definition
cx0 = 0#nImg2dbis/2
cy0 = 0#nImg2dbis/2
crm = rMask#*nImg2dbis/Fmax2dbis
cr0 = rho0#*nImg2dbis/Fmax2dbis
cr1 = rho1#*nImg2dbis/Fmax2dbis
# definition of a circle for the pupil
circle_mask = Circle((cx0, cy0), crm, color='red', fill = False, ls = '--',
                     linewidth = lw0)
circle_rho0 = Circle((cx0, cy0), cr0, color='blue', fill = False, ls = '--',
                     linewidth = lw0)
circle_rho1 = Circle((cx0, cy0), cr1, color='blue', fill = False, ls = '--',
                     linewidth = lw0)
# in the plot
extent_pup = [-0.5, 0.5, -0.5, 0.5]
extent_img = [lam00D_min, lam00D_max, lam00D_min, lam00D_max]


f2 = plt.figure(24, figsize=(8,4.5))
plt.clf()
ax0 = f2.add_subplot(121)
ax0.imshow(Apod_pyth*corono0.Pupil2d, cmap = 'inferno',
           extent = extent_pup,
           origin = 'lower')
ax0.set_title('APO4')
ax0.set_xlabel(r'Pupil radius r in D')
#ax0.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
#ax0.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")


ax1 = f2.add_subplot(122)
im = ax1.imshow(np.log10(corono_poly_img_f/direct_poly_img_f.max()), 
                cmap = "inferno", 
                vmin=-7.5, vmax=-3.5,
                extent = extent_img,
                origin = 'lower')
#exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "nmap={1:05d}", fontsize=16, horizontalalignment="center", color = "white")'.format(1,1))
#ax1.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
#ax1.tick_params(axis="y", which="both", left="off", right="on", labelleft="off", labelright="on")
ax1.set_title('Coronagraphic image')
ax1.set_xlim(lam00D_min, lam00D_max)
ax1.set_ylim(lam00D_min, lam00D_max)
ax1.add_artist(circle_mask)
ax1.add_artist(circle_rho0)
ax1.add_artist(circle_rho1)
ax1.set_xlabel(r'Angular separation in $\lambda_0$/D')

#ax1.autoscale(False)

f2.subplots_adjust(bottom=0.1, top=0.9, left=0.08, right=0.85,
                    wspace=0.2, hspace=0.02)

#f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.87, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Intensity in log scale', rotation=270, labelpad = 16)
if do_plot is True:
    plt.savefig(str(fpath_image_plane_f_disp), transparent=True)
plt.tight_layout()
plt.show()



#%% plot displays at multiple wavelengths

fname_image_plane_mono_plot = 'corono_poly_prf_std_t_mono_nPup={0}_plot.pdf'.format(nPup)
fpath_image_plane_mono_plot = fdir_plots / fname_image_plane_mono_plot


values = range(nlambis)
colors_wv = plt.cm.rainbow(np.linspace(0,1,nlambis))

if corono0.nlam > 1:
    idx_lam_peak = (corono0.nlam+1)//2
else:
    idx_lam_peak = 0

# Intensity profiles of the direct and coronagraphic images
plt.figure(12)
plt.clf()
for i in range(nlambis):
    plt.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_mono_prf_std_t[i]/direct_mono_img_t[idx_lam_peak].max(),
                label=r'{0:.3f}$\lambda_0$'.format(corono0.lam_t[i]), color = colors_wv[i])
    
plt.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
plt.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi2d.min(), xmax=corono0.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
plt.xlabel(r'Angular separation in $\lambda_0$/D')
plt.ylabel(r'5$\sigma$ normalized intensity in log scale')
plt.ylim(3e-8, 3e-4)
plt.legend()
plt.title('Intensity profile in monochromatic light')
plt.tight_layout()
if do_plot is True:
    plt.savefig(str(fpath_image_plane_mono_plot), transparent=True)

plt.show()


#%% Intensity profiles of the direct and coronagraphic images

fname_image_plane_plot = 'corono_poly_prf_std_t_nPup={0}_plot.pdf'.format(nPup)
fpath_image_plane_plot = fdir_plots / fname_image_plane_plot

rad_corono = np.arange(nImg2dbis//2)
colors_cor = plt.cm.rainbow(np.linspace(0,1,1))

i0 = 0

plt.figure(11, (8, 4.5))
plt.clf()
plt.semilogy(rad_corono*Fmax2dbis/nImg2dbis, corono_poly_prf_std_f/direct_poly_img_f.max(),
        label='new APLC, ' + SPHERE_mask, color = colortest)
#pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_prf_std_f,
#        label='map {0}'.format(0), color = colors_cor[i0])

plt.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color= colortest, linestyle='--')
#pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi2d.min(), xmax=corono0.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
plt.xlabel(f'Angular separation in $\lambda_0$/D ($\lambda_0={wv*1e6}\mu$m)')
plt.ylabel(r'1$\sigma$ normalized intensity in log scale')
plt.ylim(3e-8, 3e-4)
plt.legend(loc=loctest)
plt.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100), fontsize=14)
plt.tight_layout()
if do_plot is True:
    plt.savefig(str(fpath_image_plane_plot), transparent=True)


#%% Intensity profiles of the direct and coronagraphic images

fname_image_plane_plot_avg = 'corono_poly_prf_avg_t_nPup={0}_plot.pdf'.format(nPup)
fpath_image_plane_plot_avg = fdir_plots / fname_image_plane_plot_avg

rad_corono = np.arange(nImg2dbis//2)
colors_cor = plt.cm.rainbow(np.linspace(0,1,1))

i0 = 0

plt.figure(10, (8, 4.5))
plt.clf()
plt.semilogy(rad_corono*Fmax2dbis/nImg2dbis, corono_poly_prf_avg_f/direct_poly_img_f.max(),
        label='new APLC, ' + SPHERE_mask, color = colortest)
#pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_prf_std_f,
#        label='map {0}'.format(0), color = colors_cor[i0])

plt.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color= colortest, linestyle='--')
#pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi2d.min(), xmax=corono0.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
plt.xlabel(f'Angular separation in $\lambda_0$/D ($\lambda_0={wv*1e6}\mu$m)')
plt.ylabel(r'Normalized intensity in log scale')
plt.ylim(3e-8, 3e-4)
plt.legend(loc=loctest)
plt.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100), fontsize=14)
plt.tight_layout()
if do_plot is True:
    plt.savefig(str(fpath_image_plane_plot_avg), transparent=True)


#%%
"""
Robustness to spectral bandwidth
"""

#%%
   
nlam_ter = 101
bw_ter   = 0.92
   
fname_gen  = problem1.get_filename(nlam=nlambis)
params3    = coro.update_params(params, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis, 
                                nlam = nlam_ter, bw = bw_ter, rMask = rMask)

if corono_name == 'SP':
    corono3 = coro.design.SP2d(**params3)
elif corono_name == 'APLC':
    corono3 = coro.design.APLC2d(**params3)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

if corono_name == 'APLC':
    direct_poly_img_f3 = corono3.compute_direct_intensity_2d(Apod_pyth)
    direct_mono_img_t3 = corono3.compute_direct_intensity_2d(Apod_pyth, poly=False)
else:
    direct_poly_img_f3 = corono3.compute_direct_intensity_2d(corono0.Pupil2d)
    direct_mono_img_t3 = corono3.compute_direct_intensity_2d(corono0.Pupil2d, poly=False)
corono_poly_img_f3 = corono3.compute_corono_intensity_2d(Apod_pyth)
corono_mono_img_t3 = corono3.compute_corono_intensity_2d(Apod_pyth, poly=False)


#%%
val = 0
if nImg2dbis%2 == 0:
    val = 1/2

sepbis=3.0
septer=5.0

# array of angular distances in the final image plane
xx,yy  = np.meshgrid(np.arange(nImg2dbis)-nImg2dbis//2+val, np.arange(nImg2dbis)-nImg2dbis//2+val)
mydist = (Fmax2dbis/nImg2dbis)*np.hypot(yy,xx)        
resbis = (mydist <= sepbis +0.5)*(mydist >= sepbis -0.5)
rester = (mydist <= septer +0.5)*(mydist >= septer -0.5)

#%%
corono_poly_avg_resbis_wv_t = []
corono_poly_avg_rester_wv_t = []
corono_poly_std_resbis_wv_t = []
corono_poly_std_rester_wv_t = []

for i in range(nlam_ter):
    corono_poly_avg_resbis_wv_t.append(np.mean(corono_mono_img_t3[i, resbis != 0])/direct_mono_img_t3[(nlam_ter-1)//2].max())
    corono_poly_avg_rester_wv_t.append(np.mean(corono_mono_img_t3[i, rester != 0])/direct_mono_img_t3[(nlam_ter-1)//2].max())
    corono_poly_std_resbis_wv_t.append(np.std(corono_mono_img_t3[i, resbis != 0])/direct_mono_img_t3[(nlam_ter-1)//2].max())
    corono_poly_std_rester_wv_t.append(np.std(corono_mono_img_t3[i, rester != 0])/direct_mono_img_t3[(nlam_ter-1)//2].max())


#%%
colors_shifts = plt.cm.rainbow(np.linspace(0,1,2))
ls_shifts = ["-", "--"]

fname_bw_plot = 'corono_poly_bw_sensitivity_plot_nPup={0}_disp.pdf'.format(nPup)
fpath_bw_plot = fdir_plots / fname_bw_plot

plot_lines = []

plt.figure(31, (8, 4.5))
plt.clf()
# l1, = pl.semilogy(corono3.lam_t*wv*1e6, corono_poly_avg_resbis_wv_t,
#             color = colors_shifts[0], marker='x', ls ='-')
# l2, = pl.semilogy(corono3.lam_t*wv*1e6, corono_poly_avg_rester_wv_t,
#             color = colors_shifts[1], marker='x', ls ='-')



l1, = plt.semilogy(corono3.lam_t*wv*1e6, corono_poly_std_resbis_wv_t,
            color = colortest, ls ='--')
l2, = plt.semilogy(corono3.lam_t*wv*1e6, corono_poly_std_rester_wv_t,
            color = colortest, ls ='-')

#l5, = pl.semilogy([], [], color = "k", ls='-')
#l6, = pl.semilogy([], [], color = "k", ls='--')

plt.xlabel(r'Wavelength in $\mu$m ($\lambda_0={0}\mu$m)'.format(wv*1e6))
#pl.ylabel(r'Averaged normalized intensity')
plt.ylabel(r'1$\sigma$ intensity in log scale')
plt.axvline(x=(corono3.lam0-bw/2)*wv*1e6, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=(corono3.lam0+bw/2)*wv*1e6, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole+2), xmin=0, xmax=1,
           linewidth=1, color='k', linestyle='--')    
plt.axhline(10**(-cDarkHole), xmin=0, xmax=1, 
           linewidth=1, color='k', linestyle='--')    
plt.xlim((corono3.lam0-corono3.bw/2)*wv*1e6, (corono3.lam0+corono3.bw/2)*wv*1e6)
plt.ylim(3e-8, 3e-4)  
plt.title(r'1$\sigma$ intensity in monochromatic light', fontsize=14)
plt.grid(True,which="both",ls="--")

#legend1 = pl.legend([l5,l6], ["x-axis", "y-axis"], loc=3)
#pl.gca().add_artist(legend1)
# plt.legend([l1,l2], [r'{0:.1f} $\lambda_0/D$'.format(sepbis), r'{0:.1f} $\lambda_0/D$'.format(septer)], loc=4, title='new APLC', fontsize=14)
plt.legend([l1,l2], [r'{0:.1f} $\lambda_0/D$'.format(sepbis), r'{0:.1f} $\lambda_0/D$'.format(septer)], loc=2, title='', fontsize=14)

plt.tight_layout()
if do_plot is True:
    plt.savefig(str(fpath_bw_plot), transparent=True)

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

wv_ind = wv*corono3.lam_t >= 0.95e-6
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
fname_image_plane_f_disp = 'corono_poly_bw_sensitivity_contour_nPup={0}_disp.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_plots / fname_image_plane_f_disp

# line width parameter
lw0 = 2.5

tmp = corono_mono_prf_std_t3[wv_ind]

Z0 = np.log10(tmp[:, lam0D_ind]/direct_mono_img_f3_peak)
extent0 = [lam0D_min, lam0D_max, lam_min, lam_max]

f2 = plt.figure(32, figsize=(8,4.5))
plt.clf()
ax0 = f2.add_subplot(111)

im = ax0.imshow(Z0, 
                cmap = "inferno", 
                vmin=-7.5, vmax=-3.5,
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
if rho0 != 0.:
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
ax2.set_ylabel(r'$\lambda$ in $\mu$m ($\lambda_0={0}\mu$m)'.format(wv*1e6), 
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
    plt.savefig(str(fpath_image_plane_f_disp), transparent=True)
plt.tight_layout()
plt.show()

#%%

