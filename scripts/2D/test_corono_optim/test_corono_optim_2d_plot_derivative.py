#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:49:37 2024

@author: mndiaye
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

import numdifftools as nd

import time

#%% parameters
"""
### Parameters
"""
plt.close('all')

test_gurobi = False
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'vlt_btw' #'vlt_btw' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxContrastL1' # 'MaxContrastLinf' #'MaxTau' # ,  'MaxContrastLinf' # #  
    solver       = 'gurobipy' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    FirstDerGlobalLim = 1.
    
    #nPup = corono0.params['nPup']
    nPup0 = 100
    nPup = nPup0 + 20
    nFPM = 100
    Fmax2d = 50
    nImg2d = 500
    
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
    shift_x = 2
    shift_y = 2
    do_FPM = True
    do_gradient = False
    
    LS_OD = 0.96
    nPupLS = int(nPup0*LS_OD)
    
    shift_tot = np.sqrt(shift_x**2+shift_y**2)
    test_flip_x = False
    test_flip_y = False
    
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
    LSRobustness_qua2 = True

    LSRobustness_coeff_pre = 10 #0.05
    LSRobustness_coeff_bis = 1.0
    LSRobustness_coeff_qua = 0.3
    
    LSRobustness_coeff_qua2 = 0.5

    str_LSRcoeff_pre = ''
    str_LSRcoeff_bis = ''
    str_LSRcoeff_qua = ''
    str_LSRcoeff_qua2 = ''
    if LSRobustness:
        if LSRobustness_pre:
            str_LSRcoeff_pre = f'LSRcoeff={int(np.round(LSRobustness_coeff_pre*1e3)):05d}'
        if LSRobustness_bis:
            str_LSRcoeff_bis = f'LSRcoeff_bis={int(np.round(LSRobustness_coeff_bis*1e3)):05d}'
        if LSRobustness_qua:
            str_LSRcoeff_qua = f'LSRcoeff_qua={int(np.round(LSRobustness_coeff_qua*1e3)):05d}'
        if LSRobustness_qua2:
            str_LSRcoeff_qua2 = f'LSRcoeff_qua2={int(np.round(LSRobustness_coeff_qua2*1e3)):05d}'
    
    do_fits = False

nlambis = 3    
Fmax2dbis = 50      #50
nImg2dbis = 500    #500

# Fmax2dtmp = nPup*1
# nImg2dtmp = nPup*2

ylim_min0 = 1e-8
ylim_max0 = 1e-3

vmin0 = -10
vmax0 = 0

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
            
    fdir_apo  = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/vlt_btw/').resolve()
    
    fname_apo = 'vlt_btw_APLC_obs=0.14_lsid=0.28_lsod=1.00_IWA=0.0_OWA=20.0_BW=0.20_nlam=01_1D_N=0100_nFPM=50.000000_rMask=2.252MaxContrastL1_tau=0.756_LSRobustness=1_gurobipy_deadactLSRcoeff_sev=21000.fits'
            
    fpath_apo = fdir_apo / fname_apo
    fpath_pup = fdir_dat / fname_pup
    fpath_lys = fdir_dat / fname_lys
    Apod2d0     = fits.getdata(fpath_apo)
    Pupil2d0    = fits.getdata(fpath_pup)
    LyotStop2d0 = fits.getdata(fpath_lys)
    
    if nPup0 != nPup:
        Apod2d = np.zeros((nPup, nPup))
        Pupil2d = np.zeros((nPup, nPup))
        LyotStop2d = np.zeros((nPup, nPup))
        ini = (nPup-nPup0)//2
        end = (nPup+nPup0)//2
        Apod2d[ini:end, ini:end] = Apod2d0
        Pupil2d[ini:end, ini:end] = Pupil2d0
        LyotStop2d[ini:end, ini:end] = LyotStop2d0
    else:
        Apod2d = Apod2d0*1.
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
fname     = fname_gen + f'{str_dead_act}' + str_LSRcoeff_pre + str_LSRcoeff_bis + str_LSRcoeff_qua + str_LSRcoeff_qua2 +'.fits'
fpath     = fdir_res / fname
# print(fpath)

# if test_gurobi is True:
#     idx_pup = problem1.idx_pup
#     npp = problem1.npp
    
#     sol = []
#     import csv
#     with open('/Users/mndiaye/Desktop/apod2.sol', newline='\n') as csvfile:
#         reader = csv.reader((line.replace('  ', ' ') for line in csvfile), delimiter=' ')
#         next(reader)
#         next(reader)
#         for var, value in reader:
#             sol.append(float(value))

#     Apod1 = np.zeros((corono0.nPup**2))
#     Apod1[idx_pup] = sol[:npp]
    
#     Apod_pyth = np.reshape(Apod1, (corono0.nPup, corono0.nPup))
    
#     if Pupil2dSym == True:
#         Apod1_2dtmp =  Apod_pyth[corono0.nPup//2:, corono0.nPup//2:]
#         Apod_pyth[:corono0.nPup//2, corono0.nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
#         Apod_pyth[:, :corono0.nPup//2]          = np.flip(Apod_pyth[:, corono0.nPup//2:], axis=1)
        
# else:    
#     Apod_pyth = fits.getdata(fpath,)


#%% Display of the apodizer
# """
# ### Display of the apodizer
# """
# fname = fname_gen + '_apodisation_ampl.pdf'
# fpath = fdir_pdf / fname

# fig = plt.figure(1)
# plt.clf()
# im = plt.imshow(corono0.Pupil2d, vmin=0, vmax=1, cmap = cm.Greys_r)
# plt.title(f'Apod 1 transmission - {problem_name} problem - {solver}')
# #plt.savefig(str(fpath))

# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
# fig.colorbar(im, cax=cbar_ax, label='Normalized amplitude')
# plt.savefig(str(fpath))

# #%%
# fname = fname_gen + '_apodisation_ampl_flip_ud.pdf'
# fpath = fdir_pdf / fname

# fig = plt.figure(2)
# plt.clf()
# im = plt.imshow((Apod_pyth-np.flipud(Apod_pyth))*corono0.Pupil2d, vmin=0, vmax=1, cmap = cm.Greys_r)
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
# im = plt.imshow((Apod_pyth-np.fliplr(Apod_pyth))*corono0.Pupil2d, vmin=0, vmax=1, cmap = cm.Greys_r)
# plt.title(f'Apod 1 transmission - {problem_name} problem - {solver}')
# plt.savefig(str(fpath))

# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
# fig.colorbar(im, cax=cbar_ax, label='Normalized amplitude')
# plt.savefig(str(fpath))

#%% Display of the apodizer
"""
### Display of the apodizer
"""
# fname = fname_gen + '_lyotstop_ampl.png' # pdf is not save properly and I don't know why...
# fpath = fdir_pdf / fname

# fig = plt.figure(2)
# plt.clf()
# im = plt.imshow(LyotStop2d, vmin=0, vmax=1, cmap = cm.Greys_r)
# plt.title(f'Lyot stop transmission - {problem_name} problem - {solver}')
# #plt.savefig(str(fpath))

# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
# fig.colorbar(im, cax=cbar_ax, label='Normalized amplitude')
# plt.savefig(str(fpath))


#%% Signal in intensity
"""
### Computation of the direct and coronagraphic electric field
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis) 

if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params2)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

dtype0 = 'complex128'



#%%

#alpha = np.arctan2(np.abs(shift_x), np.abs(shift_y))
alpha = np.arctan2(shift_x, shift_y)

#%%
"""
### Electric field in the entrance pupil plane A
"""
#field_A    = Pupil2d*1.
field_A = Pupil2d*Apod2d

# field_A = coro.utils.uniform_disk(nPup, nPup0//2, CtrBtwnPix=True)

#%%
"""
### Lyot stop shift
"""
#LyotStop2d = coro.utils.uniform_disk(nPup, nPupLS//2, CtrBtwnPix=True)
LyotStop2d_shift_num = np.roll(np.roll(LyotStop2d, shift_y, axis=0), shift_x, axis=1)

# computation of the analytical Lyot Stop shift
xyp = (corono0.nPup/(2*corono0.nPup))*(np.arange(2*corono0.nPup)-2*corono0.nPup//2+1/2)
xxp, yyp  = np.meshgrid(xyp, xyp)

xidr = np.zeros((2*corono0.nPup,2*corono0.nPup), dtype='float64')

# dot product term for the complex exponential to represent the shift in spatial domain
xidr = (2.*np.pi)*(yyp*(shift_tot*np.cos(alpha)/nPup) + xxp*(shift_tot*np.sin(alpha)/nPup))

# direct Fourier transform of the Lyot stop
FT_LyotStop2d = coro.utils.sft(LyotStop2d, 2*corono0.nPup, corono0.nPup, 
          CtrBtwnPix=True)

# FT of the Lyot stop multiplied by the exponential term to represent the shift in spatial domain
weighted_FT_LyotStop2d = FT_LyotStop2d*np.exp(-1j*xidr)

# shifted Lyot stop using FTs
LyotStop2d_shift_ana = coro.utils.isft(weighted_FT_LyotStop2d, corono0.nPup, corono0.nPup, 
          CtrBtwnPix=True)

# tests for first order derivation of the electric field after the shifted Lyot Stop
xixx = (2.*np.pi)*xxp
xiyy = (2.*np.pi)*yyp

# FT of the Lyot stop multiplied by the exponential term to represent the shift in spatial domain
weighted_FT_LyotStop2d_x = FT_LyotStop2d*(-1j*xixx)
weighted_FT_LyotStop2d_y = FT_LyotStop2d*(-1j*xiyy)

LyotStop2d_shift_ana_x = coro.utils.isft(weighted_FT_LyotStop2d_x, corono0.nPup, corono0.nPup, 
          CtrBtwnPix=True)
LyotStop2d_shift_ana_y = coro.utils.isft(weighted_FT_LyotStop2d_y, corono0.nPup, corono0.nPup, 
          CtrBtwnPix=True)


#%%
"""
### Electric field in the final image plane D with or without Lyot stop shift
"""        
field_Dtmp = np.zeros((corono0.nlam,corono0.nImg2d,corono0.nImg2d), dtype=dtype0)
field_Dtmp_shift_num = np.zeros((corono0.nlam,corono0.nImg2d,corono0.nImg2d), dtype=dtype0)
field_Dtmp_shift_ana = np.zeros((corono0.nlam,corono0.nImg2d,corono0.nImg2d), dtype=dtype0)
field_Dtmp_shift_grad_x = np.zeros((corono0.nlam,corono0.nImg2d,corono0.nImg2d), dtype=dtype0)
field_Dtmp_shift_grad_y = np.zeros((corono0.nlam,corono0.nImg2d,corono0.nImg2d), dtype=dtype0)

#%%
"""
### computation of the field with or without FPM
"""
if do_FPM: 
    for i in range(corono0.nlam):    

        # electric field in the focal plane B within the FPM                   
        field_B       = corono0.mask2d*coro.utils.sft(field_A, corono0.nFPM, corono0.mB_t[i]*(nPup/nPup0), 
                                        CtrBtwnPix=True) #
        
        # electric field in the re-imaged pupil plane C before the Lyot stop
        field_C       = field_A - coro.utils.isft(field_B, corono0.nPup, corono0.mB_t[i]*(nPup/nPup0), 
                                      CtrBtwnPix=True)
        
        # electric field in the re-imaged pupil plane C after the Lyot stop (centered)
        field_L       = field_C*LyotStop2d #
        
        # electric field in the final image plane D 
        field_Dtmp[i] = coro.utils.sft(field_L, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True) 
        
        # electric field in the re-imaged pupil plane C after the shifted Lyot stop 
        field_L_shift_num = field_C*LyotStop2d_shift_num

        # electric field in the final image plane D for the shifted Lyot stop case        
        field_Dtmp_shift_num[i] = coro.utils.sft(field_L_shift_num, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True) #
                
        # electric field in the re-imaged pupil plane C after the shifted Lyot stop
        field_L_shift_ana = field_C*LyotStop2d_shift_ana #
        
        # electric field in the final image plane D after the shifted Lyot stop
        field_Dtmp_shift_ana[i] = coro.utils.sft(field_L_shift_ana, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True) 

        # tests for first order derivation of the electric field after the shifted Lyot Stop                                                 
        field_L_shift_ana_x = field_C*LyotStop2d_shift_ana_x
        field_L_shift_ana_y = field_C*LyotStop2d_shift_ana_y
        
        field_Dtmp_shift_grad_x[i] = coro.utils.sft(field_L_shift_ana_x, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True) 
        field_Dtmp_shift_grad_y[i] = coro.utils.sft(field_L_shift_ana_y, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True) 
        
#        field_Dtmp_shift_grad_x[i] = field_Dtmp[i] + ((shift_tot*np.cos(alpha)/nPup0)*grad_y + (shift_tot*np.sin(alpha)/nPup0)*grad_x) 


        
else:
    for i in range(corono0.nlam):

        # electric field in the re-imaged pupil plane C after the Lyot stop (centered)                               
        field_L       = field_A*LyotStop2d
        
        # electric field in the final image plane D 
        field_Dtmp[i] = coro.utils.sft(field_L, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True)
        
        # electric field in the re-imaged pupil plane C after the shifted Lyot stop 
        field_L_shift_num = field_A*LyotStop2d_shift_num
        
        # electric field in the final image plane D for the shifted Lyot stop case 
        field_Dtmp_shift_num[i] = coro.utils.sft(field_L_shift_num, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True)
               
        # electric field in the re-imaged pupil plane C after the shifted Lyot stop
        field_L_shift_ana = field_A*LyotStop2d_shift_ana
        
        # electric field in the final image plane D after the shifted Lyot stop
        field_Dtmp_shift_ana[i] = coro.utils.sft(field_L_shift_ana, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True) 

        # tests for first order derivation of the electric field after the shifted Lyot Stop                                                 
        field_L_shift_ana_x = field_A*LyotStop2d_shift_ana_x
        field_L_shift_ana_y = field_A*LyotStop2d_shift_ana_y
        
        field_Dtmp_shift_grad_x[i] = coro.utils.sft(field_L_shift_ana_x, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True) 
        field_Dtmp_shift_grad_y[i] = coro.utils.sft(field_L_shift_ana_y, corono0.nImg2d, corono0.mD_t[i]*(nPup/nPupLS), 
                  CtrBtwnPix=True) 

               

#%%
var0 = [shift_y, shift_x]


if do_FPM: 
    def fct_field_Dtmp_mono_shift_dif(var=var0):
        
        field_Dtmp_shift_dif = np.zeros((corono0.nImg2d, corono0.nImg2d), dtype=dtype0)
        
        field_B_v       = corono0.mask2d*coro.utils.sft(field_A, corono0.nFPM, corono0.mB_t[0]*(nPup/nPup0), 
                                        CtrBtwnPix=True)
        field_C_v       = field_A - coro.utils.isft(field_B_v, corono0.nPup, corono0.mB_t[0]*(nPup/nPup0), 
                                      CtrBtwnPix=True)

        # dot product term insde the complex exponential to represent the shift in spatial domain
        xidr0 = (2.*np.pi)*(yyp*(var[0]/nPup) + xxp*(var[1]/nPup))

        # direct Fourier transform of the Lyot stop 
        FT_LyotStop2d_v = coro.utils.sft(LyotStop2d, 2*corono0.nPup, corono0.nPup, 
                  CtrBtwnPix=True)
        
        # FT of the Lyot stop multiplied by the exponential term to represent the shift in spatial domain
        weighted_FT_LyotStop2d_v = FT_LyotStop2d_v*np.exp(-1j*xidr0)
        
        # shifted Lyot stop using FTs
        LyotStop2d_shift_ana_v = coro.utils.isft(weighted_FT_LyotStop2d_v, corono0.nPup, corono0.nPup, 
                  CtrBtwnPix=True)
                        
        field_L_shift_v = field_C_v*LyotStop2d_shift_ana_v
        
        field_Dtmp_shift_dif = coro.utils.sft(field_L_shift_v, corono0.nImg2d, corono0.mD_t[0]*(nPup/nPupLS), 
                  CtrBtwnPix=True)
        
        return np.abs(field_Dtmp_shift_dif)
    
else:
    def fct_field_Dtmp_mono_shift_dif(var=var0):
        
        field_Dtmp_shift_dif = np.zeros((corono0.nImg2d, corono0.nImg2d), dtype=dtype0)

        # dot product term for the complex exponential to represent the shift in spatial domain
        xidr0 = (2.*np.pi)*(yyp*(var[0]/nPup) + xxp*(var[1]/nPup))

        # direct Fourier transform of the Lyot stop
        FT_LyotStop2d_v = coro.utils.sft(LyotStop2d, 2*corono0.nPup, corono0.nPup, 
                  CtrBtwnPix=True)
        
        # FT of the Lyot stop multiplied by the exponential term to represent the shift in spatial domain
        weighted_FT_LyotStop2d_v = FT_LyotStop2d_v*np.exp(-1j*xidr0)
        
        # shifted Lyot stop using FTs
        LyotStop2d_shift_ana_v = coro.utils.isft(weighted_FT_LyotStop2d_v, corono0.nPup, corono0.nPup, 
                  CtrBtwnPix=True)

        field_L_shift_v = field_A*LyotStop2d_shift_ana_v
        
        field_Dtmp_shift_dif = coro.utils.sft(field_L_shift_v, corono0.nImg2d, corono0.mD_t[0]*(nPup/nPupLS), 
                  CtrBtwnPix=True)
        
        return np.abs(field_Dtmp_shift_dif)


#%%
"""
### Selected wavelength for the plots
"""
ilam0 = 0 # corono0.nlam//2

#%%
plt.figure(10, (9, 9))
plt.clf()
plt.subplot(331)
plt.imshow(np.log10(np.abs(field_Dtmp[ilam0])**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|\Psi_D|^2$')
plt.subplot(332)
plt.imshow(np.log10(np.abs(field_Dtmp_shift_num[ilam0])**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|\Psi_{Dnum}^{dr}|^2$')
plt.subplot(333)
plt.imshow(np.log10(np.abs(field_Dtmp[ilam0]-field_Dtmp_shift_num[ilam0])**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|\Psi_{Dnum}^{dr}-\Psi_D|^2$')
plt.subplot(335)
plt.imshow(np.log10(np.abs(field_Dtmp_shift_ana[ilam0])**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|\Psi_{Dana}^{dr}|^2$')
plt.subplot(336)
plt.imshow(np.log10(np.abs(field_Dtmp[ilam0]-field_Dtmp_shift_ana[ilam0])**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|\Psi_{Dana}^{dr}-\Psi_D|^2$')
plt.subplot(338)
plt.imshow(np.log10(np.abs(field_Dtmp_shift_num[ilam0]-field_Dtmp_shift_ana[ilam0])**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|\Psi_{Dnum}^{dr}-\Psi_{Dana}^{dr}|^2$')

#%%
plt.figure(11, (9, 9))
plt.clf()
plt.subplot(331)
plt.imshow(np.log10(np.abs(field_Dtmp_shift_num[ilam0])**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|\Psi_{Dnum}^{dr}|^2$')
plt.subplot(332)
plt.imshow(np.log10(np.abs(np.real(field_Dtmp_shift_num[ilam0]))**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|Re(\Psi_{Dnum}^{dr})|^2$')
plt.subplot(333)
plt.imshow(np.log10(np.abs(np.imag(field_Dtmp_shift_num[ilam0]))**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|Im(\Psi_{Dnum}^{dr})|^2$')

plt.subplot(334)
plt.imshow(np.log10(np.abs(field_Dtmp_shift_ana[ilam0])**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|\Psi_{Dana}^{dr}|^2$')
plt.subplot(335)
plt.imshow(np.log10(np.abs(np.real(field_Dtmp_shift_ana[ilam0]))**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|Re(\Psi_{Dana}^{dr})|^2$')
plt.subplot(336)
plt.imshow(np.log10(np.abs(np.imag(field_Dtmp_shift_ana[ilam0]))**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|Im(\Psi_{Dana}^{dr})|^2$')


plt.subplot(337)
plt.imshow(np.log10(np.abs(field_Dtmp_shift_num[ilam0]-field_Dtmp_shift_ana[ilam0])**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|\Psi_{Dnum}^{dr}-\Psi_{Dana}^{dr}|^2$')
plt.subplot(338)
plt.imshow(np.log10(np.abs(np.real(field_Dtmp_shift_num[ilam0])-np.real(field_Dtmp_shift_ana[ilam0]))**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|Re(\Psi_{Dnum}^{dr}-\Psi_{Dana}^{dr})|^2$')
plt.subplot(339)
plt.imshow(np.log10(np.abs(np.imag(field_Dtmp_shift_num[ilam0])-np.imag(field_Dtmp_shift_ana[ilam0]))**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
plt.title(r'$|Im(\Psi_{Dnum}^{dr}-\Psi_{Dana}^{dr})|^2$')



#%%
plt.figure(12, (12, 4.5))
plt.clf()
plt.subplot(131)
plt.imshow(LyotStop2d, cmap='inferno')
plt.title(r'LS')
plt.subplot(132)
plt.imshow(LyotStop2d_shift_num, cmap='inferno')
plt.title(r'LS$^{dr}$')
plt.subplot(133)
plt.imshow(LyotStop2d_shift_num-LyotStop2d, cmap='inferno')
plt.title(r'LS-LS$^{dr}$')

#%%
plt.figure(13, (17, 4.5))
plt.clf()
plt.subplot(141)
plt.imshow(np.abs(field_C), cmap='inferno')
plt.title(r'|$\Psi_C$|')
plt.subplot(142)
plt.imshow(np.abs(field_L), cmap='inferno')
plt.title(r'|$\Psi_L$|')
plt.subplot(143)
plt.imshow(np.abs(field_L_shift_num), cmap='inferno')
plt.title(r'|$\Psi^{dr}_L$|')
plt.subplot(144)
plt.imshow(np.abs(field_L-field_L_shift_num), cmap='inferno')
plt.title(r'|$\Psi_L$-$\Psi^{dr}_L$|')

#%%

plt.figure(30)
plt.clf()
plt.subplot(231)
plt.imshow(np.real(LyotStop2d_shift_ana))
plt.title(r'Re(shifted LS$_{ana}$)')
plt.subplot(232)
plt.imshow(np.real(LyotStop2d_shift_num))
plt.title(r'Re(shifted LS$_{num}$)')
plt.subplot(233)
plt.imshow(np.real(LyotStop2d_shift_ana-LyotStop2d_shift_num))
plt.title(r'diff')
plt.subplot(234)
plt.imshow(np.imag(LyotStop2d_shift_ana))
plt.title(r'Im(shifted LS$_{ana}$)')
plt.subplot(235)
plt.imshow(np.imag(LyotStop2d_shift_num))
plt.title(r'Im(shifted LS$_{num}$)')
plt.subplot(236)
plt.imshow(np.imag(LyotStop2d_shift_ana-LyotStop2d_shift_num))
plt.title(r'diff')

#%%
"""
### Display plots for gradient
"""
plt.figure(60, (9, 9))
plt.clf()
plt.subplot(331)
plt.imshow(np.log10(np.abs(field_Dtmp[ilam0])**2), cmap='inferno', vmin=-7, vmax=0)
plt.title(r'$|\Psi_D|^2$')
plt.subplot(332)
plt.imshow(np.log10(np.abs(field_Dtmp_shift_num[ilam0])**2), cmap='inferno', vmin=-7, vmax=0)
plt.title(r'$|\Psi_{Dnum}^{dr}|^2$')
plt.subplot(333)
plt.imshow(np.log10(np.abs(field_Dtmp[ilam0]-field_Dtmp_shift_num[ilam0])**2), cmap='inferno', vmin=-7, vmax=0)
plt.title(r'$|\Psi_{Dnum}^{dr}-\Psi_D|^2$')
plt.subplot(335)
plt.imshow(np.log10(np.abs(field_Dtmp_shift_grad_x[ilam0])**2), cmap='inferno', vmin=-7, vmax=0)
plt.title(r'$|\Psi_{Dgradx}^{dr}|^2$')
plt.subplot(336)
plt.imshow(np.log10(np.abs(field_Dtmp_shift_grad_y[ilam0])**2), cmap='inferno', vmin=-7, vmax=0)
plt.title(r'$|\Psi_{Dgrady}^{dr}|^2$')
# plt.subplot(338)
# plt.imshow(np.log10(np.abs(field_Dtmp_shift_num[ilam0]-field_Dtmp_shift_grad_x[ilam0])**2), cmap='inferno', vmin=-7, vmax=0)
# plt.title(r'$|\Psi_{Dnum}^{dr}-\Psi_{Dgradx}^{dr}|^2$')


#%%
"""
### compute gradient function 
"""
if do_gradient:

    print('compute gradient function')
    
    var1 = [shift_y, shift_x]
    
    t0 = time.time()
    grad1 = nd.Gradient(fct_field_Dtmp_mono_shift_dif)([shift_y, shift_x])
    t1 = time.time() 
    
    print(f'time: {t1-t0:.2f}s')
    
    #%%
    plt.figure(20, (12, 4.5))
    plt.clf()
    plt.subplot(131)
    plt.imshow(np.log10(np.abs((field_Dtmp[corono0.nlam//2]-field_Dtmp_shift_num[corono0.nlam//2]))**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
    # plt.imshow(np.log10(np.abs((field_Dtmp[corono0.nlam//2]-field_Dtmp_shift_num[corono0.nlam//2])/(shift_tot/nPupLS))**2), cmap='inferno', vmin=-7, vmax=0)
    plt.title(r'$|\Psi_{Dnum}^{dr}-\Psi_D|^2$')
    plt.subplot(132)
    plt.imshow(np.log10(np.abs((field_Dtmp[corono0.nlam//2]-field_Dtmp_shift_ana[corono0.nlam//2]))**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
    # plt.imshow(np.log10(np.abs((field_Dtmp[corono0.nlam//2]-field_Dtmp_shift_ana[corono0.nlam//2])/(shift_tot/nPupLS))**2), cmap='inferno', vmin=-7, vmax=0)
    plt.title(r'$|\Psi_{Dana}^{dr}-\Psi_D|^2$')
    plt.subplot(133)
    plt.imshow(np.log10((np.abs(grad1[:,0])**2+np.abs(grad1[:,1])**2)/(shift_tot/nPupLS)**2), cmap='inferno', vmin=vmin0, vmax=vmax0)
    plt.title('gradient along axis-0')
