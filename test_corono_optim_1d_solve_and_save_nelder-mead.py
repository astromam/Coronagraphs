#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:26:11 2018

@author: mndiaye
"""
import numpy as np
import pylab as pl
import time
import os

from scipy.optimize import minimize

from pathlib import Path
from corono import corono_design as cd
from corono import corono_optim_1d as co1d

from corono.utils import to_dict

import stdgrb

#%% parameters
"""
Parameters
"""
corono_name = 'DZPMbis' # 'APLC' or 'SP' or DZPMbis

nPup = 500
nFPM = 50
nImg = 88
Fmax = 22
R    = 1

bw   = 0.2
nlam = 3

PupilObs    = 0.14
rMask       = 4.0

rMask1      = 2.0
rMask2      = 3.0
OPDx2       = 0.5

LyotStopObs = 0.28
LyotStopIns = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 5.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 7.0

# tau (integrated Pupil transmission)
tau   = 0.5

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilObs)*1.0
LyotStop1d   = (r>LyotStopObs)*(r<LyotStopIns)*1.0

params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, OPDx2 = OPDx2, 
                 LyotStopObs = LyotStopObs,
                 LyotStopIns = LyotStopIns,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d)


#%%
fdir = Path('.').resolve()

fdir_pyth = fdir / 'results' / '1D' / 'dat_pyth'
fdir_plot = fdir / 'results' / '1D' / 'plots'

if not os.path.exists(fdir_plot):
    os.makedirs(fdir_plot)
    
if not os.path.exists(fdir_pyth):
    os.makedirs(fdir_pyth)    


if corono_name == 'APLC' or corono_name == 'SP': 
    corono_name_bis = 'BPLC'
    if corono_name  == 'SP':
        corono_name_bis = 'SP00'
    fname = '{0}_obs={1:2d}_FPM={2:3d}_ls={3:2d}_IWA={4:03d}_OWA={5:03d}_BW={6:02d}_C={7:02d}_1D_N={8:04d}_nFPM={9:03d}'.format(
                corono_name_bis, int(PupilObs*100), int(rMask*100),
                int(LyotStopObs*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM))
else:
    fname = '{0}_obs={1:2d}_FPM1={2:3d}_FPM2={3:3d}_ls={4:2d}_IWA={5:03d}_OWA={6:03d}_BW={7:02d}_C={8:02d}_1D_N={9:04d}_nFPM={10:03d}'.format(
                corono_name, int(PupilObs*100), int(rMask1*100), int(rMask2*100), 
                int(LyotStopObs*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM))   

fname_pyth     = fname + '_guropy_apod_test.dat'
fname_pyth_nm0 = fname + '_guropy_apod_test_nm0.dat'


#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'APLC':
    corono0 = cd.APLC1d(**params)
elif corono_name == 'SP':
    corono0 = cd.SP1d(**params)
else:
    corono0 = cd.DZPM1dbis(**params)


#%%
"""
Problem defintion
"""
t0 = time.time()
# Maximization of the integrated amplitude transmission of the apodizer
#problem1 = co1d.MaxTau(corono=corono0, **params)
# Maximization of the contrast under L1-norm
#problem1 = co1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
# Maximization of the contrast under L-infinite norm
#problem1 = co1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)

#%% Gurobi model of the problems
"""
Gurobi models
"""


#%%
if corono_name == 'APLC':
    x_init = [rMask, LyotStopObs, LyotStopIns] 
    x_lb  = [2.0, 0.14, 0.8]
    x_ub  = [4.0, 0.45, 1.0]
else:
    x_init = [rMask1, rMask2, OPDx2, LyotStopObs, LyotStopIns] 
    x_lb  = [1.0, 2.0, 0.0, 0.14, 0.8]
    x_ub  = [3.0, 4.0, 1.0, 0.45, 1.0]    

#%%
def res_energy_with_lp(x_t): 
    x_t = np.maximum(x_lb,np.minimum(x_ub,x_t))

    if corono_name == 'APLC':
        rMask       = x_t[0]
        LyotStopObs = x_t[1]
        LyotStopIns = x_t[2]
        LyotStop1d  = (r>LyotStopObs)*(r<LyotStopIns)*1.0
        params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, 
                 LyotStopObs = LyotStopObs,
                 LyotStopIns = LyotStopIns,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d)
        
    else:
        rMask1      = x_t[0]
        rMask2      = x_t[1]
        OPDx2       = x_t[2]
        LyotStopObs = x_t[3]
        LyotStopIns = x_t[4]
        LyotStop1d  = (r>LyotStopObs)*(r<LyotStopIns)*1.0
        params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs,
                 rMask1 = rMask1, rMask2 = rMask2, OPDx2 = OPDx2,
                 LyotStopObs = LyotStopObs,
                 LyotStopIns = LyotStopIns,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d)
      
    if corono_name == 'APLC':
        corono0 = cd.APLC1d(**params)
    elif corono_name == 'SP':
        corono0 = cd.SP1d(**params)
    else:
        corono0 = cd.DZPM1dbis(**params)

#    corono0.params['rMask']       = rMask
#    corono0.params['LyotStopObs'] = LyotStopObs
#    corono0.params['LyotStopIns'] = LyotStopIns    
#    corono0.params['LyotStop1d']  = LyotStop1d
    """
    Problem defintion
    """
    problem1 = co1d.MaxTau(corono=corono0, **params)
#    problem1 = co1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
#    problem1 = co1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
    A, b, c = problem1.compute_matrices()
    Apod_tmp, val = stdgrb.lp_solve(problem1.c, A=(problem1.A).T, b=problem1.b, crossover=1, logtoconsole=1)

    Apod_pyth = np.zeros((corono0.nPup))
    Apod_pyth[problem1.idx_pup] = Apod_tmp

    fpath_pyth  = fdir_pyth / fname_pyth
    test0       = np.zeros((nPup, 2))
    test0[:, 0] = corono0.r/2
    test0[:, 1] = Apod_pyth
    
    np.savetxt(fpath_pyth, test0)
    
    poly_direct_image_tmp = corono0.compute_direct_intensity_1d(Apod_pyth)
    poly_corono_image_tmp = corono0.compute_corono_intensity_1d(Apod_pyth)    
    poly_corono_image     = poly_corono_image_tmp/poly_direct_image_tmp.max()
    
    return np.sum(poly_corono_image[problem1.idx_dz])


#%%
x_init = np.maximum(x_lb,np.minimum(x_ub,x_init))
print('Nelder-Mead optimization for the 3 parameters')
t0 = time.time()
res = minimize(res_energy_with_lp, x_init, method='Nelder-Mead')
t1 = time.time()
print('execution time: {0:.2f}s'.format(t1-t0))

x_end = np.maximum(x_lb,np.minimum(x_ub,res.x))

#%% save solution

fpath_pyth = fdir_pyth / fname_pyth
test0      = np.loadtxt(fpath_pyth)
Apod_nm0   = test0[:, 1]

fpath_pyth_nm0 = fdir_pyth / fname_pyth_nm0
np.savetxt(fpath_pyth_nm0, x_end)
