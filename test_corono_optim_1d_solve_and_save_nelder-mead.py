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

#%% parameters
"""
Parameters
"""
corono_name = 'APLC' # 'APLC' or 'SP'

nPup = 500
nFPM = 50
nImg = 44
Fmax = 11
R    = 1

bw   = 0.1
nlam = 3

PupilObs    = 0.14
rMask       = 4.0
LyotStopObs = 0.40
LyotStopIns = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 3.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 10.0

# tau (integrated Pupil transmission)
tau   = 0.5

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilObs)*1.0
LyotStop1d   = (r>LyotStopObs)*(r<LyotStopIns)*1.0

params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, LyotStopObs = LyotStopObs,
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
 
fname = 'BPLC_obs={0:2d}_FPM={1:3d}_ls={2:2d}_IWA={3:03d}_OWA={4:03d}_BW={5:02d}_C={6:02d}_1D_N={7:04d}_nFPM={8:03d}'.format(
                int(PupilObs*100), int(rMask*100),int(LyotStopObs*100),
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
    stop


#%%
"""
Problem defintion
"""
t0 = time.time()
# Maximization of the integrated amplitude transmission of the apodizer

# Maximization of the contrast under L1-norm
#problem1 = co1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
# Maximization of the contrast under L-infinite norm
#problem1 = co1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)

#%% Gurobi model of the problems
"""
Gurobi models
"""


#%%
x_init = [rMask, LyotStopObs, LyotStopIns] 
x_lb  = [3.0, 0.14, 0.8]
x_ub  = [5.0, 0.45, 1.0]

#%%
def res_energy_with_lp(x_t): 
    x_t = np.maximum(x_lb,np.minimum(x_ub,x_t))
    rMask       = x_t[0]
    LyotStopObs = x_t[1]
    LyotStopIns = x_t[2]
    LyotStop1d  = (r>LyotStopObs)*(r<LyotStopIns)*1.0

    params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, LyotStopObs = LyotStopObs,
                 LyotStopIns = LyotStopIns,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d)
    if corono_name == 'APLC':
        corono0 = cd.APLC1d(**params)
    elif corono_name == 'SP':
        corono0 = cd.SP1d(**params)
    else:
        stop
    
    problem1 = co1d.MaxTau(corono=corono0, **params)
    m1 = problem1.compute_gurobi_model()
    Apod_pyth = problem1.solve_model()

    fpath_pyth = fdir_pyth / fname_pyth
    test0 = np.zeros((nPup, 2))
    test0[:, 0] = corono0.r/2
    test0[:, 1] = Apod_pyth
    
    np.savetxt(fpath_pyth, test0)
    
    poly_direct_image_tmp = corono0.compute_direct_intensity_1d(Apod_pyth)
    poly_corono_image_tmp = corono0.compute_corono_intensity_1d(Apod_pyth)    
    poly_corono_image     = poly_corono_image_tmp/poly_direct_image_tmp.max()
    
    return np.sum(poly_corono_image[problem1.idx_dz])


#%%
x_init = np.maximum(x_lb,np.minimum(x_ub,x_init))
print('Nelder-Mead optimization for the 2 parameters')
t0 = time.time()
res = minimize(res_energy_with_lp, x_init, method='Nelder-Mead')
t1 = time.time()
print('execution time: {0:.2f}s'.format(t1-t0))

x_end = np.maximum(x_lb,np.minimum(x_ub,res.x))

#%%
rMask       = x_end[0]
LyotStopObs = x_end[1]
LyotStopIns = x_end[2]
LyotStop1d  = (r>LyotStopObs)*(r<LyotStopIns)*1.0

params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
             nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
             bw = bw, nlam = nlam,
             PupilObs = PupilObs, rMask = rMask, LyotStopObs = LyotStopObs,
             LyotStopIns = LyotStopIns,
             r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d)

if corono_name == 'APLC':
    corono0 = cd.APLC1d(**params)
elif corono_name == 'SP':
    corono0 = cd.SP1d(**params)
else:
    stop

fpath_pyth = fdir_pyth / fname_pyth
test0 = np.loadtxt(fpath_pyth)
Apod_nm0 = test0[:, 1]

fpath_pyth_nm0 = fdir_pyth / fname_pyth_nm0
np.savetxt(fpath_pyth_nm0, x_end)

poly_direct_image_nm0 = corono0.compute_direct_intensity_1d(Apod_nm0)
poly_corono_image_nm0 = corono0.compute_corono_intensity_1d(Apod_nm0)


#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
#Apod_pyth = problem1.solve_model()
#t1 = time.time()
#print('optimization time with gurobipy: {0:.2f}s'.format(t1-t0))
#
#fpath_pyth = fdir_pyth / fname_pyth
#
#test0 = np.zeros((nPup, 2))
#test0[:, 0] = corono0.r/2
#test0[:, 1] = Apod_pyth
#
#np.savetxt(fpath_pyth, test0)


