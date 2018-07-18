#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:26:11 2018

@author: mndiaye
"""
import numpy as np
#import pylab as pl
import time
import os

from scipy.optimize import minimize

from pathlib import Path
from corono import corono_design as cd
from corono import corono_optim_1d as co1d

from corono.utils import to_dict, update_params

#%% parameters
"""
Parameters
"""
corono_name  = 'APLC' # 'APLC' or 'SP' or 'HDZPM' or 'HTZPM'
problem_name = 'MaxTau' # 'MaxContrastLinf' # , 'MaxContrastL1' # ,
solver       = 'stdgrb' # 'stdgrb', 'gurobipy', 'scipy.linprog'

nPup = 500
nFPM = 50
nImg = 88
Fmax = 22
R    = 1

bw   = 0.2
nlam = 5

PupilObs    = 0.14
rMask       = 4.0

rMask1      = 3.0
rMask2      = 0.25
rMask3      = 0.25
OPDx2       = 0.25
OPDx3       = 0.75

LyotStopObs = 0.28
LyotStopIns = 0.9

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 2.5
rho1 = 10.0

# contrast in the dark region
cDarkHole = 10.0

# tau (integrated Pupil transmission)
tau   = 0.01

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilObs)*1.0
LyotStop1d   = (r>LyotStopObs)*(r<LyotStopIns)*1.0

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopObs = LyotStopObs,
                 LyotStopIns = LyotStopIns,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, problem_name = problem_name,
                 corono_name = corono_name)

#%%
"""
Working directory
"""
fdir = Path('.').resolve()
fdir_pyth = fdir / 'results' / '1D' / 'dat_pyth' 
if not os.path.exists(fdir):
    os.makedirs(fdir)      

#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'APLC':
    corono0 = cd.APLC1d(**params)
elif corono_name == 'SP':
    corono0 = cd.SP1d(**params)
elif corono_name == 'HDZPM':
    corono0 = cd.HDZPM1d(**params)
elif corono_name == 'HTZPM':
    corono0 = cd.HTZPM1d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem0 = co1d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem0 = co1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem0 = co1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

fname_pyth = problem0.get_filename() + '_tmp.dat'
fpath_pyth = fdir_pyth / fname_pyth

#%%
"""
Problem defintion
"""
t0 = time.time()
#%% Gurobi model of the problems
"""
Parameter boundaries
"""
if corono_name   == 'APLC':
    x_init = [rMask, LyotStopObs, LyotStopIns] 
    x_lb   = [2.0, 0.14, 0.8]
    x_ub   = [4.5, 0.50, 1.0]
elif corono_name == 'HDZPM':
    x_init = [rMask1, rMask2, OPDx2, LyotStopObs, LyotStopIns] 
    x_lb   = [1.0, 0.1, 0.0, 0.14, 0.8]
    x_ub   = [3.0, 0.5, 1.0, 0.45, 1.0]    
elif corono_name == 'HTZPM': 
    x_init = [rMask1, rMask2, rMask3, OPDx2, OPDx3, LyotStopObs, LyotStopIns] 
    x_lb   = [2.0, 0.001, 0.001, 0.0, 0.0, 0.14, 0.8]
    x_ub   = [4.5, 0.5, 0.5, 1.0, 1.0, 0.45, 1.0] 
else:    
    raise NameError('{0}: Not a correct coronagraph for NM-optimization!'.format(corono_name))    
    
#%%
def res_energy_with_lp(x_t): 
    x_t = np.maximum(x_lb,np.minimum(x_ub,x_t))

    if   corono_name == 'APLC':
        rMask, LyotStopObs, LyotStopIns = [x_t[i] for i in range(3)]
        LyotStop1d  = (r>LyotStopObs)*(r<LyotStopIns)*1.0
        params2 = update_params(params, rMask = rMask)
        
    elif corono_name == 'HDZPM':
        rMask1, OPDx2, LyotStopObs, LyotStopIns = [x_t[i] for i in {0,2,3,4}]
        rMask2      = rMask1 + x_t[1]
        LyotStop1d  = (r>LyotStopObs)*(r<LyotStopIns)*1.0
        params2 = update_params(params, OPDx2 = OPDx2, 
                                rMask1 = rMask1, rMask2 = rMask2,)
        
    elif corono_name == 'HTZPM':
        rMask1, OPDx2, OPDx3, LyotStopObs, LyotStopIns = [x_t[i] for i in {0,3,4,5,6}]
        rMask2      = rMask1 + x_t[1]
        rMask3      = rMask2 + x_t[2]
        LyotStop1d  = (r>LyotStopObs)*(r<LyotStopIns)*1.0
        params2 = update_params(params, OPDx2 = OPDx2, OPDx3 = OPDx3,
                        rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3)
        
    else:
        raise NameError('{0}: Not a correct coronagraph for NM-optimization!'.format(corono_name))    

    params2 = update_params(params2, LyotStop1d = LyotStop1d,
                        LyotStopObs = LyotStopObs, LyotStopIns = LyotStopIns)
    
    if   corono_name == 'APLC':
        corono0 = cd.APLC1d(**params2)
    elif corono_name == 'SP':
        corono0 = cd.SP1d(**params2)
    elif corono_name == 'HDZPM':
        corono0 = cd.HDZPM1d(**params2)
    elif corono_name == 'HTZPM':
        corono0 = cd.HTZPM1d(**params2)
    else:
        raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

    """
    Problem defintion
    """
    if   problem_name == 'MaxTau':
        # Maximization of the integrated amplitude transmission of the apodizer
        problem1 = co1d.MaxTau(corono=corono0, **params2)
    elif problem_name == 'MaxContrastL1':
        # Maximization of the contrast under L1-norm
        problem1 = co1d.MaxContrast(corono=corono0, Lnorm='L1',**params2)
    elif problem_name == 'MaxContrastLinf':
        # Maximization of the contrast under L-infinite norm
        problem1 = co1d.MaxContrast(corono=corono0, Lnorm='Linf',**params2)
    else:
        raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

    Apod_pyth = problem1.solve_model()

#    A, b, c = problem1.compute_matrices()
#    Apod_tmp, val = stdgrb.lp_solve(problem1.c, A=(problem1.A).T, b=problem1.b, crossover=1, logtoconsole=0)

#    Apod_pyth = np.zeros((corono0.nPup))
#    Apod_pyth[problem1.idx_pup] = Apod_tmp

    test0       = np.zeros((nPup, 2))
    test0[:, 0] = corono0.r/2
    test0[:, 1] = Apod_pyth
    
    np.savetxt(fpath_pyth, test0)
    
#    poly_direct_image_tmp = corono0.compute_direct_intensity_1d(Apod_pyth)
#    poly_corono_image_tmp = corono0.compute_corono_intensity_1d(Apod_pyth)    
#    poly_corono_image     = poly_corono_image_tmp/poly_direct_image_tmp.max()
#
#    return np.sum(poly_corono_image[problem1.idx_dz])

#    TE_throughput = -100.*np.sum(np.abs(Apod_pyth*Pupil1d*LyotStop1d)**2)/np.sum(np.abs(Pupil1d*LyotStop1d)**2)
#    return TE_throughput

    poly_direct_image_tmp = corono0.compute_direct_intensity_1d(Apod_pyth)
    poly_nostop_image_tmp = corono0.compute_nostop_intensity_1d()    

    EE_idx = corono0.xi <= 0.7
    EE_throughput = -100.*np.sum(poly_direct_image_tmp[EE_idx])/np.sum(poly_nostop_image_tmp[EE_idx])
    return EE_throughput  


#%%
x_init = np.maximum(x_lb,np.minimum(x_ub,x_init))
print('Nelder-Mead optimization for the 3 parameters')
t0 = time.time()
res = minimize(res_energy_with_lp, x_init, method='Nelder-Mead')
t1 = time.time()
print('execution time: {0:.2f}s'.format(t1-t0))

x_end = np.maximum(x_lb,np.minimum(x_ub,res.x))

#%% save solution
fname_pyth = problem0.get_filename() + '_tmp.dat'
fpath_pyth = fdir_pyth / fname_pyth

fpath_pyth = fdir_pyth / fname_pyth
test0      = np.loadtxt(fpath_pyth)
Apod_nm0   = test0[:, 1]

fname_pyth_nm0 = problem0.get_filename() + '_nm0.dat'
fpath_pyth_nm0 = fdir_pyth / fname_pyth_nm0
np.savetxt(fpath_pyth_nm0, x_end)
