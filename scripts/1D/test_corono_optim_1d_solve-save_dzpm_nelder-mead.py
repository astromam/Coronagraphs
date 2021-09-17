#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:50:10 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

import numpy as np
#import pylab as pl
import time
import os

from scipy.optimize import minimize

from pathlib import Path
import corono as coro

#%% parameters
"""
Parameters
"""
corono_name  = 'DZPM' # 'APLC' or 'SP' or 'HDZPM' or 'HTZPM'
problem_name = 'MaxContrastL1' # 'MaxContrastLinf' # , 'MaxContrastL1' # ,
solver       = 'gurobipy' # 'stdgrb', 'gurobipy', 'scipy.linprog'
slvLogToConsole = 0
slvCrossover    = 0
slvMethod       = 2
allLogToConsole = 0

FirstDer    = False
SecondDer   = False
MinIsland   = False
FirstDerLim = 0.01
SecondDerLim= 0.001 
FirstDerGlobalLim = 1.

nPup = 500
nFPM = 50
nImg = 88
Fmax = 22
R    = 1

bw   = 0.2
nlam = 5

PupilID    = 0.14
rMask      = 4.0

rMask1      = 3.0
rMask2      = 0.25
OPDx1       = 0.25
OPDx2       = 0.75
ome1 = -2.34
ome2 = 2.051
beta = -0.236

LyotStopID = 0.14
LyotStopOD = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 2.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 10.0

# tau (integrated Pupil transmission)
tau   = 0.01

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilID)*1.0
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, 
                 OPDx1 = OPDx1, OPDx2 = OPDx2,
                 ome1 = ome1, ome2= ome2, beta = beta,
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, problem_name = problem_name,
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)

#%%
"""
Working directory
"""
fdir = Path('../../results/1D/dat_pyth').resolve()
if not os.path.exists(fdir):
    os.makedirs(fdir)      

#%%  
""" 
Coronagraph defintion
"""

if corono_name == 'DZPM':
    corono0 = coro.design.DZPM1d(**params)
else:
    raise NameError('{0}: Not a DZPM!'.format(corono_name))

#%%
"""
Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem0 = coro.optim_1d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem0 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem0 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

fname_pyth = problem0.get_filename() + '_tmp.dat'
fpath_pyth = fdir / fname_pyth

#%%
"""
Problem defintion
"""
t0 = time.time()
#%% Gurobi model of the problems
"""
Parameter boundaries
"""
if corono_name == 'DZPM':
    x_init = [rMask1, rMask2, OPDx1, OPDx2, ome1, ome2, beta, LyotStopID, LyotStopOD]
    x_lb   = [0.3, 0.0, 0.1, 0.5, -2.5, 1.5, -0.3, 0.14, 0.8]
    x_ub   = [0.6, 0.5, 0.5, 1.0, -1.5, 2.5, -0.1, 0.40, 1.0]        
else:    
    raise NameError('{0}: Not a correct coronagraph for NM-optimization!'.format(corono_name))    
    
#%%
def res_energy_with_lp(x_t): 
    x_t = np.maximum(x_lb,np.minimum(x_ub,x_t))
        
    if corono_name == 'DZPM':
        rMask1, OPDx1, OPDx2, ome1, ome2, beta, LyotStopID, LyotStopOD = [x_t[i] for i in {0,2,3,4,5,6,7,8}]
        rMask2      = rMask1 + x_t[1]
        
        LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0
        params2 = coro.update_params(params, rMask1 = rMask1, rMask2 = rMask2,
                                     OPDx1 = OPDx1, OPDx2 = OPDx2, 
                                     ome1 = ome1, ome2 = ome2, beta = beta,
                                     LyotStop1d = LyotStop1d, LyotStopID = LyotStopID, 
                                     LyotStopOD = LyotStopOD) 
        
        corono0 = coro.design.DZPM1d(**params2)
    else:
        raise NameError('{0}: Not a DZPM for NM-optimization!'.format(corono_name))    
    
    """
    Problem defintion
    """
#    if   problem_name == 'MaxTau':
#        # Maximization of the integrated amplitude transmission of the apodizer
#        problem1 = coro.optim_1d.MaxTau(corono=corono0, **params2)
#    elif problem_name == 'MaxContrastL1':
#        # Maximization of the contrast under L1-norm
#        problem1 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1',**params2)
#    elif problem_name == 'MaxContrastLinf':
#        # Maximization of the contrast under L-infinite norm
#        problem1 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='Linf',**params2)
#    else:
#        raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#    Apod_pyth = problem1.solve_model()
    Apod_pyth = 1 + ome1 *(r/2)**2 + ome2 *(r/2)**4

#    A, b, c = problem1.compute_matrices()
#    Apod_tmp, val = stdgrb.lp_solve(problem1.c, A=(problem1.A).T, b=problem1.b, crossover=1, logtoconsole=0)

#    Apod_pyth = np.zeros((corono0.nPup))
#    Apod_pyth[problem1.idx_pup] = Apod_tmp

    test0       = np.zeros((nPup, 2))
    test0[:, 0] = corono0.r/2
    test0[:, 1] = Apod_pyth
    
    np.savetxt(fpath_pyth, test0)
    
    poly_direct_image_tmp = corono0.compute_direct_intensity_1d(Apod_pyth)
    poly_corono_image_tmp = corono0.compute_corono_intensity_1d(Apod_pyth)    
    poly_corono_image     = poly_corono_image_tmp/poly_direct_image_tmp.max()

    dz      = (corono0.xi >= corono0.rho0) \
            & (corono0.xi <= corono0.rho1)
    aaa     = np.arange(corono0.nImg+1) 
    idx_dz  = list(aaa[dz])

    return np.sum(poly_corono_image[idx_dz])

#    TE_throughput = -100.*np.sum(np.abs(Apod_pyth*Pupil1d*LyotStop1d)**2)/np.sum(np.abs(Pupil1d*LyotStop1d)**2)
#    return TE_throughput

#    poly_direct_image_tmp = corono0.compute_direct_intensity_1d(Apod_pyth)
#    poly_nostop_image_tmp = corono0.compute_nostop_intensity_1d()    
#
#    EE_idx = corono0.xi <= 0.7
#    EE_throughput = -100.*np.sum(poly_direct_image_tmp[EE_idx])/np.sum(poly_nostop_image_tmp[EE_idx])
#    return EE_throughput  


#%%
x_init = np.maximum(x_lb,np.minimum(x_ub,x_init))
print('Nelder-Mead optimization for the {0} parameters'.format(np.shape(x_init)[0]))
t0 = time.time()
res = minimize(res_energy_with_lp, x_init, method='Nelder-Mead')
t1 = time.time()
print('execution time: {0:.2f}s'.format(t1-t0))

x_end = np.maximum(x_lb,np.minimum(x_ub,res.x))

#%% save solution
fname_pyth = problem0.get_filename() + '_tmp.dat'
fpath_pyth = fdir / fname_pyth

fpath_pyth = fdir / fname_pyth
test0      = np.loadtxt(fpath_pyth)
Apod_nm0   = test0[:, 1]

fname_pyth_nm0 = problem0.get_filename() + '_nm0.dat'
fpath_pyth_nm0 = fdir / fname_pyth_nm0
np.savetxt(fpath_pyth_nm0, x_end)
