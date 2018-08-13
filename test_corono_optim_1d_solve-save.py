#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:26:11 2018

@author: mndiaye
"""
import numpy as np
import time
import os

from pathlib import Path
import corono as coro

#%% parameters
"""
Parameters
"""
corono_name  = 'APLC' # 'APLC' or 'SP'
problem_name = 'MaxTau' # ,'MaxTau' # 'MaxContrastLinf' #'MaxContrastL1'
solver       = 'stdgrb' # 'stdgrb', 'gurobipy', 'scipy.linprog'
slvLogToConsole = 0
slvCrossover    = 0
slvMethod       = 2
allLogToConsole = 0

FirstDer    = False
SecondDer   = False
MinIsland   = False
Binarity    = False
FirstDerLim = 0.01
SecondDerLim= 0.001 
FirstDerGlobalLim = 10.
BinarityReg       = 0.000000001

nPup = 500
nFPM = 50
nImg = 44
Fmax = 11
R    = 1

bw   = 0.1
nlam = 5

PupilObs    = 0.20
rMask       = 4.4

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75

LyotStopObs = 0.40
LyotStopIns = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 3.5
rho1 = 10.0

# contrast in the dark region
cDarkHole = 8.0

# tau (integrated Pupil transmission)
tau   = 0.3

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilObs)*1.0
LyotStop1d   = (r>LyotStopObs)*(r<LyotStopIns)*1.0

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopObs = LyotStopObs,
                 LyotStopIns = LyotStopIns,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, problem_name = problem_name,
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg)

#%%
"""
Working directory
"""
fdir = Path('.').resolve() / 'results' / '1D' / 'dat_pyth'
if not os.path.exists(fdir):
    os.makedirs(fdir)      

#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'APLC':
    corono0 = coro.design.APLC1d(**params)
elif corono_name == 'SP':
    corono0 = coro.design.SP1d(**params)
elif corono_name == 'HDZPM':
    corono0 = coro.design.HDZPM1d(**params)
elif corono_name == 'HTZPM':
    corono0 = coro.design.HTZPM1d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_1d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
     raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))
    
#%%
"""
Problem solving
"""
t0 = time.time()
Apod_pyth = problem1.solve_model()
t1 = time.time()
print('optimization time              : {0:.2f}s'.format(t1-t0))

#%% Apodizer solution for the problems
"""
Apodizer saving 
"""
fname = problem1.get_filename() + '.dat'
fpath = fdir / fname

test0 = np.zeros((nPup, 2))
test0[:, 0] = corono0.r/2
test0[:, 1] = Apod_pyth

np.savetxt(fpath, test0)
