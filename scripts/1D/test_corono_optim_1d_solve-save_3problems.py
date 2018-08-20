#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:40:02 2018

@author: mndiaye
"""
import numpy as np
import os

from pathlib import Path
import corono as coro

#%% parameters
"""
Parameters
"""
corono_name = 'APLC' # 'APLC' or 'SP'
solver      = 'stdgrb' # 'stdgrb', 'gurobipy', 'scipy.linprog'
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
nImg = 44
Fmax = 11
R    = 1

bw   = 0.1
nlam = 5

PupilID    = 0.20
rMask       = 4.0

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75

LyotStopID = 0.40
LyotStopOD = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 3.5
rho1 = 10.0

# contrast in the dark region
cDarkHole = 10.0

# tau (integrated Pupil transmission)
tau   = 0.4

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilID)*1.0
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, 
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
# Maximization of the integrated amplitude transmission of the apodizer
params  = coro.update_params(params, problem_name = 'MaxTau')
problem1 = coro.optim_1d.MaxTau(corono=corono0, **params)
# Maximization of the contrast under L1-norm
params  = coro.update_params(params, problem_name = 'MaxContrastL1')
problem2 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
# Maximization of the contrast under L-infinite norm
params  = coro.update_params(params, problem_name = 'MaxContrastLinf')
problem3 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)

#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
Apod1 = problem1.solve_model()
Apod2 = problem2.solve_model()
Apod3 = problem3.solve_model()

#%% Apodizer solution for the problems
"""
Apodizer saving 
"""
fname1 = problem1.get_filename() + '.dat'
fname2 = problem2.get_filename() + '.dat'
fname3 = problem3.get_filename() + '.dat'

fpath1 = fdir / fname1
fpath2 = fdir / fname2
fpath3 = fdir / fname3

test1 = np.zeros((nPup, 2))
test1[:, 0] = corono0.r/2
test1[:, 1] = Apod1

test2 = np.zeros((nPup, 2))
test2[:, 0] = corono0.r/2
test2[:, 1] = Apod2

test3 = np.zeros((nPup, 2))
test3[:, 0] = corono0.r/2
test3[:, 1] = Apod3

np.savetxt(fpath1, test1)
np.savetxt(fpath2, test2)
np.savetxt(fpath3, test3)

#%%
"""
Computation of the matrices
"""
A1 = problem1.A
A2 = problem2.A
A3 = problem3.A

fname_A1 = problem1.get_filename() + '_A.dat'
fname_A2 = problem2.get_filename() + '_A.dat'
fname_A3 = problem3.get_filename() + '_A.dat'

fpath_A1 = fdir / fname_A1
fpath_A2 = fdir / fname_A2
fpath_A3 = fdir / fname_A3

np.savetxt(fpath_A1, A1)
np.savetxt(fpath_A2, A2)
np.savetxt(fpath_A3, A3)
