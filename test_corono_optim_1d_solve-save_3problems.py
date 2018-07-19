#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:40:02 2018

@author: mndiaye
"""
import numpy as np
import os

from pathlib import Path
from corono import corono_design as cd
from corono import corono_optim_1d as co1d

from corono.utils import to_dict, update_params

#%% parameters
"""
Parameters
"""
corono_name = 'APLC' # 'APLC' or 'SP'
solver      = 'stdgrb' # 'stdgrb', 'gurobipy', 'scipy.linprog'
slvLogToConsole = 0
slvCrossover    = 0
slvMethod       = 2

nPup = 500
nFPM = 50
nImg = 44
Fmax = 11
R    = 1

bw   = 0.1
nlam = 5

PupilObs    = 0.20
rMask       = 4.0

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
cDarkHole = 10.0

# tau (integrated Pupil transmission)
tau   = 0.4

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
                 solver = solver, 
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod)

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
# Maximization of the integrated amplitude transmission of the apodizer
params  = update_params(params, problem_name = 'MaxTau')
problem1 = co1d.MaxTau(corono=corono0, **params)
# Maximization of the contrast under L1-norm
params  = update_params(params, problem_name = 'MaxContrastL1')
problem2 = co1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
# Maximization of the contrast under L-infinite norm
params  = update_params(params, problem_name = 'MaxContrastLinf')
problem3 = co1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)

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
