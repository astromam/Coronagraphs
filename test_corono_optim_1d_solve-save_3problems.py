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

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75

LyotStopObs = 0.28
LyotStopIns = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 5.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 8.0

# tau (integrated Pupil transmission)
tau   = 0.5

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilObs)*1.0
LyotStop1d   = (r>LyotStopObs)*(r<LyotStopIns)*1.0

params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
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
    fname = '{0}_obs={1:2d}_FPM={2:3d}_lsid={3:2d}_lsod={4:2d}_IWA={5:03d}_OWA={6:03d}_BW={7:02d}_C={8:02d}_1D_N={9:04d}_nFPM={10:03d}'.format(
                corono_name, int(PupilObs*100), int(rMask*100),
                int(LyotStopObs*100), int(LyotStopIns*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM))
elif corono_name == 'HDZPM':
    fname = '{0}_obs={1:2d}_FPM1={2:3d}_FPM2={3:3d}_lsid={4:2d}_lsod={5:2d}_IWA={6:03d}_OWA={7:03d}_BW={8:02d}_C={9:02d}_1D_N={10:04d}_nFPM={11:03d}'.format(
                corono_name, int(PupilObs*100), int(rMask1*100), int(rMask2*100), 
                int(LyotStopObs*100), int(LyotStopIns*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM)) 
elif corono_name == 'HTZPM':
    fname = '{0}_obs={1:2d}_FPM1={2:3d}_FPM2={3:3d}_FPM3={4:3d}_lsid={5:2d}_lsod={6:2d}_IWA={7:03d}_OWA={8:03d}_BW={9:02d}_C={10:02d}_1D_N={11:04d}_nFPM={12:03d}'.format(
                corono_name, int(PupilObs*100), 
                int(rMask1*100), int(rMask2*100), int(rMask3*100), 
                int(LyotStopObs*100), int(LyotStopIns*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM)) 
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))


fname1 = fname + '_guropy_apod_MaxTau.dat'
fname2 = fname + '_guropy_apod_MaxContrastL1.dat'
fname3 = fname + '_guropy_apod_MaxContrastLinf.dat'

fname_A1 = fname + '_guropy_A_MaxTau.dat'
fname_A2 = fname + '_guropy_A_MaxContrastL1.dat'
fname_A3 = fname + '_guropy_A_MaxContrastLinf.dat'

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
problem1 = co1d.MaxTau(corono=corono0, **params)
# Maximization of the contrast under L1-norm
problem2 = co1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
# Maximization of the contrast under L-infinite norm
problem3 = co1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)

#%% Gurobi model of the problems
"""
Gurobi models
"""
m1 = problem1.compute_gurobi_model()
m2 = problem2.compute_gurobi_model()
m3 = problem3.compute_gurobi_model()

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
fpath1 = fdir_pyth / fname1
fpath2 = fdir_pyth / fname2
fpath3 = fdir_pyth / fname3

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

fpath_A1 = fdir_pyth / fname_A1
fpath_A2 = fdir_pyth / fname_A2
fpath_A3 = fdir_pyth / fname_A3

np.savetxt(fpath_A1, A1)
np.savetxt(fpath_A2, A2)
np.savetxt(fpath_A3, A3)
