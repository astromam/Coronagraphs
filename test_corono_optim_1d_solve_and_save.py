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

PupilObs    = 0.20
rMask       = 4.0
LyotStopObs = 0.40

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 5.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 8.0

# tau (integrated Pupil transmission)
tau   = 0.5

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilObs)*1.0
LyotStop1d   = (r>LyotStopObs)*1.0

params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, LyotStopObs = LyotStopObs,
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

fname_pyth = fname + '_guropy_apod.dat'

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
problem1 = co1d.MaxTau(corono=corono0, **params)
# Maximization of the contrast under L1-norm
#problem1 = co1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
# Maximization of the contrast under L-infinite norm
#problem1 = co1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
t1 = time.time()
print('problem definition time       : {0:.2f}s'.format(t1-t0))

#%%
"""
Problem solving
"""
print('problem solving')
t0 = time.time()
m1 = problem1.compute_gurobi_model()
Apod_pyth = problem1.solve_model()
t1 = time.time()

print('optimization time             : {0:.2f}s'.format(t1-t0))

import pylab as pl
pl.figure(1)
pl.clf()
pl.plot(Apod_pyth)
pl.show()


#%% Apodizer solution for the problems
"""
Apodizer saving 
"""
fpath_pyth = fdir_pyth / fname_pyth

test0 = np.zeros((nPup, 2))
test0[:, 0] = corono0.r/2
test0[:, 1] = Apod_pyth

np.savetxt(fpath_pyth, test0)
