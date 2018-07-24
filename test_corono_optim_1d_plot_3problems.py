#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:40:02 2018

@author: mndiaye
"""
import pylab as pl
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

nPup = 500
nFPM = 50
nImg = 440
Fmax = 11
R    = 1

bw   = 0.1
nlam = 5
nlambis =11

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

params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopObs = LyotStopObs,
                 LyotStopIns = LyotStopIns,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, 
                 corono_name = corono_name)

#%%
"""
Working directory
"""
fdir = Path('.').resolve() / 'results' / '1D'
fdir_pyth = fdir  / 'dat_pyth'
fdir_plot = fdir  / 'plots'

if not os.path.exists(fdir_pyth):
    os.makedirs(fdir_pyth)      

if not os.path.exists(fdir_plot):
    os.makedirs(fdir_plot)  

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
Apodizer saving 
"""
fname1 = problem1.get_filename() + '.dat'
fname2 = problem2.get_filename() + '.dat'
fname3 = problem3.get_filename() + '.dat'

fpath1 = fdir_pyth / fname1
fpath2 = fdir_pyth / fname2
fpath3 = fdir_pyth / fname3

test1 = np.loadtxt(fpath1)
test2 = np.loadtxt(fpath2)
test3 = np.loadtxt(fpath3)

Apod1 = test1[:, 1]
Apod2 = test2[:, 1]
Apod3 = test3[:, 1]

#%%
"""
Display of the matrices
"""
fname_A1 = problem1.get_filename() + '_A.dat'
fname_A2 = problem2.get_filename() + '_A.dat'
fname_A3 = problem3.get_filename() + '_A.dat'

fpath_A1 = fdir_pyth / fname_A1
fpath_A2 = fdir_pyth / fname_A2
fpath_A3 = fdir_pyth / fname_A3

A1 = np.loadtxt(fpath_A1)
A2 = np.loadtxt(fpath_A2)
A3 = np.loadtxt(fpath_A3)

pl.figure(1)
pl.clf()
pl.title('Matrix for MaxTau problem')
pl.imshow(abs(A1.T)**0.25)

pl.figure(2)
pl.clf()
pl.title(r'Matrix for MaxContrast problem, L$_1$-norm')
pl.imshow(abs(A2.T)**0.25)
#
pl.figure(3)
pl.clf()
pl.title(r'Matrix for MaxContrast problem, L$_\infty$-norm')
pl.imshow(abs(A3.T)**0.25)


#%% Display of the apodizer
"""
Plot display of the apodizers
"""
pl.figure(4)
pl.clf()
pl.title('Transmission profiles of the apodizers')
pl.plot(corono0.r, Apod1/Apod1.max(), label='MaxTau')
pl.plot(corono0.r, Apod2, label=r'MaxContrast, L$_1$-norm')
pl.plot(corono0.r, Apod3, label=r'MaxContrast, L$_\infty$-norm')
pl.xlabel(r'Pupil radius r')
pl.ylabel('Apodizer amplitude transmission')
pl.legend()

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis) 

if corono_name == 'APLC':
    corono0 = coro.design.APLC1d(**params2)
elif corono_name == 'SP':
    corono0 = coro.design.SP1d(**params2)
elif corono_name == 'HDZPM':
    corono0 = coro.design.HDZPM1d(**params2)
elif corono_name == 'HTZPM':
    corono0 = coro.design.HTZPM1d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))


poly_direct_image1 = corono0.compute_direct_intensity_1d(Apod1)
poly_corono_image1 = corono0.compute_corono_intensity_1d(Apod1)
poly_direct_image2 = corono0.compute_direct_intensity_1d(Apod2)
poly_corono_image2 = corono0.compute_corono_intensity_1d(Apod2)
poly_direct_image3 = corono0.compute_direct_intensity_1d(Apod3)
poly_corono_image3 = corono0.compute_corono_intensity_1d(Apod3)

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""
pl.figure(6)
pl.clf()
pl.title('Intensity profiles of the coronagraphic images')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label='MaxTau')
pl.semilogy(corono0.xi,poly_corono_image2/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
pl.semilogy(corono0.xi,poly_corono_image3/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
#pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()

#%%
pl.show()