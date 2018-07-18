#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:26:11 2018

@author: mndiaye
"""
import numpy as np
import pylab as pl
import os

from pathlib import Path
from corono import corono_design as cd
from corono import corono_optim_1d as co1d

from corono.utils import to_dict, update_params

#%% parameters
"""
Parameters
"""
corono_name  = 'APLC' # 'APLC' or 'SP'
problem_name = 'MaxTau' # 'MaxContrastL1' #,'MaxContrastLinf' # 'MaxTau' # ,   
solver       = 'stdgrb' # 'stdgrb', 'gurobipy', 'scipy.linprog'

nPup = 500
nFPM = 50
nImg = 110
Fmax = 11
R    = 1

bw   = 0.1
nlam = 5
nlambis = 11

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
cDarkHole = 10.0

# tau (integrated Pupil transmission)
tau   = 0.5

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
fdir = Path('.').resolve()

fdir_pyth = fdir / 'results' / '1D' / 'dat_pyth'
fdir_plot = fdir / 'results' / '1D' / 'plots'

if not os.path.exists(fdir_plot):
    os.makedirs(fdir_plot)
    
if not os.path.exists(fdir_pyth):
    os.makedirs(fdir_pyth)    

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
    problem1 = co1d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = co1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = co1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
fname_pyth = problem1.get_filename()
fpath_pyth = fdir_pyth / fname_pyth
test0 = np.loadtxt(fpath_pyth)
Apod_pyth = test0[:, 1]

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
fname = fname_pyth.replace('.dat', '')
fname_pl = fname + '_apodizers_tran.pdf'
fpath = fdir_plot / fname_pl

pl.figure(1)
pl.clf()
pl.plot(corono0.r, Apod_pyth/Apod_pyth.max(), label='gurobipy')
pl.xlabel(r'Pupil radius r')
pl.ylabel('Apodizer amplitude transmission')
pl.legend()
pl.tight_layout()
pl.show()
pl.savefig(str(fpath))

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
#params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
#                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
#                 bw = bw, nlam = nlambis,
#                 PupilObs = PupilObs, rMask = rMask, 
#                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
#                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
#                 LyotStopObs = LyotStopObs,
#                 LyotStopIns = LyotStopIns,
#                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
#                 solver = solver, problem_name = problem_name,
#                 corono_name = corono_name)

params2 = update_params(params, nlam=nlambis) 
#params2 = params.copy()
#params2['nlam'] = nlambis

if corono_name == 'APLC':
    corono0 = cd.APLC1d(**params2)
elif corono_name == 'SP':
    corono0 = cd.SP1d(**params2)
elif corono_name == 'HDZPM':
    corono0 = cd.HDZPM1d(**params2)
elif corono_name == 'HTZPM':
    corono0 = cd.HTZPM1d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

poly_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth)
poly_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth)

mono_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth, poly=False)
mono_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth, poly=False)

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

fname_pl = fname + '_intensity.pdf'
fpath = fdir_plot / fname_pl
pl.figure(2)
pl.clf()
#pl.title('Intensity profiles of the coronagraphic images')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label='gurobipy')
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-12, 1e-3)
pl.legend()
pl.tight_layout()
pl.savefig(str(fpath))

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the monochromatic intensity profiles of the coronagraphic images
"""

values = range(nlambis)
colors = pl.cm.rainbow(np.linspace(0,1,nlambis))

fname_pl = fname + '_intensity_mono.pdf'
fpath = fdir_plot / fname_pl
pl.figure(3)
pl.clf()
#pl.title('Intensity profiles of the coronagraphic images')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
for i in range(corono0.nlam):
    pl.semilogy(corono0.xi,mono_corono_image1[i]/mono_direct_image1[(corono0.nlam+1)//2].max(), 
                label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[i]), color = colors[i])
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-12, 1e-3)
pl.legend()
pl.tight_layout()
pl.savefig(str(fpath))

#%%
pl.show()