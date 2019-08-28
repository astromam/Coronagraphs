#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:43:13 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""
import numpy as np
import pylab as pl
import os

from pathlib import Path
import corono as coro

#%% parameters
"""
Parameters
"""
corono_name  = 'DZPM' # 'APLC' or 'SP' or 'HDZPM' or 'HTZPM'
problem_name = 'MaxContrastL1'#'MaxContrastLinf' # ,'MaxContrastL1' # #  'MaxContrastL1', 
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
nImg = 220
Fmax = 22
R    = 1

bw   = 0.2
nlam = 5
nlambis = 11

PupilID    = 0.14

rMask       = 4.0  #2.3 or 4.0

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

r            = np.arange(nPup)*R/nPup + R/(2*nPup)
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
fdir = Path('../../results/1D/').resolve()

fdir_pyth = fdir / 'dat_pyth'
fdir_plot = fdir / 'plots'

if not os.path.exists(fdir_plot):
    os.makedirs(fdir_plot)
    
if not os.path.exists(fdir_pyth):
    os.makedirs(fdir_pyth)   

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

#%%
fname_gen = problem0.get_filename()
fname_pyth_nm0 = fname_gen + '_nm0.dat'
fpath_pyth_nm0 = fdir_pyth / fname_pyth_nm0 
x_end = np.loadtxt(fpath_pyth_nm0)

if corono_name == 'DZPM': 
    rMask1, OPDx1, OPDx2, ome1, ome2, beta, LyotStopID, LyotStopOD = [x_end[i] for i in {0,2,3,4,5,6,7,8}]
    rMask2      = rMask1 + x_end[1]
    print('mask diameter 1         : {0:.3f} lambda_0/D'.format(2*rMask1))
    print('mask diameter 2         : {0:.3f} lambda_0/D'.format(2*rMask2))
    print('OPD 1                 : {0:.3f} lambda_0'.format(OPDx1))
    print('OPD 2                 : {0:.3f} lambda_0'.format(OPDx2))
    print('ome 1                 : {0:.3f} '.format(ome1))
    print('ome 2                 : {0:.3f} '.format(ome2))
    print('beta                  : {0:.3f} lambda_0'.format(beta))
    print('Lyot Stop obstruction : {0:.3f}'.format(LyotStopID))
    print('Lyot Stop ins. size   : {0:.3f}'.format(LyotStopOD)) 
else:
    raise NameError('{0}: Not a correct coronagraph for NM-optimization!'.format(corono_name))    

    
r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilID)*1.0
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

params = coro.update_params(params, rMask1 = rMask1, rMask2 = rMask2,
                                     OPDx1 = OPDx1, OPDx2 = OPDx2, 
                                     ome1 = ome1, ome2 = ome2, beta = beta, 
                                     LyotStop1d = LyotStop1d, LyotStopID = LyotStopID, 
                                     LyotStopOD = LyotStopOD) 

    
#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
fname_pyth = problem0.get_filename() + '_tmp.dat'
fpath_pyth = fdir_pyth / fname_pyth
test0 = np.loadtxt(fpath_pyth)
Apod_pyth = test0[:, 1]

#%%
"""
Computationn of the coronagraph transmission
"""
corono_throughput = 100.*np.sum(np.abs(Apod_pyth*Pupil1d*LyotStop1d)**2)/np.sum(np.abs(Pupil1d*LyotStop1d)**2)

print('Transmitted Energy (T.E.) throughput: {0:.1f}%'.format(corono_throughput))

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
fname_pl = fname_gen + '_apodizers_tran_test_nm0.pdf'
fpath    = fdir_plot / fname_pl

pl.figure(1)
pl.clf()
pl.plot(corono0.r, Apod_pyth/Apod_pyth.max())
pl.xlabel(r'Pupil radius r')
pl.ylabel('Apodizer amplitude transmission')
pl.axvline(x=corono0.PupilID, ymin=-0.5, ymax =2, linewidth=1, color='g', linestyle='--')
pl.axvline(x=1.0, ymin=-0.5, ymax =2, linewidth=1, color='g', linestyle='--')
pl.axvline(x=corono0.LyotStopID, ymin=-0.5, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.LyotStopOD, ymin=-0.5, ymax =2, linewidth=1, color='r', linestyle='--')
#pl.legend()
pl.tight_layout()
pl.show()
pl.savefig(str(fpath))

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
fname_gen  = problem0.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis) 

if corono_name == 'DZPM':
    corono0 = coro.design.DZPM1d(**params2)
else:
    raise NameError('{0}: Not a DZPM!'.format(corono_name))

poly_nostop_image1 = corono0.compute_nostop_intensity_1d()
poly_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth)
poly_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth)

mono_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth, poly=False)
mono_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth, poly=False)


#%%
"""
Computation of the Airy throughput
"""

EE_idx = corono0.xi <= 0.7

rel_Airy_throughput = 100.*np.sum(poly_direct_image1[EE_idx])/np.sum(poly_nostop_image1[EE_idx])

print('Encircled Energy (E.E.) throughput: {0:.1f}%'.format(rel_Airy_throughput))

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

fname_pl = fname_gen + '_intensity_test_nm0.pdf'
fpath = fdir_plot / fname_pl
pl.figure(2)
pl.clf()
pl.title('Broadband intensity profiles of the DZPM coronagraphic image')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max())
if corono_name == 'DZPM':
    pl.axvline(x=corono0.rMask1, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rMask2, ymin=-12, ymax =2, linewidth=1, color='g', linestyle='--')    
else:
    raise NameError('{0}: Not a DZPM!'.format(corono_name))

pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-6), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-12, 1e-3)
#pl.legend()
pl.tight_layout()
pl.savefig(str(fpath))

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the monochromatic intensity profiles of the coronagraphic images
"""

values = range(nlambis)
colors = pl.cm.rainbow(np.linspace(0,1,nlambis))

fname_pl = fname_gen + '_intensity_mono_test_nm0.pdf'
fpath = fdir_plot / fname_pl
pl.figure(3)
pl.clf()
pl.title('Monochromatic intensity profile of the DZPM coronagraphic images')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
for i in range(corono0.nlam):
    pl.semilogy(corono0.xi,mono_corono_image1[i]/mono_direct_image1[(corono0.nlam+1)//2].max(), 
                label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[i]), color = colors[i])
if corono_name == 'DZPM':
    pl.axvline(x=corono0.rMask1, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rMask2, ymin=-12, ymax =2, linewidth=1, color='g', linestyle='--')    
else:
    raise NameError('{0}: Not a DZPM!'.format(corono_name))

pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-6), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-12, 1e-3)
pl.legend(loc=2)
pl.tight_layout()
pl.savefig(str(fpath))

#%%
pl.show()