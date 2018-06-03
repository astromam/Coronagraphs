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
from corono import corono_design as cd

from corono.utils import to_dict

#%% parameters
"""
Parameters
"""
corono_name = 'APLC' # 'APLC' or 'SP'

nPup = 500
nFPM = 50
nImg = 440
Fmax = 11
R    = 1

bw   = 0.1
nlam = 11

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

#%% Apodizer solution for the problems
"""
Apodizer saving 
"""
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