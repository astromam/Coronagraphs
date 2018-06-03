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

from corono.utils import to_dict

#%% parameters
"""
Parameters
"""
corono_name = 'APLC' # 'APLC' or 'SP' or 'HDZPM' or 'HTZPM'

nPup = 500
nFPM = 50
nImg = 110
Fmax = 22
R    = 1

bw   = 0.2
nlam = 5
nlambis = 11

PupilObs    = 0.14

rMask       = 2.5

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75

LyotStopObs = 0.28
LyotStopIns = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 3.0
rho1 = 20.0

# contrast in the dark region
cDarkHole = 7.0

# tau (integrated Pupil transmission)
tau   = 0.5

r            = np.arange(nPup)*R/nPup + R/(2*nPup)
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
    fname = '{0}_obs={1:2d}_FPM={2:3d}_ls={3:2d}_IWA={4:03d}_OWA={5:03d}_BW={6:02d}_C={7:02d}_1D_N={8:04d}_nFPM={9:03d}'.format(
                corono_name, int(PupilObs*100), int(rMask*100),
                int(LyotStopObs*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM))
elif corono_name == 'DZPMbis':
    fname = '{0}_obs={1:2d}_FPM1={2:3d}_FPM2={3:3d}_ls={4:2d}_IWA={5:03d}_OWA={6:03d}_BW={7:02d}_C={8:02d}_1D_N={9:04d}_nFPM={10:03d}'.format(
                corono_name, int(PupilObs*100), int(rMask1*100), int(rMask2*100), 
                int(LyotStopObs*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM)) 
else:
    fname = '{0}_obs={1:2d}_FPM1={2:3d}_FPM2={3:3d}_FPM3={4:3d}_ls={5:2d}_IWA={6:03d}_OWA={7:03d}_BW={8:02d}_C={9:02d}_1D_N={10:04d}_nFPM={11:03d}'.format(
                corono_name, int(PupilObs*100), 
                int(rMask1*100), int(rMask2*100), int(rMask3*100), 
                int(LyotStopObs*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM)) 

fname_pyth     = fname + '_guropy_apod_test.dat'
fname_pyth_nm0 = fname + '_guropy_apod_test_nm0.dat'

#%%  
""" 
Coronagraph defintion
"""

fpath_pyth_nm0 = fdir_pyth / fname_pyth_nm0 
x_end = np.loadtxt(fpath_pyth_nm0)

if corono_name == 'APLC':
    rMask       = x_end[0]
    LyotStopObs = x_end[1]
    LyotStopIns = x_end[2]
    print('mask radius          : {0:.4f} lambda/D'.format(rMask))
    print('Lyot Stop obstruction: {0:.4f}'.format(LyotStopObs))
    print('Lyot Stop ins. size  : {0:.4f}'.format(LyotStopIns))
elif corono_name == 'HDZPM':
    rMask1      = x_end[0]
    rMask2      = x_end[1]
    OPDx2       = x_end[2]
    LyotStopObs = x_end[3]
    LyotStopIns = x_end[4]
    print('mask radius 1         : {0:.4f} lambda_0/D'.format(rMask1))
    print('mask radius 2         : {0:.4f} lambda_0/D'.format(rMask2))
    print('OPD 2                 : {0:.4f} lambda_0'.format(OPDx2))
    print('Lyot Stop obstruction: {0:.4f}'.format(LyotStopObs))
    print('Lyot Stop ins. size  : {0:.4f}'.format(LyotStopIns))    
else: 
    rMask1      = x_end[0]
    rMask2      = x_end[1]
    rMask3      = x_end[2]
    OPDx2       = x_end[3]
    OPDx3       = x_end[4]
    LyotStopObs = x_end[5]
    LyotStopIns = x_end[6]
    print('mask radius 1         : {0:.4f} lambda_0/D'.format(rMask1))
    print('mask radius 2         : {0:.4f} lambda_0/D'.format(rMask2))
    print('mask radius 3         : {0:.4f} lambda_0/D'.format(rMask3))
    print('OPD 2                 : {0:.4f} lambda_0'.format(OPDx2))
    print('OPD 3                 : {0:.4f} lambda_0'.format(OPDx3))
    print('Lyot Stop obstruction: {0:.4f}'.format(LyotStopObs))
    print('Lyot Stop ins. size  : {0:.4f}'.format(LyotStopIns)) 

    
r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilObs)*1.0
LyotStop1d   = (r>LyotStopObs)*(r<LyotStopIns)*1.0

params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlambis,
                 PupilObs = PupilObs, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopObs = LyotStopObs,
                 LyotStopIns = LyotStopIns,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d)


if corono_name == 'APLC':
    corono0 = cd.APLC1d(**params)
elif corono_name == 'SP':
    corono0 = cd.SP1d(**params)
elif corono_name == 'HDZPM':
    corono0 = cd.DZPM1dbis(**params)
else:
    corono0 = cd.TZPM1dbis(**params)
    
#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
fpath_pyth = fdir_pyth / fname_pyth
test0 = np.loadtxt(fpath_pyth)
Apod_pyth = test0[:, 1]

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
fname_pl = fname + '_apodizers_tran_test_nm0.pdf'
fpath = fdir_plot / fname_pl

pl.figure(1)
pl.clf()
pl.plot(corono0.r, Apod_pyth/Apod_pyth.max(), label='gurobipy')
pl.xlabel(r'Pupil radius r')
pl.ylabel('Apodizer amplitude transmission')
pl.axvline(x=corono0.PupilObs, ymin=-0.5, ymax =2, linewidth=1, color='g', linestyle='--')
pl.axvline(x=1.0, ymin=-0.5, ymax =2, linewidth=1, color='g', linestyle='--')
pl.axvline(x=corono0.LyotStopObs, ymin=-0.5, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.LyotStopIns, ymin=-0.5, ymax =2, linewidth=1, color='r', linestyle='--')
pl.legend()
pl.tight_layout()
pl.show()
pl.savefig(str(fpath))

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
poly_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth)
poly_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth)

mono_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth, poly=False)
mono_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth, poly=False)

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

fname_pl = fname + '_intensity_test_nm0.pdf'
fpath = fdir_plot / fname_pl
pl.figure(2)
pl.clf()
#pl.title('Intensity profiles of the coronagraphic images')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label='gurobipy')
if corono_name == 'APLC':
    pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
elif corono_name == 'HDZPM':
    pl.axvline(x=corono0.rMask1, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rMask2, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')    
else:
    pl.axvline(x=corono0.rMask1, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rMask2, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')    
    pl.axvline(x=corono0.rMask3, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')    

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

fname_pl = fname + '_intensity_mono_test_nm0.pdf'
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
if corono_name == 'APLC':
    pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
elif corono_name == 'HDZPM':
    pl.axvline(x=corono0.rMask1, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rMask2, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')    
else:
    pl.axvline(x=corono0.rMask1, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rMask2, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')    
    pl.axvline(x=corono0.rMask3, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')    

pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-12, 1e-3)
pl.legend(loc=2)
pl.tight_layout()
pl.savefig(str(fpath))

#%%
pl.show()