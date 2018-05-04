#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:36:52 2018

@author: mndiaye
"""

import numpy as np
import pylab as pl

from pathlib import Path
from corono import corono_design as cd

import matplotlib.gridspec as gridspec

from corono.utils import to_dict

#%% parameters
"""
Parameters
"""

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

nPup = 500
nFPM = 200
nImg = 110
Fmax = 11

params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilObs = PupilObs, rMask = rMask, LyotStopObs = LyotStopObs)

corono_name = 'APLC' # 'APLC' or 'SP'

fdir = Path('.').resolve()

fdir_ampl = fdir / 'results' / 'dat_ampl'
fdir_pyth = fdir / 'results' / 'dat_pyth'
fdir_plot = fdir / 'results' / 'plots'

fname = 'BPLC_obs={0:2d}_FPM={1:3d}_ls={2:2d}_IWA={3:03d}_OWA={4:03d}_BW={5:02d}_C={6:02d}_1D_N={7:04d}_nFPM={8:03d}'.format(
                int(PupilObs*100), int(rMask*100),int(LyotStopObs*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM))

fname_ampl = fname + '_gurobi_apod.dat'
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
read file obtained with gurobi
"""
fpath_pyth = fdir_pyth / fname_pyth
test_pyth = np.loadtxt(fpath_pyth)
rApod_pyth = test_pyth[:, 0]
Apod_pyth  = test_pyth[:, 1] 


#%%
"""
read file obtained with ampl
"""
fpath_ampl = fdir_ampl / fname_ampl
test_ampl = np.loadtxt(fpath_ampl)
rApod_ampl = test_ampl[:, 0]
Apod_ampl  = test_ampl[:, 1] 

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
fname_pl = fname + '_apodizers_tran.pdf'
fpath = fdir_plot / fname_pl

pl.figure(40)
pl.clf()
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
ax1 = pl.subplot(gs[0])
ax1.plot(corono0.r, Apod_pyth/Apod_pyth.max(), label='gurobipy')
ax1.plot(rApod_ampl*2, Apod_ampl/Apod_ampl.max(), label='ampl + gurobi')
ax1.set_ylabel('Apodizer amplitude transmission')
ax1.legend()
ax2 = pl.subplot(gs[1])
ax2.plot(corono0.r, Apod_pyth/Apod_pyth.max() - Apod_ampl/Apod_ampl.max())
ax2.set_xlabel(r'Pupil radius r')
ax2.set_ylabel('Difference')
pl.tight_layout()
pl.show()
pl.savefig(str(fpath))

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
poly_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth)
poly_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth)

poly_direct_image2 = corono0.compute_direct_intensity_1d(Apod_ampl)
poly_corono_image2 = corono0.compute_corono_intensity_1d(Apod_ampl)

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

fname_pl = fname + '_intensity.pdf'
fpath = fdir_plot / fname_pl
pl.figure(5)
pl.clf()
#pl.title('Intensity profiles of the coronagraphic images')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label='gurobipy')
pl.semilogy(corono0.xi,poly_corono_image2/poly_direct_image2.max(),label='ampl + gurobi')
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