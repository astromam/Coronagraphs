#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:26:11 2018

@author: mndiaye
"""
import numpy as np
import pylab as pl
import time

from pathlib import Path
from corono import corono_design as cd
from corono import corono_optim_1d as co1d

from corono.utils import to_dict

#%% parameters
"""
Parameters
"""

bw   = 0.1
nlam = 3

PupilObs    = 0.14
rMask       = 4.0
LyotStopObs = 0.28

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

fdir_ampl = Path('/Users/mndiaye/Dropbox/central storage/AMPL/PupilDataFiles/1D/General/dat2/')

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

#%%
"""
Display of the matrices
"""
#A1, b1, c1 = problem1.compute_matrices()
#
#pl.figure(1)
#pl.clf()
#pl.title('Matrix for MaxTau problem')
#pl.imshow(abs(A1.T)**0.25)

#%% Gurobi model of the problems
"""
Gurobi models
"""
m1 = problem1.compute_gurobi_model()

#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
Apod1 = problem1.solve_model()
t1 = time.time()
print('optimization time with gurobipy: {0:.2f}s'.format(t1-t0))

#%%
fname_ampl = 'BPLC_obs={0:2d}_FPM={1:3d}_ls={2:2d}_IWA={3:03d}_OWA={4:03d}_BW={5:02d}_C={6:02d}_1D_N={7:04d}_nFPM={8:03d}_gurobi_apod.dat'.format(
                int(PupilObs*100), int(rMask*100),int(LyotStopObs*100),
                int(rho0*10),int(rho1*10),int(bw*100), int(cDarkHole),
                int(nPup), int(nFPM))

#fname_ampl = 'BPLC_obs=14_FPM=280_ls=28_IWA=050_OWA=100_BW=20_C=06_1D_N=0500_gurobi_apod.dat'
                
fpath_ampl = fdir_ampl / fname_ampl
test = np.loadtxt(fpath_ampl)
rApod2 = test[:, 0]
Apod2  = test[:, 1] 

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
pl.figure(4)
pl.clf()
pl.title('Optimization comparison \n Transmission profiles of the apodizers')
pl.plot(corono0.r, Apod1/Apod1.max(), label='gurobipy')
pl.plot(rApod2*2, Apod2/Apod2.max(), label='ampl + gurobi')
pl.xlabel(r'Pupil radius r')
pl.ylabel('Apodizer amplitude transmission')
pl.legend()

pl.figure(41)
pl.clf()
pl.title('Optimization comparison \n Transmission profiles of the apodizers')
pl.plot(corono0.r, Apod1/Apod1.max() - Apod2/Apod2.max(), label='difference')
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

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""
pl.figure(5)
pl.clf()
pl.title('Optimization comparison \n Intensity profiles of the coronagraphic images')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label='gurobipy')
pl.semilogy(corono0.xi,poly_corono_image2/poly_direct_image2.max(),label='ampl + gurobi')
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-problem1.params['cDarkHole']), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()

#%%
pl.show()