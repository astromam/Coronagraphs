#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:52:23 2018

@author: mndiaye
"""

import numpy as np
import time
from pathlib import Path

from corono import corono_design as cd
from corono import corono_optim_2d as co2d
from corono.utils import to_dict

from astropy.io import fits

#%% parameters
"""
Parameters
"""
# Telescope name
pupil_name = 'sbr' # 'vlt' or 'sbr'

#nPup = corono0.params['nPup']
nPup = 200

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  4.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 4

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
corono_name   = 'SP' # 'SP' or 'APLC'
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym = True

#nlam
nlam=5

do_fits = True

#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = Path('./pupils/2D/').resolve()
fname = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
fpath = fdir / fname

Pupil2d = fits.getdata(fpath)
LyotStop2d = fits.getdata(fpath)


params = to_dict(nPup=nPup, rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2=CtrBtwnPix2, nlam=nlam, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym)

#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'SP':
    corono0 = cd.SP2d(**params)
else:
    corono0 = cd.APLC2d(**params)

#%%
"""
Problem defintion
"""
t0 = time.time()
# Maximization of the integrated amplitude transmission of the apodizer
problem1 = co2d.MaxTau(corono=corono0, **params)
# Maximization of the contrast under L1-norm
#problem2 = co2d.MaxContrast(corono=corono0, Lnorm='L1',**params)
# Maximization of the contrast under L-infinite norm
#problem3 = co2d.MaxContrast(corono=corono0, Lnorm='Linf',**params)

#%% Gurobi model of the problems
"""
Gurobi models
"""
m1 = problem1.compute_gurobi_model()
#m2 = problem2.compute_gurobi_model()
#m3 = problem3.compute_gurobi_model()

#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
Apod1 = problem1.solve_model()
#Apod2 = problem2.solve_model()
#Apod3 = problem3.solve_model()
t1 = time.time()
print('Pupil2dSym:{0}, total computation time: {1:.2f}s'.format(Pupil2dSym, t1-t0))

#%% Display of the apodizer
"""
Generation of full apodizer for quarter pupil optimization
"""
Apod1_2d = np.reshape(Apod1, (corono0.nPup, corono0.nPup))
#Apod2_2d = np.reshape(Apod2, (corono0.nPup, corono0.nPup))
#Apod3_2d = np.reshape(Apod3, (corono0.nPup, corono0.nPup))

if Pupil2dSym == True:
    Apod1_2d += np.flip(Apod1_2d, axis=0)
    Apod1_2d += np.flip(Apod1_2d, axis=1)  

#%%
"""
Save apodizer
"""
fdir = Path('./results/2D').resolve()

if corono_name == 'SP':
    fname_gen  = 'SP00_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}.fits'
else:
    fname_gen  = 'APLC_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}.fits'

fpath = fdir / pupil_name / fname_gen.format(**{key: corono0.params[key] for key in corono0.params})

if do_fits is True:
    fits.writeto(fpath, Apod1_2d, overwrite=True)