#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:44:36 2023

@author: mndiaye
"""
import numpy as np
import time
import os
from pathlib import Path

import corono as coro

from astropy.io import fits

import sys
import pwd

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

try:
    import gurobipy as gb
except ImportError:
    gb = False
    print('gb is False')   

#%% parameters
"""
Parameters
"""
# Telescope name
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxContrastLinf' # 'MaxTau' # ,'MaxContrastLinf' # 'MaxContrastL1' #
solver       = 'gurobipy' #,'stdgrb' #  'gurobipy', 'scipy.linprog'
slvLogToConsole = 1
slvCrossover    = 0
slvMethod       = 2
slvSparse       = 0
allLogToConsole = 1

MinIsland   = False
Binarity    = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1

#nPup = corono0.params['nPup']
nPup = 200
nFPM = 50
Fmax2d = 45#22.5
nImg2d = 90#45

# mask radius in lam0/D units
rMask = 2.8

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  5.0
rho1 = 20.0

# contrast in the dark region
cDarkHole = 7.0

# tau (integrated Pupil transmission)
tau   = 0.5

# CtrBtwnPix2
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True
ImPart = False 

#nlam
bw   = 0.1
nlam = 1

do_fits = True

#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = Path('../../../data/2D/pupils/').resolve()
#fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()
# if user == 'mndiaye':
#     if syst == 'darwin':
#         fdir = Path('~/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').expanduser()
#         sim_case = 'test' # 'test' or 'server'
#     elif syst == 'linux':
#         fdir = Path('/home/mndiaye/python/Coronagraphs/data/2D/pupils').resolve()
#         sim_case = 'server' # 'test' or 'server'            
#     else:
#         raise ValueError('Unknown operating system {0}'.format(user))
# else:
#     raise ValueError('Unknown user {0}'.format(user))


if pupil_name == 'lvr':
    fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
    fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
else:
    fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)

fpath_pup = fdir / fname_pup
fpath_lys = fdir / fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)


if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name,
                 slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 slvSparse = slvSparse,
                 allLogToConsole = allLogToConsole,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg,
                 ImPart = ImPart)

#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
# if problem_name == 'MaxTau':
#     # Maximization of the integrated amplitude transmission of the apodizer
#     problem1 = coro.optim_2d.MaxTau(corono=corono0, **params)
# elif problem_name == 'MaxContrastL1':
#     # Maximization of the contrast under L1-norm
#     problem1 = coro.optim_2d.MaxContrast(corono=corono0, Lnorm='L1',**params)
# elif problem_name == 'MaxContrastLinf':
#     # Maximization of the contrast under L-infinite norm
#     problem1 = coro.optim_2d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
# else:
#     raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))
  
#%%
"""
Problem definition without class
"""

idx_pup = 0
TR = 0
Pupil_vec = 0
 
npp = 0
nPsiD = 0
ncorono = 1
neps = 1 
ndz = 0

I0 = np.ones(ndz)
I0 = I0[None,:]

nI1 = 1 
if ImPart is True:
    nI1 = 2
I1 = np.ones(nI1*nlam*ndz)
I1 = I1[None,:]

A = np.zeros((npp+neps, nPsiD*2+ndz+1+npp*2))

# Compute coronagraph response matrix
A[0:npp,0:nPsiD] = compute_response_matrices(corono_t[k])
A[0:npp,nPsiD:2*nPsiD] = -A[0:npp,0:nPsiD]

# Compute constraints on the coronagraphic electric field
A[npp:npp+neps,0:nPsiD] = -I1
A[npp:npp+neps,nPsiD:2*nPsiD] = A[npp:npp+neps,0:nPsiD]

# Compute constraint on the auxiliary variable epsilon
A[0:npp,2*nPsiD:2*nPsiD+ndz] = 0
A[npp:npp+neps,2*nPsiD:2*nPsiD+ndz] = -I0


# Compute constraint on the integral of the apodizer transmission
A[0:npp,2*nPsiD+ndz] = -Pupil_vec[idx_pup]/TR
#A[npp:npp+neps,2*nPsiD+ndz:2*nPsiD+ndz+1] = 0
#A[npp+neps:npp+neps,2*nPsiD+ndz:2*nPsiD+ndz+1] = 0

b = np.zeros((nPsiD*2+ndz+1+npp*2))
b[nPsiD*2+ndz] = -tau
b[nPsiD*2+ndz+1+npp:] = 1

c = np.zeros(npp+neps)
c[npp+neps] = 1



Apod1 = 0

print('generating gurobi model')

# Compute the length of the A matrix along axis=1                          
nA = np.shape(A)[1]

# Create a new model  
model = gb.Model("LP max C new")

# Create variables
ApodEpsTmp = model.addVars(npp + neps, lb=0.0, name="ApodEpsTmp")        

# Set objective
model.setObjective(gb.quicksum((c[i+npp]*ApodEpsTmp[i+npp] 
        for i in range(neps))), gb.GRB.MINIMIZE)

# Add constraint:
model.addConstrs((gb.quicksum((ApodEpsTmp[i]*A[i,j] 
        for i in range(npp + neps) if A[i,j])) <=  b[j] 
        for j in np.arange(nA)), "cpos")

# Update model
model.update()
    


    
    

#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
# t0 = time.time()
# Apod1 = problem1.solve_model()
# t1 = time.time()
# print('optimization time             : {0:.2f}s'.format(t1-t0))

#%%
t0 = time.time()

# solve problem with gurobipy package
print('solving problem with gurobipy package')
try:               
    model.Params.Method       = slvMethod
    model.Params.LogToConsole = slvLogToConsole
    model.Params.Crossover    = slvCrossover
        
    print('preparing to save optimization problem')
    print('ok')
    
    model.optimize()

    for i, val in enumerate(idx_pup):
        Apod1[val] = model.getVars()[i].x
    
except gb.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')

t1 = time.time()
print('optimization time             : {0:.2f}s'.format(t1-t0))


#%% Display of the apodizer
"""
Generation of full apodizer for quarter pupil optimization
"""
Apod1_2d = np.reshape(Apod1, (corono0.nPup, corono0.nPup))

if Pupil2dSym == True:
        Apod1_2dtmp =  Apod1_2d[corono0.nPup//2:, corono0.nPup//2:]
        Apod1_2d[:corono0.nPup//2, corono0.nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
        Apod1_2d[:, :corono0.nPup//2]          = np.flip(Apod1_2d[:, corono0.nPup//2:], axis=1)
        
#%%
"""
Save apodizer
"""
fdir = Path('../../../results/2D/dat_pyth').resolve() / pupil_name
#fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
if not os.path.exists(fdir):
    os.makedirs(fdir)
    
fname = problem1.get_filename() + '.fits'
fpath = fdir / fname

if do_fits is True:
    fits.writeto(fpath, Apod1_2d, overwrite=True)
    
    