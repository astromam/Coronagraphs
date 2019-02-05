#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:52:23 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""
import numpy as np
import time
import os
from pathlib import Path
import corono as coro
from astropy.io import fits

t0_total = time.time()


working_path = '/Users/jnoss/dev/makidon-labs/48-test/'

pickle_jar = 'model'
use_pickled = None

"""
Apodizers for comparison

apodizers/HiCAT/Test_S_telserv3_HiCAT_MaxTau_nPup=0048_nFPM=050_APLC_rMask=4.271_IWA=5.0_OWA=10.0_BW=0.10_nlam=03_C=8.0_LS-Ann-bw-ID345-OD0807_gurobipy.fits
apodizers/HiCAT/Test_S_telserv3_HiCAT_MaxTau_nPup=0048_nFPM=050_APLC_rMask=4.271_IWA=5.0_OWA=10.0_BW=0.10_nlam=03_C=8.0_LS-Ann-bw-ID345-OD0807_stdgrb.fits
"""

#%% parameters
"""
Parameters
"""

#logging/debugging switches
slvLogToConsole = 1
allLogToConsole = 1

#Typpe of coronagraph, problem, and solver
# MaxTau - Maximization of the integrated amplitude transmission of the apodizer
# MaxContrastL1 - Maximization of the contrast under L1-norm
# MaxContrastLinf - Maximization of the contrast under L-infinite norm
corono_name  = 'APLC' # 'SP' or 'APLC'
problem_name = 'MaxTau' # 'MaxTau' # ,'MaxContrastLinf' # 'MaxContrastL1' #
solver       = 'gurobipy' #,'stdgrb' #  'gurobipy', 'scipy.linprog'

#some parameters that gets fed to the solver...?
slvCrossover    = 0
slvMethod       = 2

# Centering of disk - if True, the disk is centered between four pixels
CtrBtwnPix  = True
CtrBtwnPix2 = True

# Telescope aperture type and size and sampling
pup  = 'HiCAT' # 'vlt' or 'sbr' or 'lvr'
nPup = 48
Fmax2d = 32.5 
nImg2d = 65

# mask sampling -> units, also is this diameter or radius??
nFPM = 50

# mask radius in lam0/D (which D??) units
rMask = 8.543/2

fast_ft = False
# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 3.75#5.00 #3.75
rho1 = 15.00#10.00 #15.00

# contrast in the dark region
cDarkHole = 8.0

# tau (integrated Pupil transmission)
tau   = 0.4

#bandwidth and number of wavelengths used
bw   = 0.10
nlam = 3

# lyot stop inner and outer diameter, percentage of aperture -> inscribed/circumscribed??? 
LS_ID = 34.5
LS_OD = 80.7

lsid = int(LS_ID*10)
lsod = int(LS_OD*10)

#lyot stop robustness
LSRobustness = False
# maximum pixel shift along a given axis for Lyot stop; total robostness is x2 pix_max
pix_max   = 1

if LSRobustness == False:
	pix_max = 0

#symmetry keyword; if True quarter plane symmetry is used, if false full plane
Pupil2dSym  = False

# Computes matrices to add constraints that minimizes the islands in the apodizer transmission.
MinIsland   = False

#Constraint on the apodizer first derivative through the auxiliary variables with :math:`\int_{P_0} (v^{+}(r) + v^{-}(r))dr \leq \delta`.
#Only relevant if MinIsland is true
FirstDerGlobalLim = 100.

# ???
#Binarity    = False
#BinarityReg = 0.1

#save result to fits file
do_fits = True

print('Parameters loaded')
print()

#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = os.path.join(working_path, 'input_files/')
if pup == 'lvr':
    fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup)
    fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup)
elif pup == 'HiCAT':
	fname_pup = 'apertures/HiCAT-Aper_F-N{0:04d}_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004'.format(nPup)
	fname_lys = 'lyot_stops/HiCAT-Lyot_F-N{0:04d}_LS-Ann-bw-ID{1:04d}-OD{2:04d}-SpX0036'.format(nPup,lsid,lsod)
else:
    fname_pup = 'pupil={0}_nPup={1}.fits'.format(pup, nPup)
    fname_lys = 'pupil={0}_nPup={1}.fits'.format(pup, nPup)

fpath_pup = fdir + fname_pup + '.fits'
fpath_lys = fdir + fname_lys + '.fits'

Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)

print('Input files imported and loaded')

# List of Lyot stops for the design optimization
LyotStop2d_t = [LyotStop2d]

# List construction for Lyot stop position shifts and List construstion for the Lyot stops
pix_t = []
if pix_max >= 1 and LSRobustness == True:
    pix_pos_t = 1 + np.arange(pix_max)
    pix_neg_t = - pix_pos_t
    pix_t = list(-pix_pos_t) + list(pix_pos_t)
    pix_t.sort()


#for j in range(2):
#    LyotStop2d_t.append(np.roll(LyotStop2d, pix_max, axis=j))


for j in range(2):
    for i in range(len(pix_t)):
        roll_LS = np.roll(LyotStop2d, pix_t[i], axis=j)
        LyotStop2d_t.append(roll_LS)

# number of coronagraph configuration
ncorono      = len(LyotStop2d_t)
print('# of coronagraph configurations: {0}'.format(ncorono))

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = coro.to_dict(corono_name = corono_name, problem_name = problem_name, solver = solver,
				 slvCrossover = slvCrossover, slvMethod = slvMethod,
			     CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
			     pupil_name = pup, nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rMask=rMask, rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 nlam=nlam, bw=bw,
                 LSID = lsid, LSOD = lsod,
                 LSRobustness = LSRobustness, pix_max = pix_max,
                 Pupil2dSym = Pupil2dSym,
                 slvLogToConsole = slvLogToConsole,
                 allLogToConsole = allLogToConsole,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, fast_ft=fast_ft,
                 pickle_jar=pickle_jar, use_pickled=use_pickled
                 )

# list of parameters for each coronagraph configuration
params_t = []
for k in range(ncorono):
    params_t.append(coro.update_params(params, LyotStop2d=LyotStop2d_t[k])) 

# initialization of coronagraph list
corono_t = []

#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'SP':
    corono_t = coro.design.SP2d(**params)
elif corono_name == 'APLC':
    for k in range(ncorono):
        corono_t.append(coro.design.APLC2d(**params_t[k])) 
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
if problem_name == 'MaxTau':   
    problem1 = coro.optim_2d.MaxTau(corono=corono_t, **params)
elif problem_name == 'MaxContrastL1':
    problem1 = coro.optim_2d.MaxContrast(corono=corono_t, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    problem1 = coro.optim_2d.MaxContrast(corono=corono_t, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#%% Apodizer solution for the problems


"""
Apodizer solutions
"""
t0 = time.time()
Apod1 = problem1.solve_model()
t1 = time.time()
print('optimization time             : {0:.2f}s'.format(t1-t0))

#%% Display of the apodizer
"""
Generation of full apodizer for quarter pupil optimization
"""
Apod1_2d = np.reshape(Apod1, (corono_t[0].nPup, corono_t[0].nPup))

if Pupil2dSym == True:
        Apod1_2dtmp =  Apod1_2d[corono0.nPup//2:, corono0.nPup//2:]
        Apod1_2d[:corono0.nPup//2, corono0.nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
        Apod1_2d[:, :corono0.nPup//2]          = np.flip(Apod1_2d[:, corono0.nPup//2:], axis=1)
        
#%%
"""
Save apodizer
"""
fdir = Path(os.path.join(working_path, 'apodizers/HiCAT/')).resolve()
if not os.path.exists(fdir):
    os.makedirs(fdir)
    
fname = 'Test_S_jnoss_' + problem1.get_filename() + '.fits'
fpath = fdir / fname

if do_fits is True:
    fits.writeto(fpath, Apod1_2d, overwrite=True)
    
t1_total = time.time()
print('total run time             : {0:.2f}s'.format(t1_total-t0_total))
    
    