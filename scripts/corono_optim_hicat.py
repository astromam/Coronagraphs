#Design optimization of coronagraph for 2D geometry

## Initialization
## This section lists all the required packages

import numpy as np
import pylab as pl
import time
import os
#from pathlib import Path

from astropy.io import fits
import corono as coro

t0_total = time.time()

## Parameters
## This section lists all the parameters for the coronagraph and the optimization problem.

# coronagraph, problem, and solver types 
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'HiCAT' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxTau' # 'MaxTau' # ,'MaxContrastLinf' # 'MaxContrastL1' #
solver       = 'stdgrb' #,'stdgrb' #  'gurobipy', 'scipy.linprog'

# keywords for solver
slvLogToConsole = 0
slvCrossover    = 0
slvMethod       = 2
allLogToConsole = 1

# additional constraints and their parameters
MinIsland         = False
Binarity          = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1
LSRobustness      = False

# Pixel centering of the pupils and image for optimization
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = False
ImPart      = True

#Sampling
nPup = 290
nFPM = 50
Fmax2d = 32
nImg2d = 64

#Optical System
# spectral bandwidth
bw   = 0
nlam = 1

# mask radius in lam0/D units
rMask = 8.543/2

# LS inner and outer diameter
LS_OD = 807

#Optimization

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  4.00 #3.75
rho1 = 10.00 #15.0

# contrast in the dark region
cDarkHole = 8.0

# tau (integrated Pupil transmission)
tau   = 0.4

# maximum pixel shift along a given axis for Lyot stop 
pix_max   = 0

#Export fits files
do_fits = True

print('parameters loaded')

# File reading for pupil and lyot stop

fdir = 'input_files/'
if pupil_name == 'lvr':
    fname_pup = 'apertures/TelAp_full_luvoir2017novAp05ss100cobs1gap{0}_N{1:04d}.fits'.format(gap,nPup)
    fname_lys = 'lyot_stops/luvoir_LS_ann{0:02d}D{1:02d}_clear_N{2:04d}.fits'.format(int(round(100*inD)), int(round(100*outD)), nPup)
elif pupil_name == 'HiCAT':
    fname_pup = 'apertures/HiCAT-Aper_F-N0{0}_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004.fits'.format(nPup,)
    fname_lys = 'lyot_stops/HiCAT-Lyot_F-N0{0}_LS-Ann-gy-ID0345-OD0{1}-SpX0036.fits'.format(nPup,LS_OD,)
else:
    fname_pup = 'apertures/pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    fname_lys = 'lyot_stops/pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)

# path for the Pupil and Lyot stop files
fpath_pup = fdir + fname_pup
fpath_lys = fdir + fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)

print('aperture and lyot stop loaded')

# List of Lyot stops for the design optimization
LyotStop2d_t = [LyotStop2d]

# List construction for Lyot stop position shifts
pix_t = []
if pix_max >= 1:
    pix_pos_t = 1+np.arange(pix_max)
    pix_neg_t = - pix_pos_t
    pix_t = list(-pix_pos_t) + list(pix_pos_t)
    pix_t.sort()

# List construstion for the Lyot stops 
for j in range(2):
    for i in range(len(pix_t)):
        LyotStop2d_t.append(np.roll(LyotStop2d, pix_t[i], axis=j))

# number of coronagraph configuration
ncorono      = len(LyotStop2d_t)
print('# of coronagraph configurations: {0}'.format(ncorono))

#print(LyotStop2d_t[1]-LyotStop2d_t[0])
#fits.writeto('LS_test.fits', LyotStop2d_t[1]-LyotStop2d_t[0], overwrite=True)


# default solver 
if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

# list parameters for the coronagraph and the optimization problem
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
                 allLogToConsole = allLogToConsole,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg,
                 ImPart = ImPart, LSRobustness = LSRobustness)

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
# generation of a coronagraph object
if corono_name == 'SP':
    corono_t.append(coro.design.SP2d(**params))
elif corono_name == 'APLC':
    for k in range(ncorono):
        corono_t.append(coro.design.APLC2d(**params_t[k])) 
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
# generation of a optimzation problem object
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_2d.MaxTau(corono=corono_t, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono_t, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
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
        Apod1_2dtmp =  Apod1_2d[corono_t[0].nPup//2:, corono_t[0].nPup//2:]
        Apod1_2d[:corono_t[0].nPup//2, corono_t[0].nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
        Apod1_2d[:, :corono_t[0].nPup//2]          = np.flip(Apod1_2d[:, corono_t[0].nPup//2:], axis=1)
        
#%%
"""
Save apodizer
"""
fdir = 'apodizers/{0}/'.format(pupil_name)
if not os.path.exists(fdir):
    os.makedirs(fdir)
    
fname = problem1.get_filename() + 'pix_max={0}.fits'.format(pix_max)
fpath = fdir + fname

if do_fits is True:
    fits.writeto(fpath, Apod1_2d, overwrite=True)
