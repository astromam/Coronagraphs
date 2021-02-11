import numpy as np
import time
import os
from pathlib import Path
from astropy.io import fits

import corono as coro

pupil_name   = 'HiCAT'

problem_name = 'MaxTau' # 'MaxContrastL1' # 'MaxTau' # ,'MaxContrastLinf' # #  
solver       = 'stdgrb' #,'stdgrb' #  'gurobipy', 'scipy.linprog'
slvLogToConsole = 1
slvCrossover    = 0
slvMethod       = 2
allLogToConsole = 0

MinIsland   = False
Binarity    = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1

nPup = 96
nFPM = 50
Fmax2d = 16
nImg2d = 32

rMask = 8.543/2

rho0 =  3.75
rho1 = 15.0

cDarkHole = 8.0

tau   = 0.4

corono_name = 'APLC' # 'SP' or 'APLC'
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = False
ImPart      = True

bw   = 0.1
nlam = 5

pix_max   = 1

do_fits = True

fdir = Path('./pupils/2D/').resolve()
if pupil_name == 'lvr':
    fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
    fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
elif pupil_name == 'HiCAT':
    fname_pup = 'HiCAT-Aper_F-N00{0}_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004.fits'.format(nPup,)
    fname_lys = 'HiCAT-Lyot_F-N00{0}_LS-Ann-gy-ID0345-OD0807-SpX0036.fits'.format(nPup,)
else:
    fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    
fpath_pup = fdir / fname_pup
fpath_lys = fdir / fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)

#circ = coro.utils.uniform_disk(nPup, 0.95*nPup/2, CtrBtwnPix=True)
#LyotStop2d *= circ

LyotStop2d_t = [LyotStop2d]

pix_t = []
if pix_max >= 1:
    pix_pos_t = 1+np.arange(pix_max)
    pix_neg_t = - pix_pos_t
    pix_t = list(-pix_pos_t) + list(pix_pos_t)
    pix_t.sort()
    
for j in range(2):
    for i in range(len(pix_t)):
        LyotStop2d_t.append(np.roll(LyotStop2d, pix_t[i], axis=j))

ncorono      = len(LyotStop2d_t)
print('# of coronagraph configurations: {0}'.format(ncorono))

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
                 allLogToConsole = allLogToConsole,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg,
                 ImPart = ImPart)

params_t = []
for k in range(ncorono):
    params_t.append(coro.update_params(params, LyotStop2d=LyotStop2d_t[k])) 
    
corono_t = []

if corono_name == 'SP':
    corono_t.append(coro.design.SP2d(**params))
elif corono_name == 'APLC':
    for k in range(ncorono):
        corono_t.append(coro.design.APLC2d(**params_t[k])) 
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
    
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
    
t0 = time.time()
Apod1 = problem1.solve_model()
t1 = time.time()
print('optimization time             : {0:.2f}s'.format(t1-t0))

Apod1_2d = np.reshape(Apod1, (corono_t[0].nPup, corono_t[0].nPup))

if Pupil2dSym == True:
        Apod1_2dtmp =  Apod1_2d[corono_t[0].nPup//2:, corono_t[0].nPup//2:]
        Apod1_2d[:corono_t[0].nPup//2, corono_t[0].nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
        Apod1_2d[:, :corono_t[0].nPup//2]          = np.flip(Apod1_2d[:, corono_t[0].nPup//2:], axis=1)
        

fdir = Path('./results/2D/dat_pyth').resolve() / pupil_name
if not os.path.exists(fdir):
    os.makedirs(fdir)
    
fname = problem1.get_filename() + '_pix_max=1.fits'
fpath = fdir / fname

if do_fits is True:
    fits.writeto(fpath, Apod1_2d, overwrite=True)