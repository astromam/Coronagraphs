#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:29:12 2018

@author: mndiaye
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:20:57 2018

@author: mndiaye
"""

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
import corono as coro

#%% parameters
"""
Parameters
"""
pl.close('all')

corono_name  = 'APLC' # 'APLC' or 'SP'
problem_name = 'MaxContrastLinf' # 'MaxContrastL1' #,'MaxContrastLinf' # 'MaxTau' #
solver       = 'stdgrb' # 'stdgrb', 'gurobipy', 'scipy.linprog'

FirstDer    = False
SecondDer   = False
MinIsland   = True
Binarity    = False
FirstDerLim = 0.01
SecondDerLim= 0.001 
FirstDerGlobalLim = 1.
BinarityReg       = 10.

nPup = 500
nFPM = 50
nImg = 44
Fmax = 11
R    = 1

bw   = 0.1
nlam = 5

PupilID    = 0.20
rMask       = 4.0

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75

LyotStopID = 0.40
LyotStopOD = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 3.5
rho1 = 10.0

# contrast in the dark region
cDarkHole = 10.0

# tau (integrated Pupil transmission)
tau   = 0.4

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilID)*1.0
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, problem_name = problem_name,
                 corono_name = corono_name,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg)

nlambis = 11
nImgbis = 110
Fmaxbis = 11    


    
#%%
fdir = Path('../../results/1D/').resolve()

fdir_pyth = fdir / 'dat_pyth'
fdir_npy  = fdir / 'npy'
fdir_plot = fdir / 'plots'

if not os.path.exists(fdir_plot):
    os.makedirs(fdir_plot)
    
if not os.path.exists(fdir_pyth):
    os.makedirs(fdir_pyth)    
    
#%%  

nFirstDerGlobalLim  = 201
stepFirstDerGlobalLim = 0.1
FirstDerGlobalLim_t = stepFirstDerGlobalLim*np.arange(nFirstDerGlobalLim)

params = coro.update_params(params, FirstDerGlobalLim = FirstDerGlobalLim_t[-1])

""" 
Coronagraph defintion
"""
if corono_name == 'APLC':
    corono0 = coro.design.APLC1d(**params)
elif corono_name == 'SP':
    corono0 = coro.design.SP1d(**params)
elif corono_name == 'HDZPM':
    corono0 = coro.design.HDZPM1d(**params)
elif corono_name == 'HTZPM':
    corono0 = coro.design.HTZPM1d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_1d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
     raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))



#%%

fname = problem1.get_filename(nlam = nlambis) + '_apod_t.npy'
fpath = fdir_npy / fname
Apod_dic = np.load(fpath).item()

fname = problem1.get_filename(nlam = nlambis) + '_poly_direct_image_t.npy'
fpath = fdir_npy / fname
poly_direct_image_dic = np.load(fpath).item()

fname = problem1.get_filename(nlam = nlambis) + '_poly_corono_image_t.npy'
fpath = fdir_npy / fname
poly_corono_image_dic = np.load(fpath).item()


#%%
FirstDerGlobalLim_t = []
Apod_t = []
poly_corono_image_t = []
for key in Apod_dic:
    FirstDerGlobalLim_t.append(float(key))
    Apod_t.append(Apod_dic[key])
    poly_corono_image_t.append(poly_corono_image_dic[key])

FirstDerGlobalLim_t = np.asarray(FirstDerGlobalLim_t)
Apod_t = np.asarray(Apod_t)
poly_corono_image_t = np.asarray(poly_corono_image_t)


#%%
fname = problem1.get_filename(nlam = nlambis) + '_apod_vs_FirstDerGlobalLim.pdf'
fpath = fdir_plot / fname

yticks_ind = list(20*np.arange(1+nFirstDerGlobalLim//20))

aa = np.arange(nFirstDerGlobalLim)

# imshow for different apodizers
pl.figure(3,(16,4.5))
pl.clf()
pl.subplot(121)
pl.imshow(Apod_t,aspect='auto', vmin = 0., vmax = 1.,
          extent=[0,R/2,-0.5,nFirstDerGlobalLim-.5],
          origin='lower', cmap = 'inferno')
pl.yticks(aa[yticks_ind],FirstDerGlobalLim_t[yticks_ind])
pl.xlabel(r'Pupil radius r')
pl.ylabel(r'Global 1st derivative limit $\beta$')
pl.title('Transmission profiles for different apodizers')
cbar = pl.colorbar()
cbar.set_label('Normalized amplitude')
    
pl.subplot(122)
pl.imshow(np.log10(poly_corono_image_t),aspect='auto', vmin = -11., vmax = -6.,
          extent=[np.min(corono0.xi),np.max(corono0.xi),-0.5,nFirstDerGlobalLim-.5],
          origin='lower',cmap='inferno')
pl.yticks(aa[yticks_ind],FirstDerGlobalLim_t[yticks_ind])
pl.title('Broadband intensity profiles for different apodizers')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')

pl.tight_layout()
pl.savefig(str(fpath))

   
pl.show()