#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:53:10 2020

@author: mndiaye
"""

#%%
"""
*** Initialization
"""
import numpy as np
import time
import os

from pathlib import Path
import corono as coro

import pylab as pl
import pandas as pd

#%% parameters
"""
*** Parameters
"""
# coronagraph type
corono_name  = 'APLC' # 'APLC' or 'SP'

# sampling
nPup = 300
nFPM = 35
nImg = 256
Fmax = 50
R    = 1

# spectral sampling
bw   = 0.2
nlam = 11

# coronagraph configuration
PupilID    = 0.14
rMask      = 2.8
LyotStopID = 0.28
LyotStopOD = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 5.
rho1 = 40.0

# contrast in the dark region
cDarkHole = 8.0

# pupil definitions
r   = np.arange(nPup)*R/nPup + R/(2*nPup)
xi  = (Fmax/nImg)*np.arange(nImg+1)

Pupil1d      = (r>PupilID)*1.0
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

# dictionary parameters
params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 corono_name = corono_name)

#%%
"""
*** Working directory
"""
fdir = Path('../../results/1D/dat_pyth').resolve()
if not os.path.exists(fdir):
    os.makedirs(fdir)      

#%%
"""
*** GPI File directory
"""
# file directory for GPI coronagraph
fdir_gpi = Path('/Users/mndiaye/OneDrive - Université Nice Sophia \
Antipolis/Work/Code/Routines/apodizer linear programming/APLC \
paper/GPI/1D/').resolve()

# generic name for GPI coronagraph
fgen = 'obs=14_m=2800_bw=20_rI=3400_rO=20_C=08_b=005'

# filename for the GPI parts
fname_pup = fgen + '_pupIII.csv'
fname_apo = fgen + '_apoIII.csv'
fname_stp = fgen + '_stpIII.csv'
fname_psfmono = fgen + '_psfmonoIII.csv'
fname_psfpoly = fgen + '_psfpolyIII.csv'
fname_cormono = fgen + '_cormonoIII.csv'
fname_corpoly = fgen + '_corpolyIII.csv'

# filepath for the GPI parts
fpath_pup  = fdir_gpi / fname_pup
fpath_apo  = fdir_gpi / fname_apo
fpath_stp  = fdir_gpi / fname_stp
fpath_psfmono  = fdir_gpi / fname_psfmono
fpath_psfpoly  = fdir_gpi / fname_psfpoly
fpath_cormono  = fdir_gpi / fname_cormono
fpath_corpoly  = fdir_gpi / fname_corpoly

#%%
"""
*** Import GPI files and convert them into numpy array
"""
Pupil1d = pd.read_csv(fpath_pup, header = None).to_numpy().flatten()
Apod1d = pd.read_csv(fpath_apo, header = None).to_numpy().flatten()
LyotStop1d = pd.read_csv(fpath_stp, header = None).to_numpy().flatten()
Psf1dmono = pd.read_csv(fpath_psfmono, header = None).to_numpy().flatten()
Psf1dpoly = pd.read_csv(fpath_psfpoly, header = None).to_numpy().flatten()
Cor1dmono = pd.read_csv(fpath_cormono, header = None).to_numpy().flatten()
Cor1dpoly = pd.read_csv(fpath_corpoly, header = None).to_numpy().flatten()

#%%  
""" 
*** Coronagraph definition
"""
params0    = coro.update_params(params, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d) 

if corono_name == 'APLC':
    corono0 = coro.design.APLC1d(**params0)
elif corono_name == 'SP':
    corono0 = coro.design.SP1d(**params0)
elif corono_name == 'HDZPM':
    corono0 = coro.design.HDZPM1d(**params0)
elif corono_name == 'HTZPM':
    corono0 = coro.design.HTZPM1d(**params0)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
*** Computation of the direct and coronagraphic images
"""
# monochromatic images
mono_direct_image0 = corono0.compute_direct_intensity_1d(Apod1d, poly=False)
mono_corono_image0 = corono0.compute_corono_intensity_1d(Apod1d, poly=False)

mono_peak = mono_direct_image0[nlam//2].max()
mono_direct_image0 /= mono_peak
mono_corono_image0 /= mono_peak

# broadband images
poly_direct_image0 = corono0.compute_direct_intensity_1d(Apod1d)
poly_corono_image0 = corono0.compute_corono_intensity_1d(Apod1d)

poly_peak = poly_direct_image0.max()
poly_direct_image0 /= poly_peak
poly_corono_image0 /= poly_peak

#%%
"""
*** Plot display
"""
#%% check pupil parts
pl.figure(0, (8, 4.5))
pl.clf()
pl.plot(Pupil1d)
pl.plot(Apod1d)
pl.plot(LyotStop1d)
pl.show()

#%%
# Plot monochromatic psfs
# pl.figure(1, (8, 4.5))
# pl.clf()
# pl.semilogy(xi, Psf1dmono, label='Math')
# pl.semilogy(xi, mono_direct_image0[nlam//2], label='Python', ls='--')
# pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
# pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
# pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
# pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
# pl.xlabel(r'Angular separation in $\lambda_0$/D')
# pl.ylabel('Normalized intensity in log scale')
# pl.xlim(-0.5, 50.5)
# pl.ylim(10**(-8.2), 10**(1.8))
# pl.legend()
# pl.tight_layout()
# pl.show()

#%%
# Plot monochromatic coronagraphic image
pl.figure(2, (8, 4.5))
pl.clf()
pl.semilogy(xi, Cor1dmono, label='Math')
pl.semilogy(xi, mono_corono_image0[nlam//2], label='Python')
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.5, 50.5)
pl.ylim(10**(-12.2), 10**(-3.8))
pl.legend()
pl.tight_layout()
pl.show()

#%%
# Plot broadband psfs
# pl.figure(3, (8, 4.5))
# pl.clf()
# pl.semilogy(xi, Psf1dpoly, label='Math')
# pl.semilogy(xi, poly_direct_image0, label='Python', ls='--')
# pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
# pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
# pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
# pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
# pl.xlabel(r'Angular separation in $\lambda_0$/D')
# pl.ylabel('Normalized intensity in log scale')
# pl.xlim(-0.5, 50.5)
# pl.ylim(10**(-8.2), 10**(1.8))
# pl.legend()
# pl.tight_layout()
# pl.show()

#%%
# Plot broadband coronagraphic images
pl.figure(4, (8, 4.5))
pl.clf()
pl.semilogy(xi, Cor1dpoly, label='Math')
pl.semilogy(xi, poly_corono_image0, label='Python')
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.5, 50.5)
pl.ylim(10**(-12.2), 10**(-3.8))
pl.legend()
pl.tight_layout()
pl.show()
