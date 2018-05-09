#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:52:23 2018

@author: mndiaye
"""

import numpy as np
import time
from pathlib import Path

from corono.utils import to_dict

from astropy.io import fits
import pylab as pl

#%% parameters
"""
Parameters
"""
# Telescope name
pupil_name = 'sbr' # 'vlt' or 'sbr' or 'lvr'

#nPup = corono0.params['nPup']
nPup = 100

# mask radius in lam0/D unit
#rMask = 4.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  4.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 4.0

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
corono_name   = 'SP' # 'SP' or 'APLC'
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True

#nlam
nlam=5

do_fits = True

#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = Path('./pupils/2D/').resolve()
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


params = to_dict(nPup=nPup, rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2=CtrBtwnPix2, nlam=nlam, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym)

#%%
pl.figure(1)
pl.imshow(Pupil2d)

pl.show()

#%%
fdir_ampl = Path('/Users/mndiaye/Dropbox/central storage/AMPL/PupilDataFiles/2D/Subaru/dat')
fname = 'SubaruPupil_N=0150_center_quarter.dat'
fpath = fdir_ampl / fname

Pupil_ampl = np.loadtxt(fpath)

pl.figure(2)
pl.imshow(Pupil_ampl)

#%%
Pupil2d_quarter = Pupil2d[nPup//2:,nPup//2:]

pl.figure(3)
pl.imshow(Pupil2d_quarter)

fname = 'SubaruPupil_N={0:04d}_center_quarter.dat'.format(nPup//2)
fpath = fdir_ampl / fname

np.savetxt(fpath, Pupil2d_quarter)

