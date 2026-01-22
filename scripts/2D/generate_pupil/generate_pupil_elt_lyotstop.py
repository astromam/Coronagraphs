#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 11:21:08 2025

@author: mndiaye
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 22:22:37 2025

@author: mndiaye
"""

"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

from corono.utils import uniform_disk

#%%
"""
### Parameters
"""
do_sav = False
nPup = 400
OD = 0.89
ID = 0.35

#%%
"""
### File reading for pupil
"""
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/elt-andes/').resolve()
fname_ini = 'ELT_pupil_400.fits'
fpath_ini = fdir / fname_ini

"""
### File saving for symmetric pupil
"""
fname_lys = 'ELT_pupil_400_lyotstop.fits'
fpath_lys = fdir / fname_lys

#%%
"""
### Read file
"""

pupil = fits.getdata(fpath_ini)


#%%
"""
### Generate symmetric pupil
"""
LyotStop = uniform_disk(nPup, OD*nPup//2, CtrBtwnPix=True) - uniform_disk(nPup, ID*nPup//2, CtrBtwnPix=True) 

#%%
"""
### Save symmetric pupil
"""
if do_sav:
    fits.writeto(fpath_lys, LyotStop, overwrite=True)


#%%
"""
### Display pupil
"""
plt.figure(0, (8, 4.5))
plt.clf()
plt.subplot(121)
plt.imshow(pupil)
plt.title('ELT')
plt.subplot(122)
plt.imshow(LyotStop)
plt.title('ELT sym')

#%%
"""
### test on symmetric pupil
"""
plt.figure(1, (8, 4.5))
plt.clf()
plt.subplot(111)
plt.imshow(pupil-LyotStop)
plt.title('Lyot stop')
# plt.subplot(122)
# plt.imshow(pupilsym-np.flipud(pupilsym))
# plt.title('test ud')
