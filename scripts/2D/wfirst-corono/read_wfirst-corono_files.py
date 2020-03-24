#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:13:56 2020

@author: mndiaye
"""

"""
### Initialization
"""

import numpy as np
import pylab as pl
import corono as coro
import os

from pathlib import Path
from astropy.io import fits

#%%
"""
### Parameters
"""
nImg0 = 2048
nImg = 50
nBeg = (nImg0-nImg)//2
nEnd = (nImg0+nImg)//2

lam0=575e-9

pix_sz = 8.95

nBegC = -nImg//2*pix_sz
nEndC = nImg//2*pix_sz

#%%
"""
### Directories
"""
fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/Proposals/CNES/2019/WFIRST corono/phaseb_data/hlc_20190210/').resolve()

fname = 'run461_occ_lam5.75e-07theta6.69polp'
fname_re = fname + '_real.fits'
fname_im = fname + '_imag.fits'

fpath_re = fdir / fname_re
fpath_im = fdir / fname_im

#%%
"""
### File
"""
fdir_res = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/Proposals/CNES/2019/WFIRST corono/plots/hlc_20190210/')
if not os.path.exists(fdir_res):
    os.makedirs(fdir_res)
    
fname_re_im = fname + '_real_imag.png'
fname_mod_phi = fname + '_mod_phi.png'

fpath_re_im   = fdir_res / fname_re_im
fpath_mod_phi = fdir_res / fname_mod_phi

#%%
"""
### File reading
"""
mask_re0 = fits.getdata(fpath_re)
mask_im0 = fits.getdata(fpath_im)

mask_re = mask_re0[nBeg:nEnd,nBeg:nEnd]
mask_im = mask_im0[nBeg:nEnd,nBeg:nEnd]

#%%
"""
### Phase and modulus of the mask
"""
mask_cplx = mask_re0 + 1j*mask_im0
mask_mod0 = np.abs(mask_cplx)
mask_phi0 = np.angle(mask_cplx)

mask_mod = mask_mod0[nBeg:nEnd,nBeg:nEnd]
mask_phi = mask_phi0[nBeg:nEnd,nBeg:nEnd]

#%%
"""
### Image display
"""
vmin_re = -0.2
vmax_re = 1

vmin_im = -0.02
vmax_im = 0.02


pl.figure(1, figsize=(12, 4.5))
pl.clf()

pl.subplot(121)
pl.imshow(mask_re, vmin=vmin_re, vmax=vmax_re, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_re = pl.colorbar()
cbar_re.set_label('real part')
pl.title(r'HLC real part, $\lambda$={0:.1f}nm'.format(lam0*1e9))

pl.subplot(122)
pl.imshow(mask_im, vmin=vmin_im, vmax=vmax_im, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_im = pl.colorbar()
cbar_im.set_label('imag part')
pl.title('HLC imag part, $\lambda$={0:.1f}nm'.format(lam0*1e9))

pl.tight_layout()
pl.savefig(fpath_re_im)
pl.show()

#%%
vmin_mod = 0
vmax_mod = 1

vmin_phi = -np.pi
vmax_phi = np.pi


pl.figure(2, figsize=(12, 4.5))
pl.clf()

pl.subplot(121)
pl.imshow(mask_mod, vmin=vmin_mod, vmax=vmax_mod, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_mod = pl.colorbar()
cbar_mod.set_label('Modulus')
pl.title(r'HLC modulus, $\lambda$={0:.1f}nm'.format(lam0*1e9))

pl.subplot(122)
pl.imshow(mask_phi, vmin=vmin_phi, vmax=vmax_phi, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_phi = pl.colorbar()
cbar_phi.set_label('Phase in rad')
pl.title(r'HLC phase, $\lambda$={0:.1f}nm'.format(lam0*1e9))

pl.tight_layout()
pl.savefig(fpath_mod_phi)
pl.show()

#%%