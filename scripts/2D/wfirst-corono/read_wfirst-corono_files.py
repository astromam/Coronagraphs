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

lam0 = 575e-9
lam1 = 730e-9 

pix_sz = 8.95

nBegC = -nImg//2*pix_sz
nEndC = nImg//2*pix_sz

nPol = 2
str_pol = ['pols', 'polp']

nm = 1e9

#%%
"""
### Directories
"""
fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/Proposals/CNES/2019/WFIRST corono/phaseb_data/hlc_20190210/').resolve()


fname = 'run461_occ_lam5.75e-07theta6.69'

fpath_re_arr = np.empty(nPol, dtype=object)
fpath_im_arr = np.empty(nPol, dtype=object)

for i in range(nPol):
    fname_re = fname + str_pol[i] + '_real.fits'
    fname_im = fname + str_pol[i] + '_imag.fits'

    fpath_re_arr[i] = fdir / fname_re
    fpath_im_arr[i] = fdir / fname_im

#%%
"""
### File
"""
fdir_res = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/Proposals/CNES/2019/WFIRST corono/plots/hlc_20190210/')
if not os.path.exists(fdir_res):
    os.makedirs(fdir_res)
    
fname_re_im = fname + '_real_imag.png'
fname_mod_phi = fname + '_mod_phi.png'
fname_pol_diff = fname + '_pol_diff.png'

fpath_re_im   = fdir_res / fname_re_im
fpath_mod_phi = fdir_res / fname_mod_phi
fpath_pol_diff= fdir_res / fname_pol_diff

#%%
"""
### File reading
"""

# s polarization
mask_s_re0 = fits.getdata(fpath_re_arr[0])
mask_s_im0 = fits.getdata(fpath_im_arr[0])
mask_s_re = mask_s_re0[nBeg:nEnd,nBeg:nEnd]
mask_s_im = mask_s_im0[nBeg:nEnd,nBeg:nEnd]

# p polarization
mask_p_re0 = fits.getdata(fpath_re_arr[1])
mask_p_im0 = fits.getdata(fpath_im_arr[1])
mask_p_re = mask_p_re0[nBeg:nEnd,nBeg:nEnd]
mask_p_im = mask_p_im0[nBeg:nEnd,nBeg:nEnd]

#%%
"""
### Phase and modulus of the mask
"""
# s polarization
mask_s_cplx = mask_s_re0 + 1j*mask_s_im0
mask_s_mod0 = np.abs(mask_s_cplx)
mask_s_phi0 = np.angle(mask_s_cplx)
mask_s_OPD0_lam0 = mask_s_phi0*lam0/(2*np.pi)
mask_s_OPD0_lam1 = mask_s_phi0*lam1/(2*np.pi)

mask_s_mod = mask_s_mod0[nBeg:nEnd,nBeg:nEnd]
mask_s_phi = mask_s_phi0[nBeg:nEnd,nBeg:nEnd]
mask_s_OPD_lam0 = mask_s_OPD0_lam0[nBeg:nEnd,nBeg:nEnd]
mask_s_OPD_lam1 = mask_s_OPD0_lam1[nBeg:nEnd,nBeg:nEnd]

# p polarization
mask_p_cplx = mask_p_re0 + 1j*mask_p_im0
mask_p_mod0 = np.abs(mask_p_cplx)
mask_p_phi0 = np.angle(mask_p_cplx)
mask_p_OPD0_lam0 = mask_p_phi0*lam0/(2*np.pi)
mask_p_OPD0_lam1 = mask_p_phi0*lam1/(2*np.pi)

mask_p_mod = mask_p_mod0[nBeg:nEnd,nBeg:nEnd]
mask_p_phi = mask_p_phi0[nBeg:nEnd,nBeg:nEnd]
mask_p_OPD_lam0 = mask_p_OPD0_lam0[nBeg:nEnd,nBeg:nEnd]
mask_p_OPD_lam1 = mask_p_OPD0_lam1[nBeg:nEnd,nBeg:nEnd]

#%%
# conversion in nm
OPD_s_lam1_nm = mask_s_OPD_lam1*nm
OPD_p_lam1_nm = mask_p_OPD_lam1*nm

# OPD difference between two polarization states
diff_OPD_ps_nm = OPD_p_lam1_nm - OPD_s_lam1_nm

#%%
disk = coro.uniform_disk(nImg, 37/2)


#%%
"""
### Image display
"""
vmin_re = -0.2
vmax_re = 1

vmin_im = -0.02
vmax_im = 0.02


pl.figure(1, figsize=(8, 4.5))
pl.clf()

pl.subplot(221)
pl.imshow(mask_s_re, vmin=vmin_re, vmax=vmax_re, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_re = pl.colorbar()
cbar_re.set_label('real part')
pl.title(r'HLC real part, s pol, $\lambda$={0:.1f}nm'.format(lam0*1e9))

pl.subplot(222)
pl.imshow(mask_s_im, vmin=vmin_im, vmax=vmax_im, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_im = pl.colorbar()
cbar_im.set_label('imag part')
pl.title('HLC imag part, s pol, $\lambda$={0:.1f}nm'.format(lam0*1e9))

pl.subplot(223)
pl.imshow(mask_p_re, vmin=vmin_re, vmax=vmax_re, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_re = pl.colorbar()
cbar_re.set_label('real part')
pl.title(r'HLC real part, p pol, $\lambda$={0:.1f}nm'.format(lam0*1e9))

pl.subplot(224)
pl.imshow(mask_p_im, vmin=vmin_im, vmax=vmax_im, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_im = pl.colorbar()
cbar_im.set_label('imag part')
pl.title('HLC imag part, p pol, $\lambda$={0:.1f}nm'.format(lam0*1e9))

pl.tight_layout()
pl.savefig(fpath_re_im)
pl.show()

#%%
vmin_mod = 0
vmax_mod = 1

vmin_phi = -np.pi
vmax_phi = np.pi

vmin_OPD = -400
vmax_OPD = 400


pl.figure(2, figsize=(8, 4.5))
pl.clf()

pl.subplot(221)
pl.imshow(mask_s_mod, vmin=vmin_mod, vmax=vmax_mod, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_mod = pl.colorbar()
cbar_mod.set_label('Modulus')
pl.title(r'Modulus'.format(lam1*1e9))
pl.text(-600,0.,"s polarization", fontsize=13,
                           verticalalignment='center', rotation=90)

pl.subplot(222)
pl.imshow(OPD_s_lam1_nm, vmin=vmin_OPD, vmax=vmax_OPD, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_phi = pl.colorbar()
cbar_phi.set_label('nm')
pl.title(r'OPD at $\lambda$={0:.1f}nm'.format(lam1*1e9))

pl.subplot(223)
pl.imshow(mask_p_mod, vmin=vmin_mod, vmax=vmax_mod, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_mod = pl.colorbar()
cbar_mod.set_label('Modulus')
#pl.title(r'HLC modulus, $\lambda$={0:.1f}nm'.format(lam1*1e9))
pl.text(-600,0.,"p polarization", fontsize=13,
                           verticalalignment='center', rotation=90)

pl.subplot(224)
pl.imshow(OPD_p_lam1_nm, vmin=vmin_OPD, vmax=vmax_OPD, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
cbar_phi = pl.colorbar()
cbar_phi.set_label('nm')
#pl.title(r'HLC OPD, p pol, $\lambda$={0:.1f}nm'.format(lam1*1e9))

pl.tight_layout()
pl.savefig(fpath_mod_phi)
pl.show()

#%%
"""
### Histogram of the OPD values 
"""

pl.figure(3)
pl.clf()
pl.hist(OPD_s_lam1_nm.flatten(), bins=100, color='steelblue')
pl.title(r'OPD in s polarisation at $\lambda={0}$nm'.format(lam1*nm))

pl.figure(4)
pl.clf()
pl.hist(OPD_p_lam1_nm.flatten(), bins=100, color='steelblue')
pl.title(r'OPD in p polarisation at $\lambda={0}$nm'.format(lam1*nm))
pl.show()

#%%
"""
### Difference between the s and p polarization
"""
vmin_diff = 0
vmax_diff = 0.8

pl.figure(5)
pl.clf()
pl.imshow(diff_OPD_ps_nm, vmin=vmin_diff, vmax=vmax_diff, extent=[nBegC,nEndC,nBegC,nEndC])
pl.xlabel(r"X in $\mu$m")
pl.ylabel(r"Y in $\mu$m")
pl.title('OPD difference between p and s polarization states')
cbar_phi = pl.colorbar()
cbar_phi.set_label('nm')
pl.tight_layout()
pl.savefig(fpath_pol_diff)
pl.show()

#%%
"""
### test on disk
"""
pl.figure(6)
pl.clf()
pl.imshow(disk)
pl.title('generation of a mask circle')
pl.show()

# Standard deviation of the OPD difference within the mask
print(np.std(diff_OPD_ps_nm[disk == 1]))
