# -*- coding: utf-8 -*-
"""
Created on Mon may  9 2025

@author: asp
"""

# plot contrats at given angular separation for two contrast profile
# fits files produced by multi_wvl_psf_profile_cut.py

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!


#%%
"""
### scaling
"""
# lam_c = 1600e-9  #  "central" lambda (of interest) in meters
# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

as_oi = 25
as_str = str(int(as_oi))

#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots

res_dir = fdir_res
fdir_plt = fdir_plt

#%%

file = ("20250521172457/perfect/contrast_profile_L0toD_perfect_wo_noise_psf_elt_pupil.fits",)
# YJH 20250521135258
# RIZ 20250521172457
#%%

head_psf = fits.getheader(res_dir / file[0])

lamC = head_psf['LMBD']
lam_min = head_psf['LMIN']
lam_stp = head_psf['LSTP']
lam_itv = head_psf['LITV']
lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)
lam_lst = lam_lst[np.where(lam_lst < 1900e-9)]
nL = len(lam_lst)

nImg = head_psf['NIMG']
D = head_psf['DIAM']
pscale = head_psf['PSCL']
hlf_fov = nImg * pscale / 2.

aS = np.arange(nImg//2)*(mas2rad * hlf_fov / (D *1e9) ) # ang. sep.
pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))

elt_data =  (fits.getdata(res_dir / file[0]))[0:nL,:]
contrast = elt_data[:,pos_as_oi]

            
#%%
"""
plot profiles
"""
# plot of the azimutal average ratio profile
plt.figure(2, (8, 4.5))
plt.tight_layout()
plt.xlabel(r'Wavelength $\lambda$ [nm]')#[$\lambda$/D]')
plt.ylabel(f'Contrast @ {int(as_oi)} mas')
plt.yscale('log')
# plt.title('Coronagraph configuration for YJH band')
# plt.title('ELT constat @ '+as_str+' mas')
plt.grid(True)

plt.plot(lam_lst*1e9, contrast)

plt.ylim(1e-5,1e-1)

fname = ("contrast_elt_"+as_str+'mas_YJH_')
fpath_contrast_25mas_pdf = fdir_plt / file[0].split('/')[0] / (fname + '.pdf')
plt.savefig(fpath_contrast_25mas_pdf, bbox_inches='tight', pad_inches=0.1)


plt.show()

