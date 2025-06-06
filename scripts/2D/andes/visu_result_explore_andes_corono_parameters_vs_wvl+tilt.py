#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@author: mndiaye, asimmonnin, asp
"""

# visualize coronagraph parameters ranges ve throughput, integrated 
# energy of the star and planet psf in the field, with or without coro, at 
# some wavelengths in R;I.Z,Y,J and H bands for a given spaxel size

#%%
"""
### Initialization
"""
import numpy as np
from astropy.io import fits
# import slow_fourier_transform as sft
from uniform_disk import uniform_disk
# from psf_profile import profile
# from draw_vanes import six_arms

import os
from pathlib import Path
# import matplotlib.pyplot as plt

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

#%%

was_donow = '20250604114227'

# 20250604111333 7
# 20250604114227 10
# 20250604155623 16

#%%
"""
### Working directory
"""
user = 'Alain'
if user == 'Alain':
    fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()  # opd's seed value
    fdir_res = Path('D:/Andes/Data_corono/results/').resolve()  #  fits files
    fdir_res = (fdir_res / was_donow)
    fdir_plot = Path('D:/Andes/Data_corono/results/').resolve()


#%%
"""
### Parameters
"""

file_lst = os.listdir(fdir_res)
file_res = [x for x in file_lst if 'results_vs_params_' in x]
file_int = [x for x in file_lst if 'intensities_vs_params_' in x]
file_thr = [x for x in file_lst if 'Parameters_throughput_' in x]

hdr = fits.getheader(fdir_res/file_res[0])

nPup = hdr['NPUP']
nFPM = hdr['NFPM']
nImg = hdr['NIMG']
lamC = hdr['LMBD']
D = hdr['DIAM']
pscale = hdr['PSCL']

lam_min = hdr['LMIN']
lam_itv = hdr['LITV']
lam_stp = hdr['LSTP']
lam_lst = np.arange(lam_min,lam_min+lam_itv*lam_stp+1e-9,lam_stp)
# lam_lst = np.array([1.55e-06, 1.60e-06, 1.65e-06]) # debug
nL = len(lam_lst)
print("# of wvls: ",nL, ", wvl min: ", lam_min, ", wvl step: ", lam_stp,
      "\n", "wvls: ", lam_lst)

mB_min = hdr['SFPM_MIN']
mB_itv = hdr['SFPM_ITV']
mB_stp = hdr['SFPM_STP']
mB_lst = np.arange(mB_min,mB_min+mB_stp*(mB_itv+0.5),mB_stp)

obs_min = hdr['OBST_MIN']
obs_itv = hdr['OBST_ITV']
obs_stp = hdr['OBST_STP']
obs_lst = np.arange(obs_min,obs_min+obs_stp*(obs_itv+0.5),obs_stp)

dL_min = hdr['DLYO_MIN']
dL_itv = hdr['DLYO_ITV']
dL_stp = hdr['DLYO_STP']
dL_lst = np.arange(dL_min,dL_min+dL_stp*(dL_itv+.5),dL_stp)

ap_mas = hdr['D_SPAXEL']
ap_roi = hdr['SPXL_ROI']
ap_min = hdr['SPXL_MIN']
ap_itv = hdr['SPXl_ITV']
ap_stp = hdr['SPXL_STP']
ap_lst = np.arange(ap_min,ap_min+ap_stp*ap_itv+1,ap_stp)
nA = len(ap_lst)
ap_area = np.pi * (ap_mas/2.)**2.

# elt_pup_fnm = hdr['EPUP_FNM']
# fdir_pupil = fdir_dat / 'Pupil'
# fpath_elt = fdir_pupil / elt_pup_fnm
# Pupil = fits.getdata(fpath_elt,)

sim_data = fits.getdata(fdir_res/file_res[0])
thr_data = fits.getdata(fdir_res/file_thr[0])
int_data = fits.getdata(fdir_res/file_int[0])

# Coronagraphic focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)

# field of view in mas
fov_mas = nImg * pscale
# fov in radians
fov_rdn = fov_mas * rad2mas

# stackoveflow...
a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
b_ = a_.copy()
aa, bb = np.meshgrid(a_, b_)
rad_mas = np.zeros((nImg,nImg,len(ap_lst)+1))
rad_mas[:,:,0] = np.sqrt(aa**2 + bb**2) * pscale
for a in range(len(ap_lst)):
    rad_mas[:,:,a+1] = np.sqrt((aa * pscale)**2 + (bb * pscale - ap_lst[a])**2)

aee = {a:np.where(rad_mas[:,:,a+1]<=ap_mas/2.) for a in range(len(ap_lst))}
rng = {a:np.where(
    np.abs(rad_mas[:,:,0]-ap_lst[a]*1.)<=ap_mas/2) for a in range(len(ap_lst))}

eta_data = sim_data.copy()
for i in range(len(ap_lst)):
    eta_data[0,i,:] = (eta_data[0,i,:] * float(len(aee[i][0])) /
                       float(len(rng[i][0])))

contrast = eta_data[0,:] / eta_data[1,:]

