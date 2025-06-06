#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@author: mndiaye, asimmonnin, asp
"""

# explore coronagraph parameters ranges and compute throughput, integrated 
# energy of the star and planet psf in the field, with or without coro, at 
# all wavelengths in Y,J and H bands for a given spaxel size

#%%
"""
### Initialization
"""
import numpy as np
from astropy.io import fits
import slow_fourier_transform as sft
from uniform_disk import uniform_disk
from psf_profile import profile
# from draw_vanes import six_arms

import os
from pathlib import Path
from datetime import datetime  #  asp for datetime of now
# import matplotlib.pyplot as plt


#%%
"""
### Parameters
"""
# Pupil size
nPup = 400

# Sampling of the coronagraph focal plane mask
nFPM = 100

# Image size
nImg = 400

# wavelength in m
lamC = 1600e-9  #  some wvl unique value, reference wvl

# wvl = 'RIZYJHK'
# lam_min = 600e-9  #  min value in range
# lam_itv = 38  #
 
wvl = 'YJH'
lam_min = 950e-9  #  min value in range
lam_itv = 17  #  

lam_stp = 50e-9 # np.median(np.diff(lam_lst))
lam_lst = np.arange(lam_min,lam_min+lam_itv*lam_stp+1e-9,lam_stp)
# lam_lst = np.array([1.55e-06, 1.60e-06, 1.65e-06]) # debug
nL = len(lam_lst)
print("# of wvls: ",nL, ", wvl min: ", lam_min, ", wvl step: ", lam_stp,
      "\n", "wvls: ", lam_lst)

# spaxel diameter as an aperture and field locations
ap_mas = 16.  # spaxel of 7 mas (05/2025->) / 10. -> 03/2025
ap_roi = 20 # main angular separation of interest
ap_min = 15
ap_stp = 1
ap_itv = 20 # 10 for 'YJH'
ap_lst = np.arange(ap_min,ap_min+ap_stp*ap_itv+1,ap_stp)
nA = len(ap_lst)
ap_area = np.pi * (ap_mas/2.)**2.

# Pupil diameter in m 
D = 38.54

donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now
print(donow, wvl)

# Coronagraphic focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)

# YJH 20250604111333 7 / 20250604114227 10 / 20250604155623 16
#%%

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# plate scale in mas per pixel
pscale = 0.3

# field of view in mas
fov_mas = nImg * pscale
# fov in radians
fov_rdn = fov_mas * rad2mas
# in multiple of reference lambda (lamC) over D, 05/2025 -> 
mD_ref = fov_rdn / ( lamC / D )

"""
Pupil configuration
"""
### All Lyot coro parameters to test
dL_min = 0.85
dL_itv = 10
dL_stp = 0.01
dL_lst = np.arange(dL_min,dL_min+dL_stp*(dL_itv+.5),dL_stp)
obs_min = 0.30
obs_itv = 10
obs_stp = 0.01
obs_lst = np.arange(obs_min,obs_min+obs_stp*(obs_itv+0.5),obs_stp)
# mBmB_min = 3.0
mB_itv = 10
mB_min = 3.5
mB_stp = 0.1
mB_lst = np.arange(mB_min,mB_min+mB_stp*(mB_itv+0.5),mB_stp)

# 2D array pupil slope for tilt
sf_x = np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5
sf_y = np.transpose(sf_x.copy())

# stackoveflow...
a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
b_ = a_.copy()
aa, bb = np.meshgrid(a_, b_)
rad_mas = np.zeros((nImg,nImg,len(ap_lst)+1))
rad_mas[:,:,0] = np.sqrt(aa**2 + bb**2) * pscale
for a in range(len(ap_lst)):
    rad_mas[:,:,a+1] = np.sqrt((aa * pscale)**2 +
                               (bb * pscale - ap_lst[a])**2)

# aee = {a:np.where(
#     np.roll(rad_mas[:,:,0],int(np.rint(ap_lst[a]*1./pscale)),axis=0)<=ap_mas/2.)
#     for a in range(len(ap_lst))}
aee = {a:np.where(rad_mas[:,:,a+1]<=ap_mas/2.) for a in range(len(ap_lst))}
rng = {a:np.where(
    np.abs(rad_mas[:,:,0]-ap_lst[a]*1.)<=ap_mas/2.) for a in range(len(ap_lst))}

### results array: star/planet, #locations, #Ld, #obs, #fpm, #wvl
results = np.zeros(
    (2, nA, len(dL_lst), len(obs_lst), len(mB_lst), nL))
throughput = np.zeros((len(dL_lst),len(obs_lst)))
intensities = np.zeros((2, len(dL_lst), len(obs_lst), len(mB_lst),
                       nImg//2, nL))


#%%
"""
### Working directory
"""
user = 'Alain'
if user == 'Alain':
    fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()  # opd's seed value
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits files
    fdir_res = (fdir_res / donow)
    os.makedirs(fdir_res, exist_ok=True)
    
if user == 'Adrien':
    fdir_base = ("/Users/asimonnin/Desktop/PhD/Andes/Data_corono")
    fdir_dat = Path( fdir_base / 'data').resolve()
    fdir_res   = Path( fdir_base / 'results/').resolve()

elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    fdir_dat = Path(fdir_base / 'data/').resolve()
    fdir_res   = Path(fdir_base / 'results/').resolve()


#%%
# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'

# Filename and path for the ELT pupil
elt_pup_fnm = 'ELT_pupil_400.fits' # 2024
fpath_elt = fdir_pupil / elt_pup_fnm

# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)


#%%

### Loop over the parameters
for i in np.arange(nL):
    
    lam = lam_lst[i]

    # conversion lam/D to mas
    lamD2mas = (lam/D)*mas2rad
    
    # FoV in lam/D in the final image plane D
    mD = mD_ref * lamC / lam # 05/2025 ->
        
    print('lambda (nm):' ,np.round(lam*1e9,0), '. Field of view (lam/D):',
          np.round(mD,3))
    
    for Ld in range(len(dL_lst)): 
        diam = dL_lst[Ld]

        for ob in range(len(obs_lst)):
            obst=obs_lst[ob]

            LyotStop2d = (Pupil.copy() * # vanes *
                          (uniform_disk(nPup, diam*nPup/2) -
                           uniform_disk(nPup, obst*nPup/2)))
            
            for s in range(len(mB_lst)) :
                """
                ### Selection of the good apodizer configuration according to FPM
                """
                mB = mB_lst[s]

                # Field in the entrance pupil plane A
                Fld_AA0 = Pupil.copy()
    
                int_a = np.abs(Fld_AA0)**2
    
                ee_a = np.sum(int_a)
    
                # Field in the entrance pupil plane C
                Fld_CC0 = Fld_AA0 * 1.*LyotStop2d
    
                int_c = np.abs(Fld_CC0)**2
    
                ee_c = np.sum(int_c)
        
                if s == 0:

                    throughput[Ld,ob] = ee_c/ee_a
                    
                # Field in the image plane D (no coronagraph)*
                Fld_DD0 = sft.sft(Fld_CC0, nImg, mD*diam) #dL_lst
    
                # Intensity 
                Int_DD0 = np.abs(Fld_DD0)**2
    
                # Normalized intensity
                norm_peakDD0 = 1/np.max(Int_DD0)
    
        
                Int_DD0 *= norm_peakDD0 #norm_peakDD0
    
                """
                ### Calculation of corono image
                """
                    
                Fld_AA = Pupil.copy()
                # focal plane B 
                Fld_BB = mask2d*sft.sft(Fld_AA, nFPM, mB*lamC/lam)
    
                # pupil plane C before Lyot stop
                Fld_CC = Fld_AA - sft.isft(Fld_BB, nPup, mB*lamC/lam)
    
    
                # pupil plane C after Lyot stop
                Fld_LL = Fld_CC*LyotStop2d
    
                # image plane D 
                Fld_DD = sft.sft(Fld_LL, nImg, mD*diam) #dL_lst
    
                # Intensity
                Int_DD = np.abs(Fld_DD)**2
    
                # Normalized intensity
                Int_DD *= norm_peakDD0 #norm_peakDD0
                
                for a in range(len(ap_lst)):
                    
                    results[0,a,Ld,ob,s,i] = np.sum(Int_DD[rng[a]])
    
                # computation of the averaged intensity profiles of the images
                Int_DD_prf_mean,  px = profile(Int_DD, ptype='mean')
                Int_DD0_prf_mean, px = profile(Int_DD0,ptype='mean')
    
                ### Save images
                intensities[0,Ld,ob,s,:,i] = Int_DD0_prf_mean
                intensities[1,Ld,ob,s,:,i] = Int_DD_prf_mean
                
                for a in range(len(ap_lst)):
                    
                    Fld_AA = Pupil.copy() * np.exp(
                        1j * 2 * np.pi * ap_lst[a] * rad2mas * 
                        D * diam * sf_y / ( nPup * lam) )
                    
                    # focal plane B 
                    Fld_BB = mask2d*sft.sft(Fld_AA, nFPM, mB*lamC/lam)
        
                    # pupil plane C before Lyot stop
                    Fld_CC = Fld_AA - sft.isft(Fld_BB, nPup, mB*lamC/lam)
                
                    # pupil plane C after Lyot stop
                    Fld_LL = Fld_CC*LyotStop2d
        
                    # image plane D 
                    Fld_DD = sft.sft(Fld_LL, nImg, mD*diam) #dL_lst
        
                    # Intensity
                    Int_DD = np.abs(Fld_DD)**2
        
                    # Normalized intensity
                    Int_DD *= norm_peakDD0 #norm_peakDD0
                    
                    results[1,a,Ld,ob,s,i] = np.sum(Int_DD[aee[a]])
                    
                    # plt.figure(2*a)
                    # plt.imshow(Int_DD)
                    
                #     plt.figure(2*a+1)
                #     plt.imshow(Int_DD*(rad_mas[:,:,a+1]<=ap_mas/2.))
                    
                # stop

wvl_as_spxl = (wvl+'_angSep'+
               str(int(ap_roi))+'mas_spxl'+
               str(int(ap_mas))+'mas.fits')
fpath_psf_lst=(
    fdir_res / ('intensities_vs_params_'+wvl_as_spxl),
    fdir_res / ('results_vs_params_'+wvl_as_spxl),
    fdir_res / ('Parameters_throughput_'+wvl_as_spxl))

fits.writeto(fpath_psf_lst[0],intensities, overwrite=True)
fits.writeto(fpath_psf_lst[1],results, overwrite=True)
fits.writeto(fpath_psf_lst[2],np.array(throughput), overwrite=True)

for fpath in (fpath_psf_lst):
    fits.setval(fpath,'NPUP',value=nPup,comment='pupil size')
    fits.setval(fpath,'NFPM',value=nFPM,comment='FP coro. sampling')
    fits.setval(fpath,'NIMG',value=nImg,comment='image size')
    fits.setval(fpath,'LMIN',value=lam_min,comment='wavelength in meters')
    fits.setval(fpath,'LITV',value=lam_itv,comment='# of wvl intervals')
    fits.setval(fpath,'LSTP',value=lam_stp,comment='wvl step in meters')
    fits.setval(fpath,'LMBD',value=lamC,comment='wavelength in meters')
    fits.setval(fpath,'DIAM',value=D,comment='pupil dimater in meters')
    fits.setval(fpath,'PSCL',value=pscale,comment='plate scale in mas')
    fits.setval(fpath,'SFPM_MIN',value=mB_min,
                comment='minim. FPM (lam/D), first focal plane')
    fits.setval(fpath,'SFPM_ITV',value=mB_itv,
                comment='# of FPM intervals (lam/D), first focal plane')
    fits.setval(fpath,'SFPM_STP',value=mB_stp,
                comment='step FPM (lam/D), first focal plane')
    fits.setval(fpath,'OBST_MIN',value=obs_min,
                comment='minim. fractional pup. diameter')
    fits.setval(fpath,'OBST_ITV',value=obs_itv,
                comment='# of fractional obscuration')
    fits.setval(fpath,'OBST_STP',value=obs_stp,
                comment='step fractional obscuration')
    fits.setval(fpath,'DLYO_MIN',value=dL_min,
                comment='minim. fractional Lyot pupil diameter')
    fits.setval(fpath,'DLYO_ITV',value=dL_itv,
                comment='# of fractional Lyot pupil diameter')
    fits.setval(fpath,'DLYO_STP',value=dL_stp,
                comment='step fractional Lyot pupil diameter')
    fits.setval(fpath,'D_SPAXEL',value=ap_mas,
                comment='spaxel diameter in mas')
    fits.setval(fpath,'SPXL_ROI',value=ap_roi,
                comment='main spaxel location of interest in mas')
    fits.setval(fpath,'SPXL_MIN',value=ap_min,
                comment='min. spaxel diameter in mas')
    fits.setval(fpath,'SPXl_ITV',value=ap_itv,
                comment='# of spaxels intervals')
    fits.setval(fpath,'SPXL_STP',value=ap_stp,
                comment='step between two spaxel locations')
    fits.setval(fpath,'EPUP_FNM',value=elt_pup_fnm,
                comment='ELT pupil filename')
