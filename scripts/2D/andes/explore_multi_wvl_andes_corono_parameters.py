#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@author: mndiaye, asimmonnin, asp
"""

# explore coronagraph parameters ranges and compute throughput, band averaged
# intensity profiles of the psf, with or without coro, at each discret 
# wavelength and at central lambda over D width for each discret wavelength

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

# wavelength in m, free values
# lamC = 2200e-9  #  some wvl unique value
# lam_min = 1960e-9  #  min value in range
# lam_max = 2460e-9  #  max value in range
# lam_itv = 4  #  nb of intervals in range --> nb+1 wvl's !

wvl = 'HsJ'

# if wvl == 'K' : 

#     # wavelengths in m
#     lamC = 2200e-9  #  some wvl unique value
#     lam_min = 1960e-9  #  min value in range
#     lam_max = 2460e-9  #  max value in range
#     lam_itv = 4  #  nb of intervals in range --> nb+1 wvl's !

if wvl == 'H' : 

    # wavelengths in m
    lamC = 1600e-9  #  some wvl unique value
    lam_min = 1440e-9  #  min value in range
    lam_max = 1770e-9  #  max value in range  
    lam_itv = 4  #  nb of intervals in range --> nb+1 wvl's ! 

elif wvl == 'Y2H': 

    # wavelengths in m
    lamC = 1600e-9  #  some wvl unique value
    lam_min = 950e-9  #  min value in range
    lam_max = 1800e-9  #  max value in range  
    lam_itv = 18  #  
    lam_lst = np.array(
        [9.50e-07, 1.00e-06, 1.05e-06, 1.10e-06, 1.15e-06, 1.20e-06,
         1.25e-06, 1.30e-06, 1.35e-06, 1.40e-06, 1.45e-06, 1.50e-06,
         1.55e-06, 1.60e-06, 1.65e-06, 1.70e-06, 1.75e-06, 1.80e-06])
    lam_stp = 50e-9 # np.median(np.diff(lam_lst))

elif wvl == 'HsJ': 

    # wavelengths in m
    lamC = 1600e-9  #  some wvl unique value
    lam_min = 950e-9  #  min value in range
    lam_max = 1800e-9  #  max value in range  
    lam_itv = 12  #  
    lam_lst = np.array(
        [1.15e-06, 1.20e-06,1.25e-06, 1.30e-06, 1.35e-06, 1.40e-06,
         1.45e-06, 1.50e-06,1.55e-06, 1.60e-06, 1.65e-06, 1.70e-06])
    lam_stp = 50e-9 # np.median(np.diff(lam_lst))

elif wvl == 'J' :

    # wavelengths in m
    lamC = 1240e-9  #  some wvl unique value
    lam_min = 1160e-9  #  min value in range
    lam_max = 1340e-9  #  max value in range
    lam_stp = 80e-9
    lam_itv = 4  #  nb of intervals in range --> nb+1 wvl's !  
    lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)

elif wvl == 'Y': 

    # wavelengths in m
    lamC = 1020e-9  #  some wvl unique value
    lam_min = 980e-9  #  min value in range
    lam_max = 1070e-9  #  max value in range  
    lam_itv = 4  #  nb of intervals in range --> nb+1 wvl's !  
    lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)

elif wvl == 'JH': 

    # wavelengths in m
    lamC = 1600e-9  #  some wvl unique value
    lam_min = 980e-9  #  min value in range
    lam_max = 1770e-9  #  max value in range  
    lam_itv = 10  #  
    lam_lst = np.array([1.16e-06, 1.20e-06, 1.24e-06, 1.28e-06, 1.32e-06,
                        1.44e-06, 1.52e-06, 1.60e-06, 1.68e-06, 1.76e-06])
    lam_stp = np.median(np.diff(lam_lst))
    
# elif wvl == 'HK': 

#     # wavelengths in m
#     lamC = 1600e-9  #  some wvl unique value
#     lam_min = 980e-9  #  min value in range
#     lam_max = 2460e-9  #  max value in range  
#     lam_itv = 20  #
#     lam_lst = np.array([1.44e-06, 1.52e-06, 1.60e-06, 1.68e-06, 1.76e-06,
#                         1.96e-06, 2.08e-06, 2.20e-06, 2.32e-06, 2.44e-06])
#     lam_stp = np.median(np.diff(lam_lst))
    
elif wvl == 'YJH': 

    # wavelengths in m
    lamC = 1600e-9  #  some wvl unique value
    lam_min = 980e-9  #  min value in range
    lam_max = 1770e-9  #  max value in range  
    lam_itv = 14  #  
    lam_lst = np.array([9.80e-07, 1.00e-06, 1.02e-06, 1.04e-06, 1.06e-06,
                        1.16e-06, 1.20e-06, 1.24e-06, 1.28e-06, 1.32e-06,
                        1.44e-06, 1.52e-06, 1.60e-06, 1.68e-06, 1.76e-06])
    lam_stp = np.median(np.diff(lam_lst))

# elif wvl == 'YJHK': 

#     # wavelengths in m
#     lamC = 1600e-9  #  some wvl unique value
#     lam_min = 980e-9  #  min value in range
#     lam_max = 2460e-9  #  max value in range  
#     lam_itv = 20  #
#     lam_lst = np.array([9.80e-07, 1.00e-06, 1.02e-06, 1.04e-06, 1.06e-06,
#                         1.16e-06, 1.20e-06, 1.24e-06, 1.28e-06, 1.32e-06,
#                         1.44e-06, 1.52e-06, 1.60e-06, 1.68e-06, 1.76e-06,
#                         1.96e-06, 2.08e-06, 2.20e-06, 2.32e-06, 2.44e-06])
#     lam_stp = np.median(np.diff(lam_lst))

if len(wvl)==1:

    lam_stp = np.floor(np.ceil((lam_max-lam_min)*1e9/lam_itv)/10)*1e-8
    lam_lst = (np.arange(lam_min,lam_max,lam_stp) if lam_max != lam_min
               else [lamC])

lamC = 1600e-9 # fix lam_c for all band(s)

nL = len(lam_lst)

print(nL, lam_stp, lam_lst)


# if lambda of interest / D
rW_mas = 7.  # spaxel of 7 mas (05/2025->) / 10. -> 03/2025
    
# Pupil diameter in m 
D = 38.54

# lyot mask vane width in pixels
v_width = 3.  #  3. pour fichier ELT_pupil_400.fits non modifie
# vanes = six_arms(nPup, v_width) -> 03/2025

donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now
print(donow, wvl)

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
    # File directory
    fdir_dat = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/results/').resolve()

elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    # File directory
    fdir_dat = Path(fdir_base / 'data/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path(fdir_base / 'results/').resolve()

#%%
# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'

# Filename and path for the ELT pupil
# fname_elt = 'Tel-Pupil.fits'
fname_elt = 'ELT_pupil_400.fits'
fpath_elt = fdir_pupil / fname_elt

# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)


    #%%
"""
### Coronagraphic components
"""
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)


#%%

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# conversion lam/D to mas
# lamD2mas = (lamC/D)*mas2rad

# plate scale in mas per pixel
pscale = 0.3

# field of view
# in mas
fov_mas = nImg * pscale
# in radians
fov_rdn = fov_mas * rad2mas
# in multiple of reference lambda (lamC) over D, 05/2025 -> 
mD_ref = fov_rdn / ( lamC / D )

# FoV in lam/D in the final image plane D
# mD = 58.393*(nImg/1600) #* (lam_0/lam)  #  simu 2024
# print(mD)

"""
Pupil configuration
"""
### All parameters to test

dL_min = 0.81
dL_max = 0.96
dL_stp = 0.01
diametre_lyot = np.arange(dL_min,dL_max+dL_stp,dL_stp)
obs_min = 0.30
obs_max = 0.44
obs_stp = 0.01
obstruction = np.arange(obs_min,obs_max+obs_stp,obs_stp)
# mBmB_min = 3.0
mB_max = 4.5
mB_min = 3.0
mB_stp = 0.1
mB_conf = np.arange(mB_min,mB_max+mB_stp,mB_stp)

throughput = np.zeros((len(diametre_lyot),len(obstruction)))

### Création des tableaux pour stocker les résultats
results_no_coro_no_turb = np.zeros(
    (2, len(diametre_lyot), len(obstruction), len(mB_conf), nImg//2))

results_w_coro_no_turb = results_no_coro_no_turb.copy()
Int_D0_prf_avg = results_no_coro_no_turb.copy()
Int_D_prf_avg = results_no_coro_no_turb.copy()


#%%

# stackoveflow...
a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
b_ = a_.copy()
aa, bb = np.meshgrid(a_, b_)
rad_pix = np.sqrt(aa**2 + bb**2)
# rad_mas = rad_pix * (mas2rad * 58.393 / (D * 1e9) )
rad_mas = rad_pix * (mas2rad * fov_mas / (2. * D * 1e9) )

#  angular separation
# aS = np.arange(nImg//2) * (mas2rad * 58.393 / (D * 1e9) )
aS = np.arange(nImg//2) * (mas2rad * fov_mas / (2. * D * 1e9) )

### Loop over the parameters
for i in np.arange(nL):
    
    lam = lam_lst[i]
    
    # conversion lam/D to mas
    lamD2mas = (lam/D)*mas2rad
    
    # FoV in lam/D in the final image plane D
    # mD = 58.393*(nImg/(lam*1e9))  #  simu 2024 -> 03/2025
    mD = mD_ref * lamC / lam # 05/2025 ->
        
    print('lambda (nm):' ,np.round(lam*1e9,0), '. Field of view (lam/D):',
          np.round(mD,3))
    
    for dL in range(len(diametre_lyot)): 
        diam = diametre_lyot[dL]

        for ob in range(len(obstruction)):
            obst=obstruction[ob]

            LyotStop2d = (Pupil.copy() * # vanes *
                          (uniform_disk(nPup, diam*nPup/2) -
                           uniform_disk(nPup, obst*nPup/2)))
            
            for s in range(len(mB_conf)) :
                """
                ### Selection of the good apodizer configuration according to FPM
                """
                mB = mB_conf[s]
                
                Pupil_coro = Pupil.copy()  # .copy(), just in case ... asp
    
                """
                ### Compute perfect PSF
                """
                # Field in the entrance pupil plane A
                Fld_AA0 = Pupil_coro
    
                int_a = np.abs(Fld_AA0)**2
    
                ee_a = np.sum(int_a)
    
                # Field in the entrance pupil plane C
                Fld_CC0 = Fld_AA0 * 1.*LyotStop2d
    
                int_c = np.abs(Fld_CC0)**2
    
                ee_c = np.sum(int_c)
    
                if s == 0:
                    throughput[dL,ob] = ee_c/ee_a
    
                # Field in the image plane D (no coronagraph)*
                Fld_DD0 = sft.sft(Fld_CC0, nImg, mD*diam) #diametre_lyot
    
                # Intensity 
                Int_DD0 = np.abs(Fld_DD0)**2
    
                # Normalized intensity
                norm_peakDD0 = 1/np.max(Int_DD0)
    
        
                Int_DD0 *= norm_peakDD0 #norm_peakDD0
    
                """
                ### Calculation of corono image without atmospheric turbulences
                """
                    
                Fld_AA = Pupil_coro
                # focal plane B 
                Fld_BB = mask2d*sft.sft(Fld_AA, nFPM, mB*lamC/lam)
    
                # pupil plane C before Lyot stop
                Fld_CC = Fld_AA - sft.isft(Fld_BB, nPup, mB*lamC/lam)
    
    
                # pupil plane C after Lyot stop
                Fld_LL = Fld_CC*LyotStop2d
    
                # image plane D 
                Fld_DD = sft.sft(Fld_LL, nImg, mD*diam) #diametre_lyot
    
                # Intensity
                Int_DD = np.abs(Fld_DD)**2
    
                # Normalized intensity
                Int_DD *= norm_peakDD0 #norm_peakDD0
    
                # computation of the averaged intensity profiles of the images
                Int_DD_prf_avg, rad_DD_prf_avg = profile(Int_DD, ptype='mean')
                Int_DD0_prf_avg,rad_DD0_prf_avg = profile(Int_DD0,ptype='mean')
    
                ### Save images
                results_w_coro_no_turb[0,dL,ob,s,:] = rad_DD_prf_avg
                results_w_coro_no_turb[1,dL,ob,s,:] += Int_DD_prf_avg/nL
    
                results_no_coro_no_turb[0,dL,ob,s,:] = rad_DD0_prf_avg
                results_no_coro_no_turb[1,dL,ob,s,:] += Int_DD0_prf_avg/nL
                                
                for p in range(nImg//2):
                    
                    ring_val = np.where(np.abs(rad_mas-aS[p])<=rW_mas/2)
                    Int_D0_prf_avg[1,dL,ob,s,p] += np.mean(Int_DD0[ring_val])/nL
                    Int_D_prf_avg[1,dL,ob,s,p] += np.mean(Int_DD[ring_val])/nL

                Int_D0_prf_avg[0,dL,ob,s,:] = rad_DD0_prf_avg
                Int_D_prf_avg[0,dL,ob,s,:] = rad_DD0_prf_avg


fpath_psf_lst=(
    fdir_res / ('Parameters_results_contrast_no_coro_no_turb_'+wvl+'.fits'),
    fdir_res / ('Parameters_results_contrast_w_coro_no_turb_'+wvl+'.fits'),
    fdir_res / ('Parameters_results_no_coro_no_turb_'+wvl+'.fits'),
    fdir_res / ('Parameters_results_w_coro_no_turb_'+wvl+'.fits'),
    fdir_res / ('Parameters_throughput_'+wvl+'.fits'))

fits.writeto(fpath_psf_lst[0],Int_D0_prf_avg, overwrite=True)
fits.writeto(fpath_psf_lst[1],Int_D_prf_avg, overwrite=True)
fits.writeto(fpath_psf_lst[2],results_no_coro_no_turb, overwrite=True)
fits.writeto(fpath_psf_lst[3],results_w_coro_no_turb, overwrite=True)
fits.writeto(fpath_psf_lst[4],np.array(throughput), overwrite=True)

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
    fits.setval(fpath,'SFPM_MAX',value=mB_max,
                comment='maxim. FPM (lam/D), first focal plane')
    fits.setval(fpath,'SFPM_STP',value=mB_stp,
                comment='step FPM (lam/D), first focal plane')
    fits.setval(fpath,'OBST_MIN',value=obs_min,
                comment='minim. fractional pup. diameter')
    fits.setval(fpath,'OBST_MAX',value=obs_max,
                comment='maxim. fractional obscuration')
    fits.setval(fpath,'OBST_STP',value=obs_stp,
                comment='step fractional obscuration')
    fits.setval(fpath,'DLYO_MIN',value=dL_min,
                comment='minim. fractional Lyot pupil diameter')
    fits.setval(fpath,'DLYO_MAX',value=dL_max,
                comment='maxim. fractional Lyot pupil diameter')
    fits.setval(fpath,'DLYO_STP',value=dL_stp,
                comment='step fractional Lyot pupil diameter')
    fits.setval(fpath,'LYO_VW',value=v_width,
                comment='Lyot mask vane width in pixels')


