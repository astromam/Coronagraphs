#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@author: mndiaye, asimmonnin, asp
"""


#%%
"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import slow_fourier_transform as sft
from uniform_disk import uniform_disk
from psf_profile import profile
from draw_vanes import six_vanes

import os
from pathlib import Path
from datetime import datetime  #  asp for datetime of now

#definir font size
plt.rcParams.update({'font.size': 20})            


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

wl = 'H'

if wl == 'H' : 

    lam = 1600e-9
    #liste de 11 wavelengths pour H band
    # lam_list = [1.4e-6, 1.44e-6, 1.48e-6, 1.52e-6, 1.56e-6, 1.6e-6, 1.64e-6,
    #             1.68e-6, 1.72e-6, 1.76e-6, 1.80e-6]

elif wl == 'J' :

    lam = 1200e-9
elif wl == 'Y': 
    
    lam = 1000e-9

# Pupil diameter in m 
D = 38.54

# lyot mask vane width in pixels
v_width = 8.  #  3. pour fichier ELT_pupil_400.fits non modifie
vanes = six_vanes(nPup, v_width)

donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now

#%%
"""
### Working directory
"""
user = 'Alain'
if user == 'Alain':
    fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()  # opd's seed value
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots
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
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/plots/').resolve()

elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    # File directory
    fdir_dat = Path(fdir_base / 'data/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path(fdir_base / 'results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path(fdir_base / 'plots/').resolve()

# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'
# fdir_elt = '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/elt/'

# Filename and path for the ELT pupil
# fname_elt = 'Tel-Pupil.fits'
fname_elt = 'ELT_pupil_400.fits'
fpath_elt = fdir_pupil / fname_elt

#%%
"""
### Read file
"""
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
lamD2mas = (lam/D)*mas2rad

# plate scale in mas per pixel
pscale = 0.3

# FPM size in lam/D in the focal plane B
# mB = 4 

# FoV in lam/D in the final image plane D
mD = 58.393*(nImg/1600) #* (lam_0/lam)
# print(mD)


"""
Pupil configuration
"""
### All parameters to test

dL_min = 0.8
dL_max = 1.0
dL_stp = 0.01
diametre_lyot = np.arange(dL_min,dL_max+dL_stp,dL_stp)
obs_min = 0.3
obs_max = 0.5
obs_stp = 0.01
obstruction = np.arange(obs_min,obs_max+obs_stp,obs_stp)
mB_min = 3.0
mB_max = 5.0
mB_stp = 0.2
mB_conf = np.arange(mB_min,mB_max+mB_stp,mB_stp)

throughput = np.zeros((len(diametre_lyot),len(obstruction)))

diam_k=0

### Création des tableaux pour stocker les résultats
resultat_no_coro_no_turb = np.zeros(
    (2, len(diametre_lyot), len(obstruction), len(mB_conf), nImg//2))

resultat_w_coro_no_turb = resultat_no_coro_no_turb.copy()


#%%

### Loop over the parameters

for dL in range(len(diametre_lyot)): 
    diam = diametre_lyot[dL]
    obst_i=0
    for ob in range(len(obstruction)):
        obst=obstruction[ob]
        mb_i=0
        # print(np.round(diam,3),np.round(obst,3))
        LyotStop2d = (Pupil.copy() * vanes * (uniform_disk(nPup, diam*nPup/2) -
                                              uniform_disk(nPup, obst*nPup/2)))
        coro_config = 'lyot'
                
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

            # throughput[diam_k,obst_i,mb_i] = ee_c/ee_a
            if s ==0:
                throughput[dL,ob] = ee_c/ee_a

            # Field in the image plane D (no coronagraph)*
            Fld_DD0 = sft.sft(Fld_CC0, nImg, mD*diam) #diametre_lyot

            # Intensity 
            Int_DD0 = np.abs(Fld_DD0)**2

            # Normalized intensity
            norm_peakDD0 = 1/np.max(Int_DD0)

    
            Int_DD0 *= norm_peakDD0 #norm_peakDD0

            """
            ### Calcul of corono image without atmospheric turbulences
            """
                
            Fld_AA = Pupil_coro
            # focal plane B 
            Fld_BB = mask2d*sft.sft(Fld_AA, nFPM, mB)

            # pupil plane C before Lyot stop
            Fld_CC = Fld_AA - sft.isft(Fld_BB, nPup, mB)


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
            Int_DD0_prf_avg, rad_DD0_prf_avg = profile(Int_DD0, ptype='mean')

            # Int_DD0_prf_std, rad_DD0_prf_std = profile(Int_DD0, ptype='std')
            # Int_DD_prf_std, rad_DD_prf_std = profile(Int_DD, ptype='std')

            # rad_DD0_prf_avg_lamD = rad_DD0_prf_avg * mD/nImg
            # rad_DD_prf_avg_lamD = rad_DD_prf_avg * mD/nImg

            # rad_DD0_prf_avg_mas = rad_DD0_prf_avg_lamD * lamD2mas
            # rad_DD_prf_avg_mas = rad_DD_prf_avg_lamD * lamD2mas

            ### Save images
            resultat_w_coro_no_turb[0,dL,ob,s,:] = rad_DD_prf_avg
            resultat_w_coro_no_turb[1,dL,ob,s,:] = Int_DD_prf_avg

            resultat_no_coro_no_turb[0,dL,ob,s,:] = rad_DD0_prf_avg
            resultat_no_coro_no_turb[1,dL,ob,s,:] = Int_DD0_prf_avg
                

fits.writeto(
    fdir_res / ('Parameters_resultat_no_coro_no_turb_'+wl+'.fits'),
    resultat_no_coro_no_turb, overwrite=True)
fits.writeto(
    fdir_res / ('Parameters_resultat_w_coro_no_turb_'+wl+'.fits'),
    resultat_w_coro_no_turb, overwrite=True)
fits.writeto(
    fdir_res / ('Parameters_throughput_'+wl+'.fits'), np.array(throughput),
    overwrite=True)

fpath_psf_lst=(
    fdir_res / ('Parameters_resultat_no_coro_no_turb_'+wl+'.fits'),
    fdir_res / ('Parameters_resultat_w_coro_no_turb_'+wl+'.fits'),
    fdir_res / ('Parameters_throughput_'+wl+'.fits'))

for fpath in (fpath_psf_lst):
    fits.setval(fpath,'NPUP',value=nPup,comment='pupil size')
    fits.setval(fpath,'NFPM',value=nFPM,comment='FP coro. sampling')
    fits.setval(fpath,'NIMG',value=nImg,comment='image size')
    fits.setval(fpath,'LMBD',value=lam,comment='wavelength in meters')
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

