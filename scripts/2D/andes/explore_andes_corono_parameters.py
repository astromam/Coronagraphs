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

from pathlib import Path
# import pdb
# import glob

import os

# from mpl_toolkits.axes_grid1 import AxesGrid


import time

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

# OPD map number in the files
nOPD = 2000

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


#%%
"""
### Working directory
"""
user = 'Adrien'
if user == 'Adrien':
    # File directory
    fdir_dat = Path('/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path('/Users/asimonnin/Desktop/PhD/Andes/Data_corono/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path('/Users/asimonnin/Desktop/PhD/Andes/Data_corono/plots/').resolve()

elif user == 'Mamadou':
    # File directory
    fdir_dat = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/data/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/plots/').resolve()

# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'
fdir_elt = '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/elt/'

# Filename and path for the ELT pupil
# fname_elt = 'Tel-Pupil.fits'
fname_elt = 'ELT_pupil_400.fits'
fpath_elt = fdir_pupil / fname_elt

#%%
"""
### Read file
"""
# Read ELT pupil 
# Pupil = fits.getdata(fpath_elt,)

Pupil = uniform_disk(nPup, nPup/2.)

    #%%
"""
### Coronagraphic components
"""
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)

"""
### Read OPDs
"""


fdir_opd = '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/PASSATA/20240228_112033.0_Seeing_0_65' #median

# fdir_opd = '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/PASSATA/20240228_053027.0_Seeing_0_57' #JQ2

fdir_opd = Path(fdir_opd).resolve()

# Filename and path for the OPD maps
flist_opd = os.listdir(fdir_opd) 
fpath_opd = [fdir_opd / flist_opd[i] for i in range(len(flist_opd)) if flist_opd[i].endswith('.fits')]
            
fpath_opd = sorted(fpath_opd)

fpath_opd = fpath_opd[200:2750]  #5min temps exposition
        
                
t0 = time.time()
# Read OPD maps for the nOPD files
OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])




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

diametre_lyot = np.arange(0.8,1.0,0.01)
obstruction = np.arange(0.3,0.5,0.01)
mB_conf = np.arange(3.0,5.0,0.2)

throughput = np.zeros((len(diametre_lyot),len(obstruction),len(mB_conf)))

diam_k=0

### Création des tableaux pour stocker les résultats
resultat_no_coro_no_turb_rad = []
resultat_no_coro_no_turb_int = []

resultat_w_coro_no_turb_rad = []
resultat_w_coro_no_turb_int = []

resultat_no_coro_w_turb_rad = []
resultat_no_coro_w_turb_int = []

resultat_w_woro_w_turb_rad = []
resultat_w_woro_w_turb_int = []

#%%

### Loop over the parameters

for diam in diametre_lyot: 
    obst_i=0
    for obst in obstruction:
        mb_i=0
        print(diam,obst)
        LyotStop2d = Pupil*(uniform_disk(nPup, diam*nPup/2)-uniform_disk(nPup, obst*nPup/2))
        coro_config = 'lyot'
                
        for s in range(len(mB_conf)) :
            """
            ### Selection of the good apodizer configuration according to FPM
            """
            mB = mB_conf[s]

            
            Pupil_coro = Pupil  

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

            throughput[diam_k,obst_i,mb_i] = ee_c/ee_a


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

            fname_Int_DD = 'wonoise_cor_'+coro_config+'.fits'
            fname_Int_DD0 = 'wonoise_'+coro_config+'.fits'
            fname_Int_DD0_no_lyot = 'wonoise_'+coro_config+'_no_lyot.fits'
            fname_fract_lum = "fraction_lum.npy"
            

            # filepath for the direct and coronagraphic images
                
            fpath_Int_DD  = fdir_res / fname_Int_DD
            fpath_Int_DD0  = fdir_res / fname_Int_DD0
            # fpath_Int_DD0_no_lyot  = fdir_res / fname_Int_DD0_no_lyot
            # save the direct and coronagraphic images


            # computation of the averaged intensity profiles of the images
            Int_DD_prf_avg, rad_DD_prf_avg = profile(Int_DD, ptype='mean')
            Int_DD0_prf_avg, rad_DD0_prf_avg = profile(Int_DD0, ptype='mean')

            Int_DD0_prf_std, rad_DD0_prf_std = profile(Int_DD0, ptype='std')
            Int_DD_prf_std, rad_DD_prf_std = profile(Int_DD, ptype='std')


            rad_DD0_prf_avg_lamD = rad_DD0_prf_avg * mD/nImg
            rad_DD_prf_avg_lamD = rad_DD_prf_avg * mD/nImg

            rad_DD0_prf_avg_mas = rad_DD0_prf_avg_lamD * lamD2mas
            rad_DD_prf_avg_mas = rad_DD_prf_avg_lamD * lamD2mas

            ### Save images
            resultat_w_coro_no_turb_rad.append(rad_DD_prf_avg)
            resultat_w_coro_no_turb_int.append(Int_DD_prf_avg)

            resultat_no_coro_no_turb_rad.append(rad_DD0_prf_avg)
            resultat_no_coro_no_turb_int.append(Int_DD0_prf_avg)
                



np.save(fdir_res / 'Paramaeters_diff_resultat_no_coro_no_turb_rad_{wl}.npy', resultat_no_coro_no_turb_rad)
np.save(fdir_res / 'Paramaeters_resultat_no_coro_no_turb_int_{wl}.npy', resultat_no_coro_no_turb_int)

np.save(fdir_res / 'Paramaeters_resultat_w_coro_no_turb_rad_{wl}.npy', resultat_w_coro_no_turb_rad)
np.save(fdir_res / 'Paramaeters_resultat_w_coro_no_turb_int_{wl}.npy', resultat_w_coro_no_turb_int)

np.save(fdir_res / 'Paramaeters_resultat_no_coro_w_turb_rad_{wl}.npy', resultat_no_coro_w_turb_rad)
np.save(fdir_res / 'Paramaeters_resultat_no_coro_w_turb_int_{wl}.npy', resultat_no_coro_w_turb_int)

np.save(fdir_res / 'Paramaeters_resultat_w_woro_w_turb_rad_{wl}.npy', resultat_w_woro_w_turb_rad)
np.save(fdir_res / 'Paramaeters_resultat_w_woro_w_turb_int_{wl}.npy', resultat_w_woro_w_turb_int)

np.save(fdir_res / 'Paramaeters_throughput_{wl}.npy', throughput)
