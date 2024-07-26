#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@authors: mndiaye, asimonnin, asp
"""

#%%
"""
### Initialization
"""

import numpy as np
# pythonpath to update possibly...
import slow_fourier_transform as sft
from uniform_disk import uniform_disk
from psf_profile import profile

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid

from astropy.io import fits

import os
import time
from pathlib import Path
from datetime import datetime  #  asp for datetime of now
#  import pdb

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!


#%%
"""
### Parameters
"""
# Pupil size
nPup = 400  #  even/pair!

# Sampling of the coronagraph focal plane mask
nFPM = 100

# Image size
nImg = 400  #*1.6 #  #  even/pair!

# angular separation of interest in mas
as_oi = 25.

# # of phase screens
nOPD = 2000

# wavelengths in m
lamC = 1600e-9  #  some wvl unique value
lam_min = 1440e-9  #  min value in range
lam_max = 1770e-9  #  max value in range  #  2450e-9 // 1800e-9
lam_itv = 4  #  nb of intervals in range --> nb+1 wvl's !  #  18 // 10
lam_stp = np.floor(np.ceil((lam_max-lam_min)*1e9/lam_itv)/10)*1e-8  # wvl step
lam_lst = np.arange(lam_min,lam_max,lam_stp) if lam_max != lam_min else [lamC]
nL = len(lam_lst)

print(nL, lam_stp, lam_lst)

# Pupil diameter in m 
D = 38.54

# conversion l radian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# plate scale in mas per pixel
pscale = 0.3

# FPM size in lam/D in the focal plane B
# 4. with 'ELT_pupil_400.fits'
# 3.8 with 'Tel-Pupil.fits', old!
mB = 4. # 3.8  #  at 1600nm !!

lamCD2mas = (lamC/D)*mas2rad

"""
### Coronagraphic components
"""
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)
# diam = 0.86 obs = 0.30 with 'ELT_pupil_400.fits'
# diam = 0.90 obs = 0.38 with 'Tel-Pupil.fits'
diam = 0.86 # diameter of the pupil in fraction of the pupil size
obst = 0.30 # diameter of the central obscuration in fraction of the pupil size

# dispersion mas/m
disp = 8e7

# 2D array pupil slope for tilt
frq = np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5

# 2 sizes of spaxels 10 mas and 100 mas

# datetime of script execution
donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now
print('date of now : ', donow)


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()  # opd's seed value
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots

elif user == 'Adrien':
    # File directory
    fdir_dat = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/').resolve()
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
    fdir_dat = Path( fdir_base / 'data' ).resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path( fdir_base / 'results' ).resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path( fdir_base / 'plots' ).resolve()

# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'

fdir_res = fdir_res / donow
fdir_plt = fdir_plt / donow


#%%
"""
data cubes for all psfs and profiles
"""

Int_DD0 = np.zeros([nL, nImg, nImg])
Int_DD = np.zeros([nL, nImg, nImg])
Int_D0 = np.zeros([nL, nImg, nImg])
Int_D = np.zeros([nL, nImg, nImg])
Int_D0_prf_avg = np.zeros([nL, 2, nImg//2])
Int_D_prf_avg = np.zeros([nL, 2, nImg//2])


#%%

# Filename and path for the ELT pupil  //  'Tel-Pupil.fits' <-- OLD
fname_elt = 'ELT_pupil_400.fits' # New pupil with new spider
fpath_elt = fdir_pupil / fname_elt

"""
### Read file
"""
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)

# plt.figure(0)
# plt.clf()
# plt.imshow(Pupil)
# plt.title('ELT pupil')
# plt.show()
# plt.savefig(fdir_plt / 'New_ELT_pupil.png', dpi=300)
# plt.show()
# plt.close()

# Lyot stop
LyotStop2d = Pupil*(
    uniform_disk(nPup, diam*nPup/2)-uniform_disk(nPup, obst*nPup/2))


#%%
"""
### working directory of the OPD files
"""
# Directory for the OPDs with the corresponding seed value 
# (from HARMONI simulation)
# fdir_opd   = fdir_dat / 'OPD_Harmoni' / str(seed)

# New set of OPDs from PASSATA 

# opds_dir=('OPDs_PASSATA/OPD/20231124_090126.0/',
#               'OPDs_PASSATA/OPD/20231122_142204.0/',
#               'OPDs_PASSATA/OPD/20240227_234849-007/20240227_234849.0',
#               'OPDs_PASSATA/OPD/20240228_053027-001/20240228_053027.0',
#               'OPDs_PASSATA/OPD/20240302_000411-003/20240302_000411.0',
#               'OPDs_PASSATA/OPD/20240313_133532-004/20240313_133532.0',
#               'OPDs_PASSATA/OPD/20240228_112033-002/20240228_112033.0')
opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_091216',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_100705',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_103452',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_105822',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_111658',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_113534',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_121247',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_181033',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_183817',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_190418',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_192251',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_194126',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_195959',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_201835',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_203708',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-001/JQ3/20240521_200540',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-004/JQ3/20240521_181105',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-002/JQ3/20240521_213115',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-005/JQ3/20240521_222747',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-003/JQ3/20240521_210334',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_190439/JQ3/20240527_190439',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_195648/JQ3/20240527_195648',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_204843/JQ3/20240527_204843',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_214043/JQ3/20240527_214043',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_223245/JQ3/20240527_223245',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_161522/JQ4/20240528_161522',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_163416/JQ4/20240528_163416',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_165414/JQ4/20240528_165414',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_171307/JQ4/20240528_171307',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_173157/JQ4/20240528_173157',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_175246/JQ4/20240528_175246',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_181244/JQ4/20240528_181244',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_183130/JQ4/20240528_183130',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_185018/JQ4/20240528_185018',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_190906/JQ4/20240528_190906')

for dir_nb in range(len(opds_dir)):

    new_spider_flare = fdir_dat / opds_dir[dir_nb]
    fdir_opd   = new_spider_flare


#%%%
    
    opd_set = os.path.basename(fdir_opd).split('.')[0]
    print(opd_set)
    
    # opd_set='toto'
    os.makedirs(fdir_res / opd_set, exist_ok=True)
    os.makedirs(fdir_plt / opd_set, exist_ok=True)
    
    
    #%%
    
    # Filename and path for the OPD maps
    flist_opd = os.listdir(fdir_opd) 
    nof = len(flist_opd)
    fpath_opd = [fdir_opd / flist_opd[i] for i in range(nof)]
    fpath_opd = sorted(fpath_opd)
    nW = int(np.floor((nof/nOPD)+1))
    if nW>2:
        fpath_opd = [fpath_opd[i] for i in range(1,nof,nW)]
    nOPD = len(fpath_opd)
    print('sample size of OPD files:', nOPD)
    
    
    #%%
    """
    ### Read OPD files
    """
    
    t0 = time.time()
    # Read OPD maps for the nOPD files
    OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])
    t1 = time.time()
    print(f'OPD reading file time: {t1-t0:.3f}s')
    
    OPD_arr = OPD_arr*1e-9 # convert OPD from nm `to m if new OPD with new pupil
    
    # plt.figure(2)
    # plt.clf()
    # plt.imshow(OPD_arr[200]*Pupil)
    # plt.colorbar()
    # plt.show()
    
    # opd_std = np.std(OPD_arr,axis=0)
    # print(np.mean(opd_std[:]), np.std(opd_std[:]))
    
    # exit() # quit()?
    #%%
    
    for i in np.arange(nL):
        
        lam = lam_lst[i]
        
        dLam = lam - lamC
        tilt = dLam * disp * rad2mas
        
        # conversion lam/D to mas
        lamD2mas = (lam/D)*mas2rad
        
        # FoV in lam/D in the final image plane D
        mD = 58.393*(nImg/(lam*1e9))
        
        print('lambda (nm):' ,np.round(lam*1e9,0), '. Field of view (lam/D):',
              np.round(mD,3))
        
            
        #%%
        """
        ### Compute perfect PSF
        """
        
        # Field in the entrance pupil plane A
        Fld_AA0 = Pupil * 1.*LyotStop2d
        
        # Field in the image plane D (no coronagraph)
        Fld_DD0 = sft.sft(Fld_AA0, nImg, mD*diam)
        
        # Intensity 
        Int_DD0[i,:] = np.abs(Fld_DD0)**2
        
        # Normalized intensity
        norm_peakDD0 = 1/np.max(Int_DD0[i,:])
        Int_DD0[i,:] *= norm_peakDD0
        
        
        #%%
        """
        ### Compute perfect coronographic image
        """
        
        # pupil plane A
        Fld_AA = Pupil * 1.
        # focal plane B 
        Fld_BB = mask2d*sft.sft(Fld_AA, nFPM, mB)
        
        # pupil plane C before Lyot stop
        Fld_CC = Fld_AA - sft.isft(Fld_BB, nPup, mB)
        
        # # pupil plane C after Lyot stop
        Fld_LL = Fld_CC*LyotStop2d
        
        # image plane D 
        Fld_DD = sft.sft(Fld_LL, nImg, mD*diam)
        
        # Intensity
        Int_DD[i,:] = np.abs(Fld_DD)**2
        
        # Normalized intensity
        Int_DD[i,:] *= norm_peakDD0
        
        
        # pdb.set_trace()
        #%%
        
        # plt.figure(3)
        # plt.clf()
        # plt.subplot(131)
        # plt.imshow(np.abs(Fld_CC)**2)
        # plt.title('bef. Lyot stop')
        # plt.subplot(132)
        # plt.imshow(np.log10(Int_DD0[i,:]), vmax =0, vmin = -5, cmap ='inferno')
        # plt.title('perfect psf')
        # plt.subplot(133)
        # plt.imshow(np.log10(Int_DD[i,:]), vmax =0, vmin = -5, cmap ='inferno')
        # plt.title('perf. coro. psf')
        # plt_fnm = 'coronagraphic_image_lyotstop_'+str(int(lam*1e9))+'nm.png'
        # plt.savefig(fdir_plt / opd_set / plt_fnm, dpi=300)
        # plt.suptitle(f'lambda: $\lambda$={lam*1e6:.3f}$\mu$m')
        # plt.show()
        # plt.close()
        
        
    #%%    
        # computation of the averaged intensity profiles of the images  
        # Int_DD0_prf_avg, rad_DD0_prf_avg = profile(Int_DD0[i,:], ptype='mean')
        # Int_DD_prf_avg, rad_DD_prf_avg = profile(Int_DD[i,:], ptype='mean')
        
        
        # computation of the standard deviation intensity profiles of the images
        # Int_DD0_prf_std, rad_DD0_prf_std = profile(Int_DD0[i,:], ptype='std')
        # Int_DD_prf_std, rad_DD_prf_std = profile(Int_DD[i,:], ptype='std')
        
        
        #%%
        """
        ### Compute PSF (with errors)
        """
        
        t0 = time.time()
        # Int_D0 = np.zeros((nImg, nImg))
        # Int_D0_2 = np.zeros((nImg, nImg))
        # 531 - 300 = 231
        # OPD_sum = OPD_arr[0]*0 
        # for iOPD in range(nOPD):
        
        for iOPD in range(nOPD):
            
            # Field in the entrance pupil plane A
            Fld_A0 = (Pupil * np.exp(1j*2*np.pi*OPD_arr[iOPD]/(lam))
                      * np.exp(1j * tilt * D * frq / (nPup * lam)) * LyotStop2d)
            
            # Field in the image plane D (no coronagraph)
            Fld_D0 = sft.sft(Fld_A0, nImg, mD*diam) ## Avec Lyot stop
            # Fld_D0_2 = sft.sft(Fld_A0_2, nImg, mD)  # Sans Lyot stop
            
            # Intensity 
            Int_D0[i,:] += np.abs(Fld_D0)**2 ## Avec Lyot stop
            # Int_D0_2 += np.abs(Fld_D0_2)**2 # Sans Lyot stop
        
        t1 = time.time()       
        print(f'PSF computation time: {t1-t0:.3f}s')  
        
        # Normalized intensity
        Int_D0[i,:] /= nOPD ## Avec Lyot stop
        # Int_D0_2 /= flare[i][1]-flare[i][0]# Sans Lyot stop
        
        # Normalized intensity
        norm_peakD0 = 1/np.max(Int_D0[i,:])
        # norm_peakD0_2 = 1/np.max(Int_D0_2)
        
        Int_D0[i,:] *= norm_peakD0
        # Int_D0 *= norm_peakDD0 #norm_peakD0 DD0 to have common scale
        # Int_D0_2 *= norm_peakD0_2
        
        """
        ### Compute coronographic image (with errors)
        """
        t0 = time.time()
        # Int_D = np.zeros((nImg, nImg))
        
        for iOPD in range(nOPD):
            # pupil plane A
            Fld_A0 = (Pupil * np.exp(1j*2*np.pi*OPD_arr[iOPD]/(lam))
                      * np.exp(1j * tilt * D * frq / (nPup * lam)))
            
            # focal plane B 
            Fld_B = mask2d*sft.sft(Fld_A0, nFPM, mB)
            
            # pupil plane C before Lyot stop
            Fld_C = Fld_A0 - sft.isft(Fld_B, nPup, mB)
            
            # pupil plane C after Lyot stop
            Fld_L = Fld_C*LyotStop2d
            
            # image plane D 
            Fld_D = sft.sft(Fld_L, nImg, mD*diam)
            
            # Intensity
            Int_D[i,:] += np.abs(Fld_D)**2    
    
        t1 = time.time()       
        
        print(f'Coro image computation time: {t1-t0:.3f}s')  
        
        # Normalized intensity
        Int_D[i,:] /= nOPD
        
        # Normalized intensity
        # Int_D *= norm_peakDD0 #norm_peakD0 DD0 to have common scale
        Int_D[i,:] *= norm_peakD0
        
        """
        ### Compute the radial intensity profiles of the images
        """
        # computation of the averaged intensity profiles of the images   
        # Int_D0_prf_avg2, rad_D0_prf_avg = profile(Int_D0_2, ptype='mean')
        Int_D0_prf_avg[i,1,:], rad_D0_prf_avg = profile(Int_D0[i,:], ptype='mean')
        Int_D_prf_avg[i,1,:], rad_D_prf_avg = profile(Int_D[i,:], ptype='mean')
        
        # computation of the standard deviation intensity profiles of the images
        # Int_D0_prf_std2, rad_D0_prf_std = profile(Int_D0_2, ptype='std')
        # Int_D0_prf_std, rad_D0_prf_std = profile(Int_D0, ptype='std')
        # Int_D_prf_std, rad_D_prf_std = profile(Int_D, ptype='std')
        
        # convert pixel scale into lam/D scale for the x-axis
        rad_D0_prf_avg_lamD = rad_D0_prf_avg * mD/nImg
        rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg
        
        # rad_D0_prf_avg_mas = rad_D0_prf_avg_lamD * lamD2mas
        Int_D0_prf_avg[i,0,:] = rad_D0_prf_avg_lamD * lamD2mas
        rad_D_prf_avg_mas = rad_D_prf_avg_lamD * lamD2mas
        Int_D_prf_avg[i,0,:] = rad_D_prf_avg_lamD * lamD2mas
        # rad_DD0_prf_avg_mas = rad_DD0_prf_avg_lamD * lamD2mas 
    
    
    #%%
    """
    save data cubes in fits files with keywords
    """
    
    # filename for the direct and coronagraphic images and profiles
    # fname_Int_DD0 = 'wonoise_psf_'+donow+'.fits'
    # fname_Int_DD = 'wonoise_coro_psf_'+donow+'.fits'
    fname_Int_D0 = 'ao_corr_psf_'+donow+'.fits'
    fname_Int_D = 'ao_corr_coro_psf_'+donow+'.fits'
    fname_Prf_D0 = 'ao_corr_psf_profile_'+donow+'.fits'
    fname_Prf_D = 'ao_corr_coro_psf_profile_'+donow+'.fits'
    
    
    # filepath for the direct and coronagraphic images
    # fpath_Int_DD0 = fdir_res / opd_set / fname_Int_DD0
    # fpath_Int_DD  = fdir_res / opd_set / fname_Int_DD
    fpath_Int_D0 = fdir_res / opd_set / fname_Int_D0
    fpath_Int_D  = fdir_res / opd_set / fname_Int_D
    fpath_Prf_D0 = fdir_res / opd_set / fname_Prf_D0
    fpath_Prf_D  = fdir_res / opd_set / fname_Prf_D
    
    
    # save the direct and coronagraphic images
    # fits.writeto(fpath_Int_DD0, Int_DD0, overwrite=True)
    # fits.writeto(fpath_Int_DD, Int_DD, overwrite=True)
    fits.writeto(fpath_Int_D0, Int_D0, overwrite=True)
    fits.writeto(fpath_Int_D, Int_D, overwrite=True)
    fits.writeto(fpath_Prf_D0, Int_D0_prf_avg, overwrite=True)
    fits.writeto(fpath_Prf_D, Int_D_prf_avg, overwrite=True)
    
    fpath_psf_lst=(fpath_Int_D0, fpath_Int_D,fpath_Prf_D0, fpath_Prf_D)
    # fpath_psf_lst=(fpath_Int_DD0, fpath_Int_DD, fpath_Int_D0, fpath_Int_D,
    #                fpath_Prf_D0, fpath_Prf_D)
    
    for fpath in fpath_psf_lst:
        fits.setval(fpath,'NPUP',value=nPup,comment='pupil size')
        fits.setval(fpath,'NFPM',value=nFPM,comment='FP coro. sampling')
        fits.setval(fpath,'NIMG',value=nImg,comment='image size')
        fits.setval(fpath,'NOPD',value=nOPD,comment='number of OPD files')
        fits.setval(fpath,'LMIN',value=lam_min,comment='wavelength in meters')
        #fits.setval(fpath,'LMAX',value=lam_max,comment='wavelength in meters')
        fits.setval(fpath,'LITV',value=lam_itv,comment='# of wvl intervals')
        fits.setval(fpath,'LSTP',value=lam_stp,comment='wvl step in meters')
        fits.setval(fpath,'DIAM',value=D,comment='pupil dimater in meters')
        fits.setval(fpath,'PSCL',value=pscale,comment='plate scale in mas')
        fits.setval(fpath,'SFPM',value=mB,comment='FPM (lam/D), first focal plane')
        fits.setval(fpath,'FDIA',value=diam,comment='fractional pup. diameter')
        fits.setval(fpath,'OBST',value=obst,comment='fractional obscuration')
        fits.setval(fpath,'OPDS',value=opd_set,comment='opd set creation date')
    

    #%%
    """
    display images
    # """
        
    # # filename of the plot
    # fname_images_svg = 'ao_corrected_coro_psf_'+donow+'.svg'
    # fname_images_pdf = 'ao_corrected_coro_psf_'+donow+'.pdf'
    
    # # filepath for the direct and coronagraphic images
    # fpath_images_svg = fdir_plt / opd_set / fname_images_svg
    # fpath_images_pdf = fdir_plt / opd_set / fname_images_pdf
    
    # index of wvl to display
    # iD = [0,nL//4,nL//2,nL*3//4,nL-1]
    
    # boundaries for the images in log scale
    # vmin0 = -7
    # vmax0 = 0
    
    # fig = plt.figure(4, figsize=(16,6))
    # plt.clf()
    # plt.tight_layout()
    # plt.suptitle('ao corrected psf (top) vs ao corrected coro. psf (bottom)')
    
    # grid = AxesGrid(fig, 111,
    #         nrows_ncols=(2, 5),
    #         axes_pad=0.3,
    #         cbar_mode='single',
    #         cbar_location='right',
    #         cbar_pad=0.2
    #         )
    
    # for i in range(5):
        
    #     im = grid[i].imshow(
    #         np.log10(Int_D0[iD[i],:]), vmin=vmin0, vmax=vmax0, cmap='inferno')
    #     grid[i].set_title(str(int(lam_lst[iD[i]]*1e9+.1))+'nm')
        
    #     im = grid[i+5].imshow(
    #         np.log10(Int_D[iD[i],:]), vmin=vmin0, vmax=vmax0, cmap='inferno')
    #     # grid[i+1+5].set_title('coro. psf')
        
    # # colorbar
    # cbar = grid[0].cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)
    # cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('Intensity in log scale', rotation=270)
    
    # plt.savefig(fpath_images_svg)
    # plt.savefig(fpath_images_pdf)
    
    # if i==5:
    # plt.show()
    # else : 
    # plt.close()
    # plt.close()
    
    
    #%%
    """
    plot profiles
    """
    
        
    # filepath for the direct and coronagraphic images
    fname_prf_svg = 'intensities_profiles_'+donow+'.svg'
    fname_prf_pdf = 'intensities_profiles_'+donow+'.pdf'
    fpath_prf_svg = fdir_plt / opd_set / fname_prf_svg
    fpath_prf_pdf = fdir_plt / opd_set / fname_prf_pdf
    
    # plot of the radial profiles
    colors = plt.cm.rainbow(np.linspace(0,1,nL))
    plt.figure(5, (8, 4.5))
    plt.clf()
    plt.tight_layout()
    plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
    plt.ylabel('intensity (log)')
    plt.yscale('log')
    plt.grid(True)
    
    for i in range(0,nL,2):
        
        # AO corrected coronagraphic image
        plt.plot(Int_D_prf_avg[i,0,:], Int_D_prf_avg[i,1,:],
                label=str(int(lam_lst[i]*1e9+.1))+'nm', color=colors[i])
        
    # Focal plane mask boundary
    x = np.arange(0.0, mB/2, 0.01)
    plt.axvline(as_oi, color='k', ls='--')
    plt.legend(fontsize='small', ncols=4)
    # Focal plane mask grey area
    plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lamC/rad2mas, color='gray',
                     alpha=0.3)
    plt.xlim(-0.05,np.max(rad_D_prf_avg_mas)+0.05)
    plt.ylim(1e-5, 2e0)  #  (2e-5, 2e0)

    plt.savefig(fpath_prf_svg)
    plt.savefig(fpath_prf_pdf)
        
    plt.show()

