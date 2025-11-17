#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@authors: mndiaye, asimonnin, asp
"""
# for multiple wavelengths compute normalized psf profiles intensity for coro 
# and no coro, with or without residual opds/windshake including 
# tilt/atmospheric dispersion
# normalized to the peak intensity of the no coro, no opds pupil with lyot stop

#%%
"""
### Initialization
"""
import numpy as np
from scipy import ndimage
import pyzelda.utils.zernike as zernike

# pythonpath to update possibly...
import slow_fourier_transform as sft
from uniform_disk import uniform_disk
import profile as pp
from draw_vanes import six_petals #, six_arms

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid

from astropy.io import fits

import os
import time
from pathlib import Path
from datetime import datetime  #  asp for datetime of now
#  import pdb

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #  mdiaye 15!


#%%
"""
### Parameters
"""
# Sampling of the coronagraph focal plane mask
nFPM = 100

# Image size
nImg = 400  #*1.6 #  #  even/pair!

# angular separation of interest in mas
as_oi = 25.

# # of OPD phase screens
nOPD = 2000

# wavelengths in m
lamC = 1600e-9  #  some reference wvl unique value
lam_min = 960e-9  #  min value in range
lam_max = 2450e-9  #  max value in range  #  2450e-9 // 1800e-9
lam_itv = 18  #  nb of intervals in range --> nb+1 wvl's !  #  18 // 10
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

lamCD2mas = (lamC/D)*mas2rad

# field of view
# in mas
fov_mas = nImg * pscale
# in radians
fov_rdn = fov_mas * rad2mas
# in multiple of reference lambda (lamC) over D
mD_ref = fov_rdn / ( lamC / D )

"""
### Coronagraphic components
"""
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)
diam = 0.90 # diameter of the pupil in fraction of the pupil size
obst = 0.37 # diameter of the central obscuration in fraction of the pupil size
# FPM size in lam/D in the focal plane B
mB = 0. # 4.0

# dispersion mas/m
disp = 0.  #  e.g 8e7 ou 80e6 = 80 mas / 1e-6 m

# psf to fpm decentering in lamC/D then radians
# fpm_dec = 0.
# fpm_dec *= lamC/D
# psf to fpm decentering in mas then for legacy in radians
fpm_dec_mas = 0.
fpm_dec = fpm_dec_mas * rad2mas

# lyot stop angular position error in degres
ls_ape = 0.
# lyot stop vertical - elevation - offset error in pixels
ls_voe = 0
# lyot stop horizontal - azimut - offset error in pixels
ls_hoe = 0

# ncpa phase screens
ncpa_rms = 0  #  nm
# defocus at fpm in nm RMS for reference wvl
fpm_dfe_elt = 0
# fpm_dfe = 0 if fpm_dfe_elt = 0., computed dynamicaly otherwise
fpm_dfe = fpm_dfe_elt

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

# elif user == 'toto':

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
"""
ncpa's, pupils, ncpa, defocus
"""
# Filename and path for the ELT pupil  
fname_elt = 'ELT_pupil_400.fits' # New pupil with new spider
fpath_elt = fdir_pupil / fname_elt
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)
nPup = Pupil.shape[0]
Pupil = np.where(Pupil==1,1.,0.)
ipup = np.nonzero(Pupil)
nb_pup = float(np.asarray(ipup).shape[1])

petals = six_petals(nPup)
pupil_cube = petals * Pupil[None,:,:] #* arms[None,:,:]
i_p = {i:np.nonzero(pupil_cube[i,:]) for i in range(6)}

# 2D array pupil slope for tilt
sf_x = np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5
sf_y = np.transpose(sf_x.copy())

#2D array for distance to center pixel in pupil, in [0,1]
kx = (np.arange(nPup)-nPup//2)/(nPup/2)
ky = (np.arange(nPup)-nPup//2)/(nPup/2)
kx2, ky2 = np.meshgrid(kx, ky)
rho = np.sqrt(kx2**2 + ky2**2)

ncpa_d = np.zeros((nOPD,nPup,nPup))
if ncpa_rms != 0:
    print("NCPA [nm RMS]:",ncpa_rms)
    # ncpa_d = np.zeros((nOPD,nImg,nImg))
    fnm = ('ncpa_pupil_new_'+str(ncpa_rms)+'nm.fits')
    ncpa_d = fits.getdata(fdir_dat/fnm)


dfc = np.zeros((nPup,nPup))
if fpm_dfe != 0:
    
    dfc = uniform_disk(nPup,nPup//2)
    iok = np.nonzero(dfc)
    dfc[iok] = np.sqrt(3.)*(rho[iok]*rho[iok]-1.)

    dfc_temp = dfc.copy()
    dfc_temp *= Pupil.copy()
    mp = np.mean(dfc_temp[ipup])
    dfc_temp -= mp
    sp = np.std(dfc_temp[ipup])
    dfc_temp /= sp
    dfc_temp *= fpm_dfe_elt * 1e-9
    dfc -= mp
    dfc /= sp
    dfc *= fpm_dfe_elt * 1e-9
    dfc -= np.mean(dfc[iok])
    fpm_dfe = np.std(dfc[iok]) * 1e9
    dfc = dfc_temp.copy()    
    
    print('elt pupil circumcircle input defocus:', np.round(fpm_dfe,3),\
          ',\nelt pupil input defocus:', np.round(fpm_dfe_elt,3))


#%%
'''
zernike
'''

# number of Zernike modes
nZern = 7
nDim = nPup + 2

# crop the pupil to have array size equal to the pupil diameter
ini = (nDim-nPup)//2
end = (nDim+nPup)//2

"""
### Generation of the Zernike modes
"""

Zern_arr0 = np.zeros((nZern, nDim, nDim))
for i in range(nZern):
    Zern_arr0[i] = zernike.zernike1(i+2, npix=nDim)
    Zern_arr0[i, np.isnan(Zern_arr0[i])] = 0.
    Zern_arr = np.zeros((nZern, nPup, nPup))
# crop the pupil to have array size equal to the pupil diameter
Zern_arr = Zern_arr0[:, ini:end, ini:end]


#%%
'''
rotate then shift Lyot Stop and defocus
'''

# Lyot stop
LyotStop2d = Pupil*(uniform_disk(nPup, diam*nPup/2) -
                    uniform_disk(nPup, obst*nPup/2))

i_ls = np.nonzero(Pupil * LyotStop2d)
lyot_cube = petals * (Pupil * LyotStop2d)[None,:,:] #* arms[None,:,:]
i_l = {i:np.nonzero(lyot_cube[i,:]) for i in range(6)}

if ls_ape != 0:
    
    pup_rot = ndimage.rotate(Pupil,ls_ape, reshape=False)
    pup_rot = pup_rot > 0.5
    LyotStop2d = pup_rot*(uniform_disk(nPup, diam*nPup/2) -
                          uniform_disk(nPup, obst*nPup/2))

if ls_voe != 0 or ls_hoe != 0:
    
    LyotStop2d = np.roll(LyotStop2d,(ls_hoe,ls_voe),(1,0))

    
#%%
"""
### working directory of the OPD files
"""

opds_dir=('OPDs_PASSATA/OPD/WS/ASI',)


#%%
for dir_nb in range(len(opds_dir)):

    new_spider_flare = fdir_dat / opds_dir[dir_nb]
    fdir_opd   = new_spider_flare
    
    opd_set = os.path.basename(fdir_opd).split('.')[0]
    print(opd_set)
    
    # opd_set='toto'
    os.makedirs(fdir_res / opd_set, exist_ok=True)
    os.makedirs(fdir_plt / opd_set, exist_ok=True)
    
    
    #%%
    
    # Filename and path for the OPD maps
    flist_opd = sorted(os.listdir(fdir_opd),key=len)
    nof = len(flist_opd)
    fpath_opd = [fdir_opd / flist_opd[i] for i in range(nof)]
    # fpath_opd = sorted(fpath_opd)
    nW = int(np.floor((nof/nOPD)+1))
    if nW>2:
        fpath_opd = [fpath_opd[i] for i in range(1,nof,nW)]
    nOPD = len(fpath_opd)
    # nOPD = 1
    print('sample size of OPD files:', nOPD)
    

    #%%
    """
    ### Read OPD files
    """
    
    t0 = time.time()
    # Read OPD maps for the nOPD files
    OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])

    # pour jeu fichier fits unique: opds_dir=('OPDs_PASSATA/OPD/WS/ASI',)
    # OPD_arr = OPD_arr[0,:,:,500:]
    # OPD_arr = np.transpose(OPD_arr,(2,0,1))
    nOPD = OPD_arr.shape[0]
    
    t1 = time.time()
    print(f'OPD reading file time: {t1-t0:.3f}s')
    
    OPD_arr *= 1e-9 # convert OPD from nm `to m if new OPD with new pupil

    # scaling the opd manually on JQ1...
    # scl2opd = 0.5
    # OPD_arr *= scl2opd
    
    # 20250318154752 0.9
    # 20250318154814 0.8
    # 20250318154832 0.7
    # 20250319083152 0.6
    # 20250319083224 0.5
    
    strehl = np.ones(nOPD)
    aa = np.zeros(nOPD)
    bb = np.zeros(nOPD)
    opd_ref = np.zeros(nOPD)
    petal_cube = OPD_arr.copy() * 0.
    
        #%%
    
    for i in np.arange(nL):
        
        lam = lam_lst[i]
        
        dLam = lam - lamC
        tilt = dLam * disp * rad2mas
        
        # conversion lam/D to mas
        lamD2mas = (lam/D)*mas2rad
        
        # FoV in lam/D in the final image plane D
        mD = mD_ref * lamC / lam
        print('lambda (nm):' ,np.round(lam*1e9,0), '. Field of view (lam/D):',
              np.round(mD,3))
        
            
        #%%
        """
        ### Compute perfect PSF
        """
        
        # Field in the entrance pupil plane A
        Fld_AA0 = Pupil # * np.exp(0.+1j*0.) # *LyotStop2d
        
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
        Fld_BB = mask2d*sft.sft(Fld_AA, nFPM, mB*lamC/lam)
        
        # pupil plane C before Lyot stop
        Fld_CC = Fld_AA - sft.isft(Fld_BB, nPup, mB*lamC/lam)
        
        # # pupil plane C after Lyot stop
        Fld_LL = Fld_CC  # *LyotStop2d
        
        # image plane D 
        Fld_DD = sft.sft(Fld_LL, nImg, mD*diam)
        
        # Intensity
        Int_DD[i,:] = np.abs(Fld_DD)**2
        
        # Normalized intensity
        Int_DD[i,:] *= norm_peakDD0
        
        
        #%%
        """
        ### Compute PSF (with errors)
        """
        
        t0 = time.time()
                
        for iOPD in range(nOPD):
            
            opd_tmp = OPD_arr[iOPD,:,:].copy() * Pupil * LyotStop2d
            # opd_tmp[i_ls] = 0.

            if iOPD == 0:
                
                # ph_tmp = opd_tmp * 2. * np.pi / lam
                
                # print('strehl before (|<exp(i.phi)>|**2):', np.round(
                #     np.abs(np.mean(np.exp(opd_tmp[ipup]*2j*np.pi/lam)))**2.,4))
                print('strehl before (|<exp(i.phi)>|**2):', np.round(
                    np.abs(np.mean(np.exp(opd_tmp[i_ls]*2j*np.pi/lam)))**2.,4))
                
                # print('strehl before (exp(-sigma**2)):', np.round(
                #     np.exp(-np.std(opd_tmp[ipup]*2.*np.pi/lam)**2.),4))

            # for petal in range(6):
                
            #     petal_opd = np.mean(opd_tmp[i_l[petal]])
            #     # petal_opd = np.mean(opd_tmp[i_p[petal]])
            #     if petal==1:
            #         opd_ref[iOPD] = petal_opd
                    
            #     petal_cube[iOPD,:,:][i_l[petal]] = petal_opd
            #     # petal_cube[iOPD,:,:][i_p[petal]] = petal_opd

            # petal_cube[iOPD,:,:] -= opd_ref[iOPD]
            # petal_cube[iOPD,:,:] *= Pupil * LyotStop2d
            
            # continue
                
            # #     opd_tmp[i_p[petal]] -= np.mean(opd_tmp[i_p[petal]])
            #     opd_tmp[i_l[petal]] -= np.mean(opd_tmp[i_l[petal]])
                # if petal in np.arange(1,4):
                #     opd_tmp[i_l[petal]] += 1e-8
                # else:
                #     opd_tmp[i_l[petal]] -= 1e-8
                    
            aa[iOPD] = (np.sum(opd_tmp*Zern_arr[0]) /
                        np.sum(Zern_arr[0]*Zern_arr[0] * LyotStop2d * Pupil))
            bb[iOPD] = (np.sum(opd_tmp*Zern_arr[1]) /
                        np.sum(Zern_arr[1]*Zern_arr[1] * LyotStop2d * Pupil))
            
            opd_tmp -= (aa[iOPD] *
                        Zern_arr[0] / np.sum(Zern_arr[0]*Zern_arr[0]) +
                        bb[iOPD] *
                        Zern_arr[1] / np.sum(Zern_arr[1]*Zern_arr[1]))

            # opd_tmp = petal_cube[iOPD,:,:] * LyotStop2d * Pupil 
                        
            opd_tot = (ncpa_d[iOPD] + dfc + opd_tmp +
                       tilt * D * sf_y / nPup + fpm_dec * D * sf_y / nPup)
            
            # opd_tot -= np.mean(opd_tot[ipup])
            opd_tot -= np.mean(opd_tot[i_ls])
            
            # for petal in range(6):
                
            #     opd_tot[i_p[petal]] -= np.mean(opd_tot[i_p[petal]])
            
            # strehl[iOPD] = (
            #     np.exp(-np.std((opd_tot*Pupil)[ipup]*2.*np.pi/lam)**2.))
            # strehl[iOPD] = (
            #     np.abs(np.mean(np.exp(opd_tot[ipup]*2j*np.pi/lam)))**2.)
            strehl[iOPD] = (
                np.abs(np.mean(np.exp(opd_tot[i_ls]*2j*np.pi/lam)))**2.)
            
            if iOPD == 0:
                print('strehl after (|<exp(i.phi)>|**2):', np.round(
                    np.abs(np.mean(np.exp(opd_tot[i_ls]*2j*np.pi/lam)))**2.,4))
                # print('strehl after (|<exp(i.phi)>|**2):', np.round(
                #     np.abs(np.mean(np.exp(opd_tot[ipup]*2j*np.pi/lam)))**2.,4))
                
                # print('strehl after (exp(-sigma**2)):', np.round(
                #     np.exp(-np.std((opd_tot*Pupil)[ipup]*2.*np.pi/lam)**2.),4))

            # Field in the entrance pupil plane A
            Fld_A0 = Pupil * np.exp(1j*2*np.pi * opd_tot /lam) * LyotStop2d
            
            # Field in the image plane D (no coronagraph)
            Fld_D0 = sft.sft(Fld_A0, nImg, mD*diam)
            
            # Intensity 
            Int_D0[i,:] += np.abs(Fld_D0)**2 ## Avec Lyot stop
        
        t1 = time.time()       
        print(f'PSF computation time: {t1-t0:.3f}s')
        
        # if i==0:
        # if True:
            
        #     petal_movie = petal_cube.copy() * Pupil * 1e9 * LyotStop2d 
        #     # petal_movie *= 127. / np.max(np.abs(petal_cube))
                                          
        #     fits.writeto(fdir_res / opd_set / ('../petal_cube_JQM.fits'),
        #                  np.rint(petal_movie).astype(int),overwrite=1)
        
        # stop
        
        # Normalized intensity
        Int_D0[i,:] /= float(nOPD) ## Avec Lyot stop
        
        # Normalized intensity
        norm_peakD0 = 1/np.max(Int_D0[i,:])
        
        Int_D0[i,:] *= norm_peakD0
        
        """
        ### Compute coronographic image (with errors)
        """
        t0 = time.time()
        # Int_D = np.zeros((nImg, nImg))
        
        for iOPD in range(nOPD):
            # pupil plane A
            # opd_tmp = petal_cube[iOPD,:,:] * LyotStop2d * Pupil 
            # opd_tmp[i_ls] = 0.

            opd_tmp = OPD_arr[iOPD].copy() * Pupil * LyotStop2d

            # for petal in range(6):
                
            # # #     opd_tmp[i_p[petal]] -= np.mean(opd_tmp[i_p[petal]])
            #     opd_tmp[i_l[petal]] -= np.mean(opd_tmp[i_l[petal]])
                # if petal in np.arange(1,4):
                #     opd_tmp[i_l[petal]] += 1e-8
                # else:
                #     opd_tmp[i_l[petal]] -= 1e-8

            opd_tmp -= (aa[iOPD] *
                        Zern_arr[0] / np.sum(Zern_arr[0]*Zern_arr[0]) +
                        bb[iOPD] *
                        Zern_arr[1] / np.sum(Zern_arr[1]*Zern_arr[1]))
            
            opd_tot = (ncpa_d[iOPD] + dfc + opd_tmp +
                       tilt * D * sf_y / nPup + fpm_dec * D * sf_y / nPup)
            
            opd_tot -= np.mean(opd_tot[i_ls])
            # opd_tot -= np.mean(opd_tot[ipup])
            
            # for petal in range(6):
                
            #     opd_tot[i_p[petal]] -= np.mean(opd_tot[i_p[petal]])
                                
            Fld_A0 = Pupil * np.exp(1j*2*np.pi * opd_tot / lam) * LyotStop2d
                      # - Pupil*np.sqrt(strehl[iOPD]))
            
            # focal plane B 
            # Fld_B = mask2d*sft.sft(Fld_A0, nFPM, mB*lamC/lam)
            
            # pupil plane C before Lyot stop
            # Fld_C = Fld_A0 - sft.isft(Fld_B, nPup, mB*lamC/lam)
            
            # pupil plane C after Lyot stop
            # Fld_L = Fld_C*LyotStop2d
            
            # image plane D 
            # Fld_D = sft.sft(Fld_L, nImg, mD*diam)

            Fld_D = (sft.sft(
                Fld_A0 - np.sqrt(strehl[iOPD]) * Pupil * LyotStop2d,
                nImg, mD*diam))
            # Fld_D = (sft.sft(
            #     Fld_A0 - Pupil * LyotStop2d,
            #     nImg, mD*diam))
            
            # Intensity
            Int_D[i,:] += np.abs(Fld_D)**2    
    
        t1 = time.time()       
        
        print(f'Coro image computation time: {t1-t0:.3f}s')  
        
        # Normalized intensity
        Int_D[i,:] /= float(nOPD )
        
        # Normalized intensity
        # Int_D *= norm_peakDD0 #norm_peakD0 DD0 to have common scale
        Int_D[i,:] *= norm_peakD0
        
        """
        ### Compute the radial intensity profiles of the images
        """
        # computation of the averaged intensity profiles of the images   
        Int_D0_prf_avg[i,1,:], rad_D0_prf_avg = pp.radial_profile(Int_D0[i,:], ptype='mean')
        Int_D_prf_avg[i,1,:], rad_D_prf_avg = pp.radial_profile(Int_D[i,:], ptype='mean')
        
        # convert pixel scale into lam/D scale for the x-axis
        rad_D0_prf_avg_lamD = rad_D0_prf_avg * mD/nImg
        rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg
        
        # rad_D0_prf_avg_mas = rad_D0_prf_avg_lamD * lamD2mas
        Int_D0_prf_avg[i,0,:] = rad_D0_prf_avg_lamD * lamD2mas
        rad_D_prf_avg_mas = rad_D_prf_avg_lamD * lamD2mas
        Int_D_prf_avg[i,0,:] = rad_D_prf_avg_lamD * lamD2mas
    
    
    #%%
    """
    save data cubes in fits files with keywords
    """
    
    # filename for the direct and coronagraphic images and profiles
    fname_Int_D0 = 'ao_corr_psf_'+donow+'.fits'
    fname_Int_D = 'ao_corr_coro_psf_'+donow+'.fits'
    fname_Prf_D0 = 'ao_corr_psf_profile_'+donow+'.fits'
    fname_Prf_D = 'ao_corr_coro_psf_profile_'+donow+'.fits'
    
    
    # filepath for the direct and coronagraphic images
    fpath_Int_D0 = fdir_res / opd_set / fname_Int_D0
    fpath_Int_D  = fdir_res / opd_set / fname_Int_D
    fpath_Prf_D0 = fdir_res / opd_set / fname_Prf_D0
    fpath_Prf_D  = fdir_res / opd_set / fname_Prf_D
    
    
    # save the direct and coronagraphic images
    fits.writeto(fpath_Int_D0, Int_D0, overwrite=True)
    fits.writeto(fpath_Int_D, Int_D, overwrite=True)
    fits.writeto(fpath_Prf_D0, Int_D0_prf_avg, overwrite=True)
    fits.writeto(fpath_Prf_D, Int_D_prf_avg, overwrite=True)
    
    fpath_psf_lst=(fpath_Int_D0, fpath_Int_D,fpath_Prf_D0, fpath_Prf_D)
    
    for fpath in fpath_psf_lst:
        fits.setval(fpath,'NPUP',value=nPup,comment='pupil size')
        fits.setval(fpath,'NFPM',value=nFPM,comment='FP coro. sampling')
        fits.setval(fpath,'NIMG',value=nImg,comment='image size')
        fits.setval(fpath,'NOPD',value=nOPD,comment='number of OPD files')
        fits.setval(fpath,'FOVS',value=fov_mas,comment='field of view in mas')
        fits.setval(fpath,'LMIN',value=lam_min,comment='wavelength in meters')
        fits.setval(fpath,'LITV',value=lam_itv,comment='# of wvl intervals')
        fits.setval(fpath,'LSTP',value=lam_stp,comment='wvl step in meters')
        fits.setval(fpath,'LMBD',value=lamC,comment='reference wvl in meters')
        fits.setval(fpath,'DIAM',value=D,comment='pupil dimater in meters')
        fits.setval(fpath,'PSCL',value=pscale,comment='plate scale in mas')
        fits.setval(fpath,'SFPM',value=mB,
                    comment='FPM (LMBD/D), 1st focal plane')
        fits.setval(fpath,'FDIA',value=diam,comment='fractional pup. diameter')
        fits.setval(fpath,'OBST',value=obst,comment='fractional obscuration')
        fits.setval(fpath,'OPDS',value=opd_set,comment='opd set creation date')
        fits.setval(fpath,'DISP',value=disp,comment='achr. disp. in mas/m bw')
        fits.setval(fpath,'NCPA',value=ncpa_rms,comment='ncpa rms in meters')
        fits.setval(fpath,'FDEC',value=fpm_dec,
                    comment='psf to fpm offset in radians')
        fits.setval(fpath,'FTLT',value=fpm_dec_mas,
                    comment='psf to fpm offset in mas')
        fits.setval(fpath,'LSAE',value=ls_ape,
                    comment='lyot stop angular position error (degrees)')
        fits.setval(fpath,'LSVE',value=ls_voe,
                    comment='lyot stop vertical offset error (pixels)')
        fits.setval(fpath,'LSHE',value=ls_hoe,
                    comment='lyot stop horizontal offset error (pixels)')
        fits.setval(fpath,'ELT_DFOC',value=fpm_dfe_elt,
                    comment='elt pupil input defocus in nm RMS@LMBD')
        fits.setval(fpath,'DIAM_DFC',value=np.round(fpm_dfe,3),
                    comment='defoc at pupil circulcircle nm RMS@LMBD')
        # fits.setval(fpath,'SCL2OPD',value=scl2opd,comment='opd rough scaling')


    #%%
    """
    plot profiles
    """
        
    # filepath for the direct and coronagraphic images
    fname_prf_pdf = 'intensities_profiles_'+donow+'.pdf'
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
        
        plt.plot(Int_D_prf_avg[i,0,:], Int_D_prf_avg[i,1,:],
                label=str(int(lam_lst[i]*1e9+.1))+'nm', color=colors[i])
        plt.plot(Int_D0_prf_avg[i,0,:], Int_D0_prf_avg[i,1,:], color=colors[i],
                 alpha=0.5, ls='--')
                 
        # AO corrected coronagraphic image
         
    # Focal plane mask boundary
    x = np.arange(0.0, mB/2, 0.01)
    plt.axvline(as_oi, color='k', ls='--')
    plt.legend(fontsize='small', ncols=4)
    # Focal plane mask grey area
    # plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lamC/rad2mas, color='gray',
    #                  alpha=0.3)
    plt.xlim(-0.05,np.max(rad_D_prf_avg_mas)+0.05)
    plt.ylim(1e-5, 2e0)  #  (2e-5, 2e0)
    # plt.ylim(1e-40, 2e0)  #  (2e-5, 2e0)

    plt.savefig(fpath_prf_pdf)
        
    plt.show()

    # # plot of the radial profiles
    # colors = plt.cm.rainbow(np.linspace(0,1,nL))
    # plt.figure(6, (8, 4.5))
    # plt.clf()
    # plt.tight_layout()
    # plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
    # plt.ylabel('intensity (log)')
    # plt.yscale('log')
    # plt.grid(True)
    
    # for i in range(0,nL,4):
        
    #     # AO corrected coronagraphic image
    #     plt.plot(Int_D_prf_avg[i,0,:], Int_D_prf_avg[i,1,:],
    #             label=str(int(lam_lst[i]*1e9+.1))+'nm', color=colors[i])
    #     plt.plot(Int_D0_prf_avg[i,0,:], Int_D0_prf_avg[i,1,:], color=colors[i],
    #              alpha=0.5, ls='--')
         
    # # Focal plane mask boundary
    # x = np.arange(0.0, mB/2, 0.01)
    # plt.axvline(as_oi, color='k', ls='--')
    # plt.legend(fontsize='small', ncols=3)
    # # Focal plane mask grey area
    # # plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lamC/rad2mas, color='gray',
    # #                  alpha=0.3)
    # plt.xlim(-0.05,np.max(rad_D_prf_avg_mas)+0.05)
    # plt.ylim(1e-5, 2e0)  #  (2e-5, 2e0)

    # fpath_prf_pdf = (fdir_plt / opd_set /
    #                  ('intensities_profiles_YJHK_5wvl'+donow+'.pdf'))
    # plt.savefig(fpath_prf_pdf)
        
    # plt.show()


    # # plot of the radial profiles
    # colors = plt.cm.rainbow(np.linspace(0,1,nL))
    # plt.figure(6, (8, 4.5))
    # plt.clf()
    # plt.tight_layout()
    # plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
    # plt.ylabel('intensity (log)')
    # plt.yscale('log')
    # plt.grid(True)
    
    # for i in range(0,nL-8,2):
        
    #     # AO corrected coronagraphic image
    #     plt.plot(Int_D_prf_avg[i,0,:], Int_D_prf_avg[i,1,:],
    #             label=str(int(lam_lst[i]*1e9+.1))+'nm', color=colors[i])
    #     plt.plot(Int_D0_prf_avg[i,0,:], Int_D0_prf_avg[i,1,:], color=colors[i],
    #              alpha=0.5, ls='--')
         
    # # Focal plane mask boundary
    # x = np.arange(0.0, mB/2, 0.01)
    # plt.axvline(as_oi, color='k', ls='--')
    # plt.legend(fontsize='small', ncols=3)
    # # Focal plane mask grey area
    # # plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lamC/rad2mas, color='gray',
    # #                  alpha=0.3)
    # plt.xlim(-0.05,np.max(rad_D_prf_avg_mas)+0.05)
    # plt.ylim(1e-5, 2e0)  #  (2e-5, 2e0)

    # fpath_prf_pdf = (fdir_plt / opd_set /
    #                  ('intensities_profiles_YJH_6wvl'+donow+'.pdf'))
    # plt.savefig(fpath_prf_pdf)
        
    # plt.show()
