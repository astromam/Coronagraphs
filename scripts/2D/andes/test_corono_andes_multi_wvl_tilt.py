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

# pythonpath to update possibly...
import slow_fourier_transform as sft
# import cupy_slow_fourier_transform as sft
from uniform_disk import uniform_disk
import psf_profile as pp

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

# Sampling of the coronagraph focal plane mask
nFPM = 100

# "field of view" in mas, not the real fov but some legacy nIMG*pscale/2...
# f_o_v = 58.393

# Image size
nImg = 400  #*1.6 #  #  even/pair!

# angular separation of interest in mas
as_oi = 20.

# # of OPD phase screens
nOPD = 4000

# wavelengths in m
lamC = 1600e-9  #  some reference wvl unique value

# yjhk  2025-05 -->
lam_min = 950e-9  #  min value in range
lam_itv = 18  #◘ 30 if + K band
lam_stp = 50e-9

lam_lst = np.arange(lam_min,lam_min+(lam_itv+0.5)*lam_stp,lam_stp)
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
# mD = fov_rdn * D / lam_lst

"""
### Coronagraphic components
"""
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)
# diam = 0.89 obs = 0.35, mB = 3.9 2025 with 'ELT_pupil_400.fits' YJH
# diam = 0.90 obs = 0.37, mB = 4.0 2024 with 'ELT_pupil_400.fits' YJH @ 25 mas / 75% thr
# diam = 0.96 obs = 0.30, mB = 4.5 2024 with 'ELT_pupil_400.fits' K
# diam = 0.90 obs = 0.38 with 'Tel-Pupil.fits'

diam = 0.89 # diameter of the pupil in fraction of the pupil size
obst = 0.35 # diameter of the central obscuration in fraction of the pupil size
mB = 3.9    # FPM size in lam/D in the focal plane B

# dispersion mas/m
disp = 5e6  #  e.g 8e7 ou 80e6 = 80 mas / 1e-6 m
# psf to fpm decentering in mas then for legacy in radians
fpm_dec_mas = 2
fpm_dec = fpm_dec_mas * rad2mas
# lyot stop angular position error in degres
ls_ape = 1
# lyot stop vertical - elevation - offset error in pixels
ls_voe = 0
# lyot stop horizontal - azimut - offset error in pixels
ls_hoe = 4
# ncpa phase screens
ncpa_rms = 30  #  nm
# defocus at fpm in nm RMS for reference wvl
fpm_dfe_elt = 30
# fpm_dfe = 0 if fpm_dfe_elt = 0., computed dynamicaly otherwise
fpm_dfe = fpm_dfe_elt * -1.

# compute & keep elt psf & profiles
simu_elt = False

# datetime of script execution
donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now

print('\n', diam, obst, mB)
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

#  elif user == 'toto':

# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'

fdir_res = fdir_res / donow
fdir_prfct = fdir_res / 'perfect'
os.makedirs(fdir_prfct, exist_ok=True)

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
Int_elt = np.zeros([nL, nImg, nImg])


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
ipup = np.nonzero(Pupil)

# 2D array pupil slope for tilt
sf_x = np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5
sf_y = np.transpose(sf_x.copy())

#2D array for distance to center pixel in pupil, in [0,1]
kx = (np.arange(nPup)-nPup//2)/(nPup/2)
ky = (np.arange(nPup)-nPup//2)/(nPup/2)
kx2, ky2 = np.meshgrid(kx, ky)
rho = np.sqrt(kx2**2 + ky2**2)

if ncpa_rms != 0:
    
    print("NCPA [nm RMS]:",ncpa_rms)
    
    fnm = ('ncpa_unscaled_x2_ELT_pupil_400.fits')
    ncpa_1 = fits.getdata(fdir_dat/fnm)
    nMap = ncpa_1.shape[0]
    
    N=nMap//2
    hlf=N//2
    
dfc = np.zeros((nPup,nPup))
if fpm_dfe_elt > 0:
    
    # create circular pupil (elt circumcircle) phase with defocus
    dfc = uniform_disk(nPup,nPup//2)
    iok = np.nonzero(dfc)
    dfc[iok] = np.sqrt(3.)*(rho[iok]*rho[iok]-1.)

    # scale defocus inside elt pupil to required value
    dfc_temp = dfc.copy()
    dfc_temp *= Pupil.copy()
    mp = np.mean(dfc_temp[ipup])
    dfc_temp -= mp
    sp = np.std(dfc_temp[ipup])
    dfc_temp /= sp
    dfc_temp *= fpm_dfe_elt * 1e-9

    # compute input defocus over elt circum circle in front of elt pupil
    dfc -= mp
    dfc /= sp
    dfc *= fpm_dfe_elt * 1e-9
    dfc -= np.mean(dfc[iok])
    fpm_dfe = np.std(dfc[iok]) * 1e9
    
    # update defocus phase screen to elt pupil shape with the required value
    dfc = dfc_temp.copy()    
    
    print('elt pupil circumcircle input defocus:', np.round(fpm_dfe,3),\
          ',\nelt pupil input defocus:', np.round(fpm_dfe_elt,3))


#%%
'''
rotate then shift Lyot Stop and defocus
'''
# Lyot stop
LyotStop2d = Pupil*(uniform_disk(nPup, diam*nPup/2) -
                    uniform_disk(nPup, obst*nPup/2))

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
#%%
# root = 'OPDs_PASSATA/OPD/WS/500HzVarWS/'
# opds_dir=(root+'20250704_105833.0',
#           root+'20250704_112007.0',
#           root+'20250704_114101.0',
#           root+'20250704_121550.0',
#           root+'20250704_123540.0',
#           root+'20250704_125507.0',
#           root+'20250704_131433.0',
#           root+'20250704_135325.0',
#           root+'20250704_141253.0',
#           root+'20250707_114321.0')

root = 'OPDs_PASSATA/OPD/WS/1kHzVarWS/'
opds_dir=(root+'1',
          root+'2',
          root+'3',
          root+'4',
          root+'5',
          root+'6',
          root+'7',
          root+'8',
          root+'9',
          root+'10')


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
    flist_opd = []
    for fnm in os.listdir(fdir_opd):
        if fnm.endswith(".fits"):
            flist_opd.append(fnm)
    
    flist_opd = sorted(flist_opd,key=len)
    # flist_opd = sorted(os.listdir(fdir_opd),key=len) 
    nof = len(flist_opd)
    print("# of files in phase screens dir:", nof)
    fpath_opd = [fdir_opd / flist_opd[i] for i in range(nof)]
    # fpath_opd = sorted(fpath_opd)
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

    # pour jeu fichier fits unique, e.g: OPDs_PASSATA/OPD/WS/ASI_*
    if nof < 2:
        start = int((OPD_arr.shape)[3]/5)
        print("skip:", start)
        OPD_arr = OPD_arr[0,:,:,start:]
        OPD_arr = np.transpose(OPD_arr,(2,0,1))
        nOPD = OPD_arr.shape[0]
        print("phase screen array shape from cube file: ", OPD_arr.shape)
    else:
        print("phase screen array shape from set of files: ", OPD_arr.shape)

    t1 = time.time()
    print(f'OPD reading file time: {t1-t0:.3f}s')
    
    OPD_arr *= 1e-9 # convert OPD from nm to m if new OPD with new pupil
    
    if ncpa_rms != 0:
         
        # fnm = ('ncpa_ELT_pupil_400_30nm_4096screens.fits')
        # ncpa_1 = fits.getdata(fdir_dat/fnm)        
        rnd=np.random.randn(nOPD)
        rnd /= 2.
        xi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)
        
        rnd=np.random.randn(nOPD)
        rnd /= 2.
        yi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)

        for n in range(nOPD):
            
            temp = ((ncpa_1[hlf+xi[n]:hlf+N+xi[n],hlf+yi[n]:hlf+N+yi[n]]).copy() *
                    Pupil.copy())
            
            temp -= np.mean(temp[ipup])
            temp /= np.std(temp[ipup])
            temp *= float(ncpa_rms) * 1e-9
            OPD_arr[n,:,:] += temp.copy() * Pupil.copy()
            temp *= 0.
            # OPD_arr[n,:,:] += ncpa_1[n,:,:] * Pupil.copy()

    
        #%%
    
    for i in np.arange(nL):
        
        lam = lam_lst[i]
        
        dLam = lam - lamC
        tilt = dLam * disp * rad2mas
        
        # conversion lam/D to mas
        lamD2mas = (lam/D)*mas2rad
        
        # FoV in lam/D in the final image plane D
        # mD = f_o_v*(nImg/(lam*1e9))
        mD = mD_ref * lamC / lam
        
        print('lambda (nm):' ,np.round(lam*1e9,0), '. Field of view (lam/D):',
              np.round(mD,3))
        
            
        #%%
        """
        ### Compute perfect PSF
        """
        
        # Field in the entrance pupil plane A
        Fld_AA0 = Pupil * 1.* LyotStop2d
        
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
        Fld_LL = Fld_CC * LyotStop2d
        
        # image plane D 
        Fld_DD = sft.sft(Fld_LL, nImg, mD*diam)
        
        # Intensity
        Int_DD[i,:] = np.abs(Fld_DD)**2
        
        # Normalized intensity
        Int_DD[i,:] *= norm_peakDD0
        
        
    #%%    
        # computation of the averaged intensity profiles of the images  
        Int_DD0_prf_avg, rad_DD0_prf_avg = pp.radial_profile(Int_DD0[i,:], ptype='mean')
        Int_DD_prf_avg, rad_DD_prf_avg = pp.radial_profile(Int_DD[i,:], ptype='mean')
        
        if simu_elt:
            Fld_elt = sft.sft(Pupil*1., nImg, mD*diam)
            Int_elt[i,:] = np.abs(Fld_elt)**2
            Int_elt[i,:] *= norm_peakDD0

        
        #%%
        """
        ### Compute PSF (with errors)
        """
        
        t0 = time.time()
        
        for iOPD in range(nOPD):
            
            # Field in the entrance pupil plane A
            Fld_A0 = (Pupil 
                      * np.exp(1j*2*np.pi *
                               (OPD_arr[iOPD] +
                                # ncpa_d[iOPD] +
                                tilt * D * diam * sf_y / nPup +
                                fpm_dec * D * diam * sf_y / nPup +
                                dfc)/lam)
                      * LyotStop2d)
            
            # Field in the image plane D (no coronagraph)
            Fld_D0 = sft.sft(Fld_A0, nImg, mD*diam) ## Avec Lyot stop
            
            # Intensity 
            Int_D0[i,:] += np.abs(Fld_D0)**2 ## Avec Lyot stop
        
        t1 = time.time()       
        print(f'PSF computation time: {t1-t0:.3f}s')  
        
        # Normalized intensity
        Int_D0[i,:] /= nOPD ## Avec Lyot stop
        # Int_D0_2 /= flare[i][1]-flare[i][0]# Sans Lyot stop
        
        # Normalized intensity
        norm_peakD0 = 1/np.max(Int_D0[i,:])
        # norm_peakD0_2 = 1/np.max(Int_D0_2)
        
        Int_D0[i,:] *= norm_peakD0
        
        """
        ### Compute coronographic image (with errors)
        """
        t0 = time.time()
        # Int_D = np.zeros((nImg, nImg))
        
        for iOPD in range(nOPD):
            # pupil plane A
            Fld_A0 = (Pupil 
                      * np.exp(1j*2*np.pi * 
                               (OPD_arr[iOPD] +
                                # ncpa_d[iOPD] +
                                tilt * D * diam * sf_y / nPup +
                                fpm_dec * D * diam * sf_y / nPup +
                                dfc)/lam))
            
            # focal plane B 
            Fld_B = mask2d*sft.sft(Fld_A0, nFPM, mB*lamC/lam)
            
            # pupil plane C before Lyot stop
            Fld_C = Fld_A0 - sft.isft(Fld_B, nPup, mB*lamC/lam)
            
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
    
    fpath_psf_lst=(fpath_Int_D0, fpath_Int_D, fpath_Prf_D0, fpath_Prf_D)
                
    if len(os.listdir(fdir_prfct))==0:
        fname_Int_DD0 = 'wonoise_psf_'+donow+'.fits'
        fname_Int_DD = 'wonoise_coro_psf_'+donow+'.fits'
        fpath_Int_DD0 = fdir_prfct/ fname_Int_DD0
        fpath_Int_DD  = fdir_prfct / fname_Int_DD

        if not os.path.isfile(fpath_Int_DD0):
            fits.writeto(fpath_Int_DD0, Int_DD0, overwrite=True)
            fpath_psf_lst += (fpath_Int_DD0,)
            
        if not os.path.isfile(fpath_Int_DD):
            fits.writeto(fpath_Int_DD, Int_DD, overwrite=True)
            fpath_psf_lst += (fpath_Int_DD,)
    
        if simu_elt:
            fname_elt = 'wo_noise_psf_elt_pupil.fits'
            fpath_elt = fdir_prfct / fname_elt
            if not os.path.isfile(fpath_elt):
                fits.writeto(fpath_elt, Int_elt, overwrite=True)
                fpath_psf_lst += (fpath_elt,)

        hdr_keys = {'NPUP':(nPup,'pupil size'),
                'NFPM':(nFPM,'FP coro. sampling'),
                'NIMG':(nImg,'image size'),
                'NOPD':(nOPD,'number of OPD files'),
                'FOVS':(fov_mas,'field of view in mas'),
                'LMIN':(lam_min,'wavelength in meters'),
                'LITV':(lam_itv,'# of wvl intervals'),
                'LSTP':(lam_stp,'wvl step in meters'),
                'LMBD':(lamC,'reference wvl in meters'),
                'DIAM':(D,'pupil dimater in meters'),
                'PSCL':(pscale,'plate scale in mas'),
                'SFPM':(mB,'FPM (LMBD/D), first focal plane'),
                'FDIA':(diam,'fractional pup. diameter'),
                'OBST':(obst,'fractional obscuration'),
                'OPDS':(opd_set,'opd set creation date'),
                'DISP':(disp,'achr. disp. in mas/m bw'),
                'NCPA':(ncpa_rms,'ncpa rms in meters'),
                'FDEC':(fpm_dec,'psf to fpm offset in radians'),
                'FTLT':(fpm_dec_mas,'psf to fpm offset in mas'),
                'LSAE':(ls_ape,'lyot stop angular position error in degrees'),
                'LSVE':(ls_voe,'lyot stop vertical offset error in pixels'),
                'LSHE':(ls_hoe,'lyot stop horizontal offset error in pixels'),
                'ELT_DFOC':(fpm_dfe_elt,'elt pupil defocus in nm RMS at LMBD'),
                'DIAM_DFC':(fpm_dfe,'pup. circumcirc. defocus nm RMS at LMBD'),
                'EPUP_FNM':(fname_elt,'ELT pupil filename'),
                'DATE_NOW':(donow,'date of now, i.e. script execution date')}
    
    for fpath in fpath_psf_lst:
        for n, k in enumerate(hdr_keys):
            fits.setval(fpath,k,value=hdr_keys[k][0],comment=hdr_keys[k][1])


    #%%
    """
    display images
    # """
        
    # # filename of the plot
    # fname_images_pdf = 'ao_corrected_coro_psf_'+donow+'.pdf'
    
    # # filepath for the direct and coronagraphic images
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

    plt.savefig(fpath_prf_pdf)
        
    plt.show()

print(diam,obst,mB)
print('date of now : ', donow)

