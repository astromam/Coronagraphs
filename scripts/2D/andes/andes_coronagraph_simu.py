#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@authors: mndiaye, asimonnin, asp
"""

"""
for multiple wavelengths compute normalized psf profiles intensity for coro 
and no coro, with or without residual opds/windshake, including tilt, 
atmospheric dispersion or ncpa. the output is normalized to the peak intensity 
of the aberration free not coronagraphic (no fpm) pupil with lyot stop
"""

#%%
"""
### Initialization
"""

import numpy as np
from astropy.io import fits
from scipy import ndimage

import matplotlib.pyplot as plt

import os
import time
from pathlib import Path
from datetime import datetime  #  asp for datetime of now

from slow_fourier_transform import sft, isft
from uniform_disk import uniform_disk
from psf_profile import profile
from ncpa import ncpa

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!

# datetime of script execution
donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  datetime of now
print('date of script execution : ', donow)


#%%
"""
### various parameters
"""
# Sampling of the coronagraph focal plane mask
nFPM = 100
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)

# Image size
nImg = 400  #*1.6 #  #  even/pair!

# Pupil diameter in m 
D = 38.54

# plate scale in mas per pixel
pscale = 0.3

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

print(nL, lam_stp, lam_lst, lamC)

# conversions
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

lamCD2mas = (lamC/D)*mas2rad

# field of view
# in mas
fov_mas = nImg * pscale
# in radians
fov_rdn = fov_mas * rad2mas
# in multiple of reference lambda (lamC) over D
mD_ref = fov_rdn / ( lamC / D )


#%%
"""
### Coronagraphic components & aberrations
"""

# config tag: pupil fits filename, diam and obst in fraction of D, mB in lamc/D
configs = {'H_23':('Tel-Pupil.fits',0.9,0.38,3.8),
           'YJH_24':('ELT_pupil_400.fits',0.9,0.37,4.5),
           'HK_24':('ELT_pupil_400.fits',0.96,0.3,4.5)}

config = 'YJH_24'
diam = configs[config][1]  # lyot pupil diameter in fraction of D
obst = configs[config][2]  # lyot central obscuration in fraction of D
mB = configs[config][3]  # FPM size in lamC/D in the focal plane B
fname_elt = configs[config][0]  # input pupil fits filename

# dispersion mas/m
disp = 0.  #  e.g 8e7 ou 80e6 = 80 mas / 1e-6 m

# psf to fpm decentering in lamC/D then radians
fpm_dec = 0.
fpm_dec *= lamC/D

# ncpa rms in meters for phase screens
ncpa_rms = 0.  # e.g if 30 nm rms then 30e-9 m
# power for power law of ncpa's dsp
pwr=-2.

# lyot stop angular position error in degres
ls_ape = 0.

#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()  # wfe data
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits files
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots

elif user == 'Adrien':
    # File directory
    fdir_base = ("/Users/asimonnin/Desktop/PhD/Andes/Data_corono")
    fdir_dat = Path(fdir_base / 'data/').resolve()
    fdir_res   = Path(fdir_base / 'results/').resolve()
    fdir_plt   = Path(fdir_base / 'plots/').resolve()
elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    fdir_dat = Path( fdir_base / 'data' ).resolve()
    fdir_res   = Path( fdir_base / 'results' ).resolve()
    fdir_plt   = Path( fdir_base / 'plots' ).resolve()

# Directory for the pupils, opds; results, plots
fdir_pup = fdir_dat / 'Pupil'
fdir_opd = fdir_dat / 'OPDs_PASSATA' / 'OPD'
fdir_res = fdir_res / donow
fdir_plt = fdir_plt / donow


#%%
"""
data cubes for all psfs and intensity profiles
"""

Int_DD0 = np.zeros([nL, nImg, nImg])
Int_DD = np.zeros([nL, nImg, nImg])
Int_D0 = np.zeros([nL, nImg, nImg])
Int_D = np.zeros([nL, nImg, nImg])
Int_D0_prf_avg = np.zeros([nL, 2, nImg//2])
Int_D_prf_avg = np.zeros([nL, 2, nImg//2])


#%%
'''
pupils
'''
fpath_elt = fdir_pup / fname_elt

# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)
nPup = Pupil.shape[0]

# 2D arrays for pupil slope with tilt/dispersion
sf_x = np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5
sf_y = np.transpose(sf_x.copy())

# Lyot stop
LyotStop2d = Pupil*(uniform_disk(nPup, diam*nPup/2) -
                    uniform_disk(nPup, obst*nPup/2))
ls2d = LyotStop2d.copy()

if ls_ape != 0:
    
    pup_rot = ndimage.rotate(Pupil,ls_ape, reshape=False)
    pup_rot = pup_rot > 0.5
    LyotStop2d = pup_rot*(uniform_disk(nPup, diam*nPup/2) -
                          uniform_disk(nPup, obst*nPup/2))


#%%
'''
ncpa
'''
ncpa_d = np.zeros((nOPD*2,nPup,nPup))

if ncpa_rms!=0:
    
    ncpa_d = ncpa(ncpa_rms, nOPD*2, nPup, Pupil, pwr)


#%%
"""
### from subdirectoies name of the set of OPD files to paths of OPD data
"""
sets={'set0':('20231124_090126.0','20231122_142204.0'),
      'set1':('20240227_234849.0','20240228_053027.0','20240302_000411.0',
              '20240313_133532.0','20240228_112033.0'),
      'JQ1':('20240515_163822','20240517_091216','20240517_100705',
             '20240517_103452','20240517_105822','20240517_111658',
             '20240517_113534','20240517_121247'),
      'JQ2':('20240517_181033','20240517_183817','20240517_190418',
             '20240517_192251','20240517_194126','20240517_195959',
             '20240517_201835','20240517_203708'),
      'JQM':('20240509_182041.0','20240509_183915.0','20240509_191620.0',
             '20240509_193453.0','20240509_195327.0','20240509_201200.0',
             '20240509_203033.0','20240509_204907.0','20240509_210742.0'),
      'JQ3':('20240521_200540','20240521_181105','20240521_213115',
             '20240521_222747','20240521_210334','20240527_190439',
             '20240527_195648','20240527_204843','20240527_214043',
             '20240527_223245'),
      'JQ4':('20240528_161522','20240528_163416','20240528_165414',
             '20240528_171307','20240528_173157','20240528_175246',
             '20240528_181244','20240528_183130','20240528_185018',
             '20240528_190906'),
      'JQM_test':('20240509_201200.0',)}

opd_sets = ('JQM_test',)  # tuple, select one or more sets

roots = []
for root, subdirs, files in os.walk(fdir_opd):
    roots.append(root)

opds_dirs = []    
for o_s in range(len(opd_sets)):
    for s_i in sets.items():
        if s_i[0] == opd_sets[o_s]:
            for s_v in s_i[1][:]:
                # should be one subdir only, the last/longest path if not ...
                opds_dirs.append([s for s in roots if s_v in s][-1])

print(opds_dirs)
#%%

for dir_nb in range(len(opds_dirs)):

    opds_dir   = opds_dirs[dir_nb]

    opd_set = os.path.basename(opds_dir).split('.')[0]
    print('opd set: ', opd_set)
    
    os.makedirs(fdir_res / opd_set, exist_ok=True)
    os.makedirs(fdir_plt / opd_set, exist_ok=True)
    
    # Filename and path for the OPD maps
    flist_opd = os.listdir(opds_dir) 
    nof = len(flist_opd)
    fpath_opd = [(opds_dir+'/'+flist_opd[i]) for i in range(nof)]
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
    
    # Read OPD maps for the nOPD files
    OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])
    # convert from nm `to m if new OPD with new pupil
    OPD_arr = OPD_arr*1e-9 
    

    #%%
    
    for i in np.arange(nL):
        
        lam = lam_lst[i]
        
        dLam = lam - lamC
        tilt = dLam * disp * rad2mas
        
        # conversion lam/D to mas
        lamD2mas = (lam/D) * mas2rad
        
        mD = mD_ref * lamC / lam  #  f_o_v * ( nImg / ( lam * 1e9 ) )
        
        print('lambda (nm):' ,np.round(lam*1e9,0), '. Field of view (lam/D):',
              np.round(mD,3))
        
            
        #%%
        """
        ### Compute perfect PSF
        """
        
        # Field in the entrance pupil plane A
        Fld_AA0 = Pupil * 1.*LyotStop2d
        
        # Field in the image plane D (no coronagraph)
        Fld_DD0 = sft(Fld_AA0, nImg, mD*diam)
        
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
        Fld_BB = mask2d*sft(Fld_AA, nFPM, mB*lamC/lam)
        
        # pupil plane C before Lyot stop
        Fld_CC = Fld_AA - isft(Fld_BB, nPup, mB*lamC/lam)
        
        # # pupil plane C after Lyot stop
        Fld_LL = Fld_CC*LyotStop2d
        
        # image plane D 
        Fld_DD = sft(Fld_LL, nImg, mD*diam)
        
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
            
            # Field in the entrance pupil plane A
            Fld_A0 = (Pupil 
                      * np.exp(1j*2*np.pi *
                               (OPD_arr[iOPD] +
                                ncpa_d[iOPD] +
                                tilt * D * sf_y / nPup +
                                fpm_dec * D * sf_y / nPup)/lam)
                      * LyotStop2d)
            
            # Field in the image plane D (no coronagraph)
            Fld_D0 = sft(Fld_A0, nImg, mD*diam) ## with Lyot stop
            
            # Intensity 
            Int_D0[i,:] += np.abs(Fld_D0)**2 ## with Lyot stop
        
        t1 = time.time()       
        print(f'PSF computation time: {t1-t0:.3f}s')  
        
        # Normalized intensity
        Int_D0[i,:] /= nOPD ## with Lyot stop
        # Int_D0_2 /= flare[i][1]-flare[i][0]# without Lyot stop
        
        # Normalized intensity
        norm_peakD0 = 1/np.max(Int_D0[i,:])
        # norm_peakD0_2 = 1/np.max(Int_D0_2)
        
        Int_D0[i,:] *= norm_peakD0
        
        """
        ### Compute coronographic image (with errors)
        """
        for iOPD in range(nOPD):
            # pupil plane A
            Fld_A0 = (Pupil 
                      * np.exp(1j*2*np.pi * 
                               (OPD_arr[iOPD] +
                                ncpa_d[iOPD] +
                                tilt * D * sf_y / nPup +
                                fpm_dec * D * sf_y / nPup)/lam))
            
            # focal plane B 
            Fld_B = mask2d*sft(Fld_A0, nFPM, mB*lamC/lam)
            
            # pupil plane C before Lyot stop
            Fld_C = Fld_A0 - isft(Fld_B, nPup, mB*lamC/lam)
            
            # pupil plane C after Lyot stop
            Fld_L = Fld_C*LyotStop2d
            
            # image plane D 
            Fld_D = sft(Fld_L, nImg, mD*diam)
            
            # Intensity
            Int_D[i,:] += np.abs(Fld_D)**2    
    
        t2 = time.time()       
        
        print(f'Coro image computation time: {t2-t1:.3f}s')  
        
        # Normalized intensity
        Int_D[i,:] /= nOPD
        
        # Normalized intensity
        Int_D[i,:] *= norm_peakD0
        
        """
        ### Compute the radial intensity profiles of the images
        """
        # computation of the averaged intensity profiles of the images   
        Int_D0_prf_avg[i,1,:], rad_D0_prf_avg = profile(Int_D0[i,:], ptype='mean')
        Int_D_prf_avg[i,1,:], rad_D_prf_avg = profile(Int_D[i,:], ptype='mean')
        
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
    fname_Int_DD0 = 'wonoise_psf_'+donow+'.fits'
    fname_Int_DD = 'wonoise_coro_psf_'+donow+'.fits'
    fname_Int_D0 = 'ao_corr_psf_'+donow+'.fits'
    fname_Int_D = 'ao_corr_coro_psf_'+donow+'.fits'
    fname_Prf_D0 = 'ao_corr_psf_profile_'+donow+'.fits'
    fname_Prf_D = 'ao_corr_coro_psf_profile_'+donow+'.fits'
    
    
    # filepath for the direct and coronagraphic images
    fpath_Int_DD0 = fdir_res / fname_Int_DD0
    fpath_Int_DD  = fdir_res / fname_Int_DD
    fpath_Int_D0 = fdir_res / opd_set / fname_Int_D0
    fpath_Int_D  = fdir_res / opd_set / fname_Int_D
    fpath_Prf_D0 = fdir_res / opd_set / fname_Prf_D0
    fpath_Prf_D  = fdir_res / opd_set / fname_Prf_D
    
    
    # save the direct and coronagraphic images
    fits.writeto(fpath_Int_DD0, Int_DD0, overwrite=True)
    fits.writeto(fpath_Int_DD, Int_DD, overwrite=True)
    fits.writeto(fpath_Int_D0, Int_D0, overwrite=True)
    fits.writeto(fpath_Int_D, Int_D, overwrite=True)
    fits.writeto(fpath_Prf_D0, Int_D0_prf_avg, overwrite=True)
    fits.writeto(fpath_Prf_D, Int_D_prf_avg, overwrite=True)
    
    if os.path.isfile(fpath_Int_DD0):
        
        fpath_psf_lst=(fpath_Int_D0, fpath_Int_D, fpath_Prf_D0, fpath_Prf_D)
        
    else:
        
        fpath_psf_lst=(fpath_Int_DD0, fpath_Int_DD, fpath_Int_D0, fpath_Int_D,
                    fpath_Prf_D0, fpath_Prf_D)
    
    for fpath in fpath_psf_lst:
        fits.setval(fpath,'CCFG',value=config,comment='coro config')
        fits.setval(fpath,'PFNM',value=fname_elt,comment='elt pupil file name')
        fits.setval(fpath,'NPUP',value=nPup,comment='pupil size')
        fits.setval(fpath,'NFPM',value=nFPM,comment='FP coro. sampling')
        fits.setval(fpath,'NIMG',value=nImg,comment='image size')
        fits.setval(fpath,'NOPD',value=nOPD,comment='number of OPD files')
        fits.setval(fpath,'FOVS',value=nImg*pscale,comment='fov in mas')
        fits.setval(fpath,'LMIN',value=lam_min,comment='wavelength in meters')
        fits.setval(fpath,'LITV',value=lam_itv,comment='# of wvl intervals')
        fits.setval(fpath,'LSTP',value=lam_stp,comment='wvl step in meters')
        fits.setval(fpath,'LMBD',value=lamC,comment='reference wvl in meters')
        fits.setval(fpath,'DIAM',value=D,comment='pupil dimater in meters')
        fits.setval(fpath,'PSCL',value=pscale,comment='plate scale in mas')
        fits.setval(fpath,'SFPM',value=mB,
                    comment='FPM (in LMBD/DIAM), first focal plane')
        fits.setval(fpath,'FDIA',value=diam,comment='fractional pup. diameter')
        fits.setval(fpath,'OBST',value=obst,comment='fractional obscuration')
        fits.setval(fpath,'OPDS',value=opd_set,comment='opd set creation date')
        fits.setval(fpath,'DISP',value=disp,comment='achr. disp. in mas/m bw')
        fits.setval(fpath,'NCPA',value=ncpa_rms,comment='ncpa rms in meters')
        fits.setval(fpath,'FDEC',value=fpm_dec,
                    comment='psf to fpm offset in radians')
        fits.setval(fpath,'LSAE',value=ls_ape,
                    comment='lyot stop angular position error in degrees')
        fits.setval(fpath,'DNOW',value=donow,comment='date of script exec.')



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

