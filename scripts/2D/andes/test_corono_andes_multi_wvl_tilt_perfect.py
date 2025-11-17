#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@authors: mndiaye, asimonnin, asp
"""
# for multiple wavelengths compute normalized psf profiles intensity for coro 
# and no coro, without residual opds/windshake
# normalized to the peak intensity of the no coro, no opds pupil with lyot stop

#%%
"""
### Initialization
"""

import numpy as np
# pythonpath to update possibly...
import slow_fourier_transform as sft
from uniform_disk import uniform_disk
import psf_profile as pp

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid

from astropy.io import fits

import os
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
as_oi = 20.

# # of phase screens
nOPD = 4000

lamC = 1600e-9  #  some reference wvl unique value

# yjhk  2025-05 -->
lam_min = 950e-9  #  min value in range
lam_itv = 30
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

"""
### Coronagraphic components
"""
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)
# diam = 0.90 obs = 0.37, mB = 4. with 'ELT_pupil_400.fits' YJH 2024
diam = 0.90
obst = 0.37
mB = 4.

# dispersion mas/m
disp = 0e7  #  e.g 8e7 = 80 mas / 1e-6 m

# 2D array pupil slope for tilt
frq = (np.transpose(
    np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5))

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

# elif user == 'toto':

# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'

fdir_res = fdir_res / donow / 'perfect'
fdir_plt = fdir_plt / donow / 'perfect'

os.makedirs(fdir_res , exist_ok=True)
os.makedirs(fdir_plt , exist_ok=True)


#%%
"""
data cubes for all psfs and profiles
"""

Int_DD0 = np.zeros([nL, nImg, nImg])
Int_DD = np.zeros([nL, nImg, nImg])
Int_DD0_prf_avg = np.zeros([nL, 2, nImg//2])
Int_DD_prf_avg = np.zeros([nL, 2, nImg//2])


#%%

# Filename and path for the ELT pupil  //  'Tel-Pupil.fits' <-- OLD
fname_elt = 'ELT_pupil_400.fits' # New pupil with new spider
fpath_elt = fdir_pupil / fname_elt

"""
### Read file
"""
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)

# Lyot stop
LyotStop2d = Pupil*(
    uniform_disk(nPup, diam*nPup/2)-uniform_disk(nPup, obst*nPup/2))

#%%

for i in np.arange(nL):
    
    lam = lam_lst[i]
    
    dLam = lam - lamC
    tilt = dLam * disp * rad2mas
    
    # conversion lam/D to mas
    lamD2mas = (lam/D)*mas2rad
    
    # FoV in lam/D in the final image plane D
    # mD = hlf_fov*(nImg/(lam*1e9))
    mD = mD_ref * lamC / lam

    print('lambda (nm):' ,np.round(lam*1e9,0), '. Field of view (lam/D):',
          np.round(mD,3))
    
        
    #%%
    """
    ### Compute perfect PSF
    """
    
    # Field in the entrance pupil plane A
    Fld_AA0 = Pupil * 1. * LyotStop2d
    
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
    Int_DD0_prf_avg[i,1,:], rad_DD0_prf_avg = pp.radial_profile(Int_DD0[i,:], ptype='mean')
    Int_DD_prf_avg[i,1,:], rad_DD_prf_avg = pp.radial_profile(Int_DD[i,:], ptype='mean')
       
    """
    ### Compute the radial intensity profiles of the images
    """

    # convert pixel scale into lam/D scale for the x-axis
    rad_DD0_prf_avg_lamD = rad_DD0_prf_avg * mD/nImg
    rad_DD_prf_avg_lamD = rad_DD_prf_avg * mD/nImg
    
    # rad_D0_prf_avg_mas = rad_D0_prf_avg_lamD * lamD2mas
    Int_DD0_prf_avg[i,0,:] = rad_DD0_prf_avg_lamD * lamD2mas
    rad_DD_prf_avg_mas = rad_DD_prf_avg_lamD * lamD2mas
    Int_DD_prf_avg[i,0,:] = rad_DD_prf_avg_lamD * lamD2mas


#%%
"""
save data cubes in fits files with keywords
"""

# filename for the direct and coronagraphic images and profiles
fname_Int_DD0 = 'wonoise_psf_'+donow+'.fits'
fname_Int_DD = 'wonoise_coro_psf_'+donow+'.fits'


# filepath for the direct and coronagraphic images
fpath_Int_DD0 = fdir_res / fname_Int_DD0
fpath_Int_DD  = fdir_res / fname_Int_DD

# save the direct and coronagraphic images
fits.writeto(fpath_Int_DD0, Int_DD0, overwrite=True)
fits.writeto(fpath_Int_DD, Int_DD, overwrite=True)

fpath_psf_lst=(fpath_Int_DD0, fpath_Int_DD)

for fpath in fpath_psf_lst:
    fits.setval(fpath,'NPUP',value=nPup,comment='pupil size')
    fits.setval(fpath,'NFPM',value=nFPM,comment='FP coro. sampling')
    fits.setval(fpath,'NIMG',value=nImg,comment='image size')
    fits.setval(fpath,'NOPD',value=nOPD,comment='number of OPD files')
    fits.setval(fpath,'FOVS',value=fov_mas,comment='field of view in as')
    fits.setval(fpath,'LMIN',value=lam_min,comment='wavelength in meters')
    fits.setval(fpath,'LITV',value=lam_itv,comment='# of wvl intervals')
    fits.setval(fpath,'LSTP',value=lam_stp,comment='wvl step in meters')
    fits.setval(fpath,'LMBD',value=lamC,comment='reference wvl in meters')
    fits.setval(fpath,'DIAM',value=D,comment='pupil dimater in meters')
    fits.setval(fpath,'PSCL',value=pscale,comment='plate scale in mas')
    fits.setval(fpath,'SFPM',value=mB,comment='FPM (lam/D), first focal plane')
    fits.setval(fpath,'FDIA',value=diam,comment='fractional pup. diameter')
    fits.setval(fpath,'OBST',value=obst,comment='fractional obscuration')
    fits.setval(fpath,'DISP',value=0.,comment='achr. disp. in mas/m bw')
    fits.setval(fpath,'NCPA',value=0.,comment='ncpa rms in meters')
    fits.setval(fpath,'FDEC',value=0.,comment='psf to fpm offset in radians')
    fits.setval(fpath,'OPDS',value='',comment='opd set creation date')
    fits.setval(fpath,'DISP',value=disp,comment='achr. disp. in mas/m bw')
    fits.setval(fpath,'NCPA',value=0,comment='ncpa rms in meters')
    fits.setval(fpath,'FDEC',value=0,
                comment='psf to fpm offset in radians')
    fits.setval(fpath,'FTLT',value=0,
                comment='psf to fpm offset in mas')
    fits.setval(fpath,'LSAE',value=0,
                comment='lyot stop angular position error in degrees')
    fits.setval(fpath,'LSVE',value=0,
                comment='lyot stop vertical offset error in pixels')
    fits.setval(fpath,'LSHE',value=0,
                comment='lyot stop horizontal offset error in pixels')
    fits.setval(fpath,'ELT_DFOC',value=0,
                comment='defocus at elt pupil in nm RMS at reference wvl')
    fits.setval(fpath,'DIAM_DFC',value=np.round(0,3),
                comment='pup. circumcirc. input defocus nm RMS ref. wvl')
    fits.setval(fpath,'EPUP_FNM',value=fname_elt,
            comment='ELT pupil filename')
    fits.setval(fpath,'DATE_NOW',value=donow,
            comment='daye of now, i.e. script execution')



#%%
"""
plot profiles
"""

fname_prf_svg = 'intensities_profiles_wo_noise'+donow+'.svg'
fname_prf_pdf = 'intensities_profiles_wo_noise'+donow+'.pdf'
fpath_prf_svg = fdir_plt / fname_prf_svg
fpath_prf_pdf = fdir_plt / fname_prf_pdf

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
    plt.plot(Int_DD_prf_avg[i,0,:], Int_DD_prf_avg[i,1,:],
            label=str(int(lam_lst[i]*1e9+.1))+'nm', color=colors[i])
    
# Focal plane mask boundary
x = np.arange(0.0, mB/2, 0.01)
plt.axvline(as_oi, color='k', ls='--')
plt.legend(fontsize='small', ncols=4)
# Focal plane mask grey area
plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lamC/rad2mas, color='gray',
                 alpha=0.3)
plt.xlim(-0.05,np.max(rad_DD_prf_avg_mas)+0.05)
# plt.ylim(1e-5, 2e0)  #  (2e-5, 2e0)

plt.savefig(fpath_prf_svg)
plt.savefig(fpath_prf_pdf)
    
plt.show()

print('date of now : ', donow)

