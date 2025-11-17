# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:14:11 2024

@author: asp
"""

# visualize outputs of explore_multi_wvl_contrast_andes_coro_params.py

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import os
from uniform_disk import uniform_disk
from draw_vanes import six_arms

        
plt.rcParams.update({'font.size': 12})

# conversion radian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# angular separation radius of interest in mas
as_oi = 20.

fdir_dat = Path("d:/Andes/Data_corono/results/exploreCoroParamsYJHK_25_30_35_mas_lbdCtoDwidth").resolve()

files=['20240923175955/Parameters_results_contrast_w_coro_no_turb_Y.fits',
       '20240924100841/Parameters_results_contrast_w_coro_no_turb_J.fits',
       '20240925112915/Parameters_results_contrast_w_coro_no_turb_H.fits',
       '20240926114503/Parameters_results_contrast_w_coro_no_turb_JH.fits',
       '20240927090738/Parameters_results_contrast_w_coro_no_turb_YJH.fits']
files=['20240927090738/Parameters_results_contrast_w_coro_no_turb_YJH.fits']

# Directory for the pupils
fdir_pupil = fdir_dat / '../../data/Pupil'

# Filename and path for the ELT pupil
fname_elt = 'ELT_pupil_400.fits'
fpath_elt = fdir_pupil / fname_elt

"""
### Read file
"""
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)
nPup = Pupil.shape[0]
# lyot mask vane width in pixels
v_width = 3.  #  3. pour fichier ELT_pupil_400.fits non modifie
vanes = six_arms(nPup, v_width)

#%%

all_contrast_range = {}
allThrVsInt = {}

for i in range(len(files)):

    fnm = Path(fdir_dat / files[i] )
    donow = files[i].split('/')[0]
    wvl = (fdir_dat / files[i]).stem.split('_')[-1]
    
    print(donow, wvl)

    coro_data = fits.getdata(fnm)

    head_data = fits.getheader(fnm)
    lam = head_data['LMBD']
    nImg = head_data['NIMG']
    
    mB_min = head_data['SFPM_MIN']
    mB_max = head_data['SFPM_MAX']
    mB_stp = head_data['SFPM_STP']
    mB = np.arange(mB_min,mB_max+mB_stp,mB_stp)
    
    obs_min = head_data['OBST_MIN']
    obs_max = head_data['OBST_MAX']
    obs_stp = head_data['OBST_STP']
    obs = np.arange(obs_min,obs_max+obs_stp,obs_stp)
    
    dL_min = head_data['DLYO_MIN']
    dL_max = head_data['DLYO_MAX']
    dL_stp = head_data['DLYO_STP']
    dL = np.arange(dL_min,dL_max+dL_stp,dL_stp)
    
    v_width = head_data['LYO_VW']
    D = head_data['DIAM']
   
    lamD2mas = (lam/D)*mas2rad
    # FoV in lam/D in the final image plane D
    mD = 58.393*(nImg/(lam*1e9))
    i_as_oi = np.argmin(abs(np.arange(nImg//2) * mD * lamD2mas / nImg - as_oi))

    coro_data = coro_data[1,:,:,:,i_as_oi]
    data_shape = coro_data.shape
    
    i_min = np.unravel_index(np.argmin(coro_data), data_shape)
    cd_min = np.log10(np.min(coro_data))
    cd_max = np.log10(np.max(coro_data))

    thr_cube = coro_data.copy()
    
    for d in range(len(dL)): 
        diam = dL[d]

        for ob in range(len(obs)):
            obst=obs[ob]

            LyotStop2d = (Pupil.copy() * vanes *
                          (uniform_disk(nPup, diam*nPup/2) -
                           uniform_disk(nPup, obst*nPup/2)))
            
            for s in range(len(mB)) :
                
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
    
                thr_cube[d,ob,s] = ee_c/ee_a

    thr_data = thr_cube[:,:,0]

# %%
    
    save_dir = ("d:/Andes/Data_corono/plots/"+donow)
    os.makedirs(save_dir, exist_ok=True)
    
    thrVsInt = []
    thr = 0.
    b2g = 0.
    # thr_cube = thr_cube[:,:,0:9]
    # coro_data = coro_data[:,:,0:9]
    while True:
        
        ma_data = np.ma.where(thr_cube > thr, coro_data.copy(), 1.)
        if ma_data.mean() == 1.:
            break
        
        i_min = np.unravel_index(np.argmin(ma_data), data_shape)
        cd_min = np.log10(np.min(ma_data))
        cd_max = np.log10(np.max(ma_data[ma_data<1.]))
        
        ma_tp_data = np.ma.masked_where(thr_data<thr,thr_data.copy(),1.)
            
        print("\t", np.round(thr,3), "\t",
              np.round(thr_data[i_min[0],i_min[1]],3), "\t",
              np.round(np.min(ma_data),9), "\t",
              np.round(np.min(ma_data)/thr_data[i_min[0],i_min[1]],9), "\t",
              np.round([dL[i_min[0]],obs[i_min[1]],mB[i_min[2]]],2))

        thr = thr_data[i_min[0],i_min[1]].copy()
        thrVsInt.append([[np.round(thr_data[i_min[0],i_min[1]],3)],
                         [np.round(np.min(ma_data),9)],
                         [np.round(dL[i_min[0]],2)],
                         [np.round(obs[i_min[1]],2)],
                         [np.round(mB[i_min[2]],2)]])

    thrVsInt = np.array(thrVsInt)
    plt.figure(8, (8,5))
    bad = np.argwhere(thrVsInt[:,0,0]<b2g)
    good = np.argwhere(thrVsInt[:,0,0]>=b2g) 
    plt.plot(thrVsInt[bad,0,0], thrVsInt[bad,1,0],color='r')
    plt.plot(thrVsInt[good,0,0], thrVsInt[good,1,0])
    plt.xlabel("throughput")
    plt.ylabel("contrast @ " + str(int(as_oi)) + " mas ang. sep.")
    plt.title("band: "+wvl+", wvl: " + str(int(lam*1e9)) + " nm")
    fnm = ("/throughputVsContrast_allThresholds_"+wvl+"_"+str(int(lam*1e9)) +
           "nm_radMas"+str(int(np.round(as_oi))))
    plt.savefig(save_dir+fnm+".pdf")
    plt.savefig(save_dir+fnm+".svg")
    fits.writeto(Path(fdir_dat / donow / (fnm+'.fits')), thrVsInt,
                 header=head_data, overwrite=True)
    plt.show()
    
    all_contrast_range[wvl] = (
        (np.round(np.max(thrVsInt[good,1,0])/np.min(thrVsInt[good,1,0]),2),
         np.median(thrVsInt[good,1,0])))
    
    allThrVsInt[wvl] = thrVsInt[good,:,0]
    

for v, r in all_contrast_range.items():
    print(f'{v} {r}')


plt.figure(9)
plt.xlabel("throughput")
plt.ylabel("contrast @ " + str(int(as_oi)) + " mas ang. sep.")
for v, a in allThrVsInt.items():
    plt.plot(a[:,0,0],a[:,0,1],label=v)
plt.legend(fontsize='x-small', ncol=2)
    
plt.figure(10)
plt.xlabel("Lyot diameter (to D)")
plt.ylabel("contrast @ " + str(int(as_oi)) + " mas ang. sep.")
for v, a in allThrVsInt.items():
    plt.plot(a[:,0,2],a[:,0,1],label=v)
plt.legend(fontsize='x-small', ncol=2)

plt.figure(11)
plt.xlabel("Lyot obstruction (to D)")
plt.ylabel("contrast @ " + str(int(as_oi)) + " mas ang. sep.")
for v, a in allThrVsInt.items():
    plt.plot(a[:,0,3],a[:,0,1],label=v)
plt.legend(fontsize='x-small', ncol=2)

plt.figure(12)
plt.xlabel('FPM scale (in $\lambda$/D)')
plt.ylabel("contrast @ " + str(int(as_oi)) + " mas ang. sep.")
for v, a in allThrVsInt.items():
    plt.plot(a[:,0,4],a[:,0,1],label=v)
plt.legend(fontsize='x-small', ncol=2)

