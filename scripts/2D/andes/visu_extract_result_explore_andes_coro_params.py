# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:58:14 2024

@author: asp
"""

# plot coro params versus contrats in 4 bands ans 3 angular separations
# from outputs of explore_multi_wvl_andes_coro_parameters.py
# print best parameters for each case

import matplotlib.pyplot as plt
import numpy as np
# import os
from astropy.io import fits
from pathlib import Path


#%%

thr = 0.75

# from path_recursive_file_search import recursive_search
# fdir_dat = Path(
#     "d:/Andes/Data_corono/results/exploreCoroParamsYJHK_25_30_35_mas_lbdCtoDwidth/").resolve()
fdir_dat = Path("d:/Andes/Data_corono/results/").resolve()

# file_list=recursive_search(".")
# fnm = '20240801100039/Parameters_throughput_Y.fits'
fnm = '20240917115845/Parameters_throughput_H.fits'

throughput = fits.getdata( fdir_dat/fnm )

head_thr = fits.getheader(fdir_dat/fnm)
obs_min = head_thr['OBST_MIN']
obs_max = head_thr['OBST_MAX']
dL_min = head_thr['DLYO_MIN']
dL_max = head_thr['DLYO_MAX']

ok = throughput >= thr
iko = np.argwhere(~ok)

plt.figure(9)
plt.imshow((throughput[:,:]),
            vmin=np.min((throughput)),
            vmax=np.max((throughput)),
            extent=(obs_min,obs_max,dL_max,dL_min), cmap='gray')

plt.colorbar()
plt.xlabel("obstruction")
plt.ylabel("Lyot stop diam.")
plt.show()

plt.figure(10)
plt.imshow((throughput[:,:]*ok),
            vmin=np.min((throughput)),
            vmax=np.max((throughput)),
            extent=(obs_min,obs_max,dL_max,dL_min), cmap='gray')

plt.colorbar()
plt.xlabel("obstruction")
plt.ylabel("Lyot stop diam.")
plt.show()


#%%

files=['20240923175955/Parameters_results_contrast_w_coro_no_turb_Y.fits',
       '20240924100841/Parameters_results_contrast_w_coro_no_turb_J.fits',
       '20240925112915/Parameters_results_contrast_w_coro_no_turb_H.fits',
       '20240924115352/Parameters_results_contrast_w_coro_no_turb_K.fits',
       '20240926114503/Parameters_results_contrast_w_coro_no_turb_JH.fits',
       '20240926154034/Parameters_results_contrast_w_coro_no_turb_HK.fits',
       '20240927090738/Parameters_results_contrast_w_coro_no_turb_YJH.fits',
       '20240927105903/Parameters_results_contrast_w_coro_no_turb_YJHK.fits']

for f in np.arange(len(files)):
    
    parameters_results_contrast = fits.getdata(fdir_dat / files[f])
    head_res = fits.getheader(fdir_dat / files[f])
    
    D = head_res['DIAM']
    rad2mas = np.pi/(180.*3600*1000)
    mas2rad = 1/rad2mas
    nImg = head_res['NIMG']
    aS = np.arange(nImg//2) * (mas2rad * 58.393 / (D * 1e9) )
    i25 = np.argmin(np.abs(aS-25.))
    i30 = np.argmin(np.abs(aS-30.))
    i35 = np.argmin(np.abs(aS-35.))
    
    mB_min = head_res['SFPM_MIN']
    mB_max = head_res['SFPM_MAX']
    mB_stp = head_res['SFPM_STP'] 
    mB = np.round(np.arange(mB_min,mB_max+mB_stp,mB_stp),3)

    obs_min = head_res['OBST_MIN']
    obs_max = head_res['OBST_MAX']
    obs_stp = head_res['OBST_STP'] 
    obs = np.round(np.arange(obs_min,obs_max+obs_stp,obs_stp),3)

    dL_min = head_res['DLYO_MIN']
    dL_max = head_res['DLYO_MAX']
    dL_stp = head_res['DLYO_STP'] 
    dL = np.round(np.arange(dL_min,dL_max+dL_stp,dL_stp),3)

    data25 = parameters_results_contrast[1,:,:,:,i25]
    data30 = parameters_results_contrast[1,:,:,:,i30]
    data35 = parameters_results_contrast[1,:,:,:,i35]
    data = [data25,data30,data35]
    ang_sep = ['25','30','35']
    band = (fdir_dat / files[f]).stem[-1]

    lamC = head_res['LMBD']
    
    for i in np.arange(iko.shape[0]):
        data25[iko[i][0],iko[i][1],:] = 1.
        data30[iko[i][0],iko[i][1],:] = 1.
        data35[iko[i][0],iko[i][1],:] = 1.
      
    min25 = np.unravel_index(np.argmin(data25), data25.shape)
    min30 = np.unravel_index(np.argmin(data30), data25.shape)
    min35 = np.unravel_index(np.argmin(data35), data25.shape)

    print(files[f], ', lamc (nm):', int(np.round(lamC*1e9,0)))
    print('25 mas:',dL[min25[0]],obs[min25[1]],mB[min25[2]],
          throughput[min25[0:2]])
    print('30 mas:',dL[min30[0]],obs[min30[1]],mB[min30[2]])
    print('35 mas:',dL[min35[0]],obs[min35[1]],mB[min35[2]])

    cd_min = -6
    cd_max = -2
    
    for s in np.arange(3):
        
        plt.figure()
        plt.imshow(np.log10(data[s][min25[0],:,:]), vmin=cd_min, vmax=cd_max,
                    extent=(mB_min,mB_max,obs_max,obs_min),
                    aspect=(mB_max-mB_min)/(obs_max-obs_min),cmap='inferno')
        plt.xlabel("FPM [lam/D]")
        plt.ylabel("obstruction")
        plt.title((band+', '+ang_sep[s]+' mas, '+'lyot stop diam. : ' +
                   str(np.round(dL[min25[0]],2))))
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(np.log10(data[s][:,min25[1],:]), vmin=cd_min, vmax=cd_max,
                    extent=(mB_min,mB_max,dL_max,dL_min),
                    aspect=(mB_max-mB_min)/(dL_max-dL_min), cmap='inferno')
        plt.xlabel("FPM [lam/D]")
        plt.ylabel("Lyot stop diam.")
        plt.title((band+', '+ang_sep[s]+' mas, '+'obstruction : ' +
                   str(np.round(obs[min25[1]],2))))
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(np.log10(data[s][:,:,min25[2]]), vmin=cd_min, vmax=cd_max,
                    extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno')
        plt.xlabel("obstruction")
        plt.ylabel("Lyot stop diam.")
        plt.title((band+', '+ang_sep[s]+' mas, '+'FPM [lam/D] : ' +
                   str(np.round(mB[min25[2]],2))))
        plt.colorbar()
        plt.show()

