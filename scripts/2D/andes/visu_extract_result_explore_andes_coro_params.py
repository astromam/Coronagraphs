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

dL_min = 0.81
dL_max = 0.96
dL_stp = 0.01
dL = np.round(np.arange(dL_min,dL_max+dL_stp,dL_stp),3)
obs_min = 0.30
obs_max = 0.44
obs_stp = 0.01
obs = np.round(np.arange(obs_min,obs_max+obs_stp,obs_stp),3)
mB_min = 3.0
mB_max = 4.5
mB_stp = 0.1
mB = np.round(np.arange(mB_min,mB_max+mB_stp,mB_stp),3)

thr = 0.75

# D = 38.54
# rad2mas = np.pi/(180.*3600*1000)
# mas2rad = 1/rad2mas
# nImg = 400
# aS = np.arange(nImg//2) * (mas2rad * 58.393 / (D * 1e9) )

# from path_recursive_file_search import recursive_search
fdir_dat = Path("d:/Andes/Data_corono/results/exploreCoroParamsYJHK_25_30_35_mas_lbdCtoDwidth/").resolve()

# file_list=recursive_search(".")
fnm = '20240801100039/Parameters_throughput_Y.fits'
throughput = fits.getdata( fdir_dat/fnm )
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

files = ['20240801100039\\Parameters_results_contrast_w_coro_no_turb_Y.fits',
 '20240801112122\\Parameters_results_contrast_w_coro_no_turb_J.fits',
 '20240801134347\\Parameters_results_contrast_w_coro_no_turb_H.fits',
 '20240801145829\\Parameters_results_contrast_w_coro_no_turb_K.fits']


for f in np.arange(len(files)):
    
    parameters_results_contrast = fits.getdata(fdir_dat / files[f])
    #80 = 25 mas, 96 = 30 mas, 112 = 35 mas, indices from aS
    data25 = parameters_results_contrast[1,:,:,:,80]
    data30 = parameters_results_contrast[1,:,:,:,96]
    data35 = parameters_results_contrast[1,:,:,:,112]
    data = [data25,data30,data35]
    ang_sep = ['25','30','35']
    band = (fdir_dat / files[f]).stem[-1]

    for i in np.arange(iko.shape[0]):
        data25[iko[i][0],iko[i][1],:] = 1.
        data30[iko[i][0],iko[i][1],:] = 1.
        data35[iko[i][0],iko[i][1],:] = 1.
      
    min25 = np.unravel_index(np.argmin(data25), data25.shape)
    min30 = np.unravel_index(np.argmin(data30), data25.shape)
    min35 = np.unravel_index(np.argmin(data35), data25.shape)

    print(files[f])
    print('25 mas:',dL[min25[0]],obs[min25[1]],mB[min25[2]])
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
