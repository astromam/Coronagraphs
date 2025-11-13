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

plt.rcParams.update({'font.size': 12})

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# field radius of interest in mas
f_rad = 25.


#%%
res_dirs=("d:/Andes/Data_corono/results/20240927090738",)
# 20240801100039 , 20240801112122 , 20240801134347 , 20240801145829
# wvl in Y, J, H, K
wvl = 'YJH'

#%%

print("threshold  throughput  intensity  [parameters DLyot obst. FPM]")

for i in range(len(res_dirs)):
    
    donow_dir = os.path.basename(res_dirs[i]).split('.')[0]
    print(donow_dir, wvl, f_rad)

    fnm = Path(res_dirs[i] +
               '/Parameters_results_contrast_w_coro_no_turb_'+wvl+'.fits')
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

    # fnm = Path(res_dirs[i] +
    #            '/Parameters_results_contrast_no_coro_no_turb_'+wvl+'.fits')
    # psf_data = fits.getdata(fnm)
    # gain_data = psf_data / coro_data
    
    lamD2mas = (lam/D)*mas2rad
    # FoV in lam/D in the final image plane D
    mD = 58.393*(nImg/(lam*1e9))
    i_f_rad = np.argmin(abs(np.arange(nImg//2) * mD * lamD2mas / nImg - f_rad))

    coro_data = coro_data[1,:,:,:,i_f_rad]
    data_shape = coro_data.shape
    i_min = np.unravel_index(np.argmin(coro_data), data_shape)
    cd_min = np.log10(np.min(coro_data))
    cd_max = np.log10(np.max(coro_data))

    # gain_data = gain_data[1,:,:,:,i_f_rad]

    fnm = Path(res_dirs[i] + '/Parameters_throughput_'+wvl+'.fits')
    thrp_data = fits.getdata(fnm)

# %%
    
    save_dir = ("d:/Andes/Data_corono/plots/"+donow_dir)
    os.makedirs(save_dir, exist_ok=True)
    
#     plt.figure(0)
#     plt.imshow(np.log10(coro_data[i_min[0],:,:]), vmin=cd_min, vmax=cd_max,
#                extent=(mB_min,mB_max,obs_max,obs_min),
#                aspect=(mB_max-mB_min)/(obs_max-obs_min),cmap='inferno')
#     plt.xlabel("FPM [lam/D]")
#     plt.ylabel("obstruction")
#     plt.title(('lyot stop diam. : ' + str(np.round(dL[i_min[0]],2))))
#     plt.colorbar()
#     plt.savefig(save_dir+"/obsVsFPM.pdf")
#     plt.savefig(save_dir+"/obsVsFPM.svg")
#     plt.show()
    
#     plt.figure(1)
#     plt.imshow(np.log10(coro_data[:,i_min[1],:]), vmin=cd_min, vmax=cd_max,
#                extent=(mB_min,mB_max,dL_max,dL_min),
#                aspect=(mB_max-mB_min)/(dL_max-dL_min), cmap='inferno')
#     plt.xlabel("FPM [lam/D]")
#     plt.ylabel("Lyot stop diam.")
#     plt.title(('obstruction : ' + str(np.round(obs[i_min[1]],2))))
#     plt.colorbar()
#     plt.savefig(save_dir+"/DLyotVsFPM.pdf")
#     plt.savefig(save_dir+"/DLyotVsFPM.svg")
#     plt.show()
    
#     plt.figure(2)
#     plt.imshow(np.log10(coro_data[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
#                extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno')
#     plt.xlabel("obstruction")
#     plt.ylabel("Lyot stop diam.")
#     plt.title(('FPM [lam/D] : ' + str(np.round(mB[i_min[2]],2))))
#     plt.colorbar()
#     plt.savefig(save_dir+"/DLyoVsobs.pdf")
#     plt.savefig(save_dir+"/DLyoVsobs.svg")
#     plt.show()
    
#     plt.figure(3)
#     plt.imshow((thrp_data[:,:]),
#                vmin=np.min((thrp_data)),
#                vmax=np.max((thrp_data)),
#                extent=(obs_min,obs_max,dL_max,dL_min), cmap='gray')
    
#     plt.colorbar()
#     plt.xlabel("obstruction")
#     plt.ylabel("Lyot stop diam.")
#     plt.savefig(save_dir+"/throughput_DLyoVsobs.pdf")
#     plt.savefig(save_dir+"/throughputDLyoVsobs.svg")
#     plt.show()

#     print(donow_dir, np.round(v_width),
#           np.round(thrp_data[i_min[0],i_min[1]],3),
#           np.round(np.min(coro_data),9),
#           np.round([dL[i_min[0]],obs[i_min[1]],mB[i_min[2]]],2))


#%%
# masked

    thr_cube = np.zeros(data_shape)
    for j in range(data_shape[2]):
        thr_cube[:,:,j] = thrp_data.copy()
    
    thrVsInt = []
    thr = 0.

    while True:
        
        ma_data = np.ma.where(thr_cube > thr, coro_data.copy(), 1.)
        if ma_data.mean() == 1.:
            break
        
        i_min = np.unravel_index(np.argmin(ma_data), data_shape)
        cd_min = np.log10(np.min(ma_data))
        cd_max = np.log10(np.max(ma_data[ma_data<1.]))
        
        
        plt.figure(4)
    
        plt.imshow(np.log10(ma_data[i_min[0],:,:]), vmin=cd_min, vmax=cd_max,
                    extent=(mB_min,mB_max,obs_max,obs_min),
                    aspect=(mB_max-mB_min)/(obs_max-obs_min),cmap='inferno')
    
        plt.xlabel("FPM [lam/D]")
        plt.ylabel("obstruction")
        plt.title(('lyot stop diam. : ' + str(np.round(dL[i_min[0]],2))))
        plt.colorbar()
        fnm = ("/obsVsFPM_radMas"+str(int(np.round(f_rad)))+"_t"+
               str(int(np.round(1000*thr)))+"m.")
        plt.savefig(save_dir+fnm+"pdf")
        plt.savefig(save_dir+fnm+"svg")
        plt.show()
        
        plt.figure(5)
    
        plt.imshow(np.log10(ma_data[:,i_min[1],:]), vmin=cd_min, vmax=cd_max,
                    extent=(mB_min,mB_max,dL_max,dL_min),
                    aspect=(mB_max-mB_min)/(dL_max-dL_min), cmap='inferno')
    
        plt.xlabel("FPM [lam/D]")
        plt.ylabel("Lyot stop diam.")
        plt.title(('obstruction : ' + str(np.round(obs[i_min[1]],2))))
        plt.colorbar()
        fnm = ("/DLyotVsFPM_radMas"+str(int(np.round(f_rad)))+"_t"+
               str(int(np.round(1000*thr)))+"m.")
        plt.savefig(save_dir+fnm+"pdf")
        plt.savefig(save_dir+fnm+"svg")
        plt.show()
    
        
        plt.figure(6)
    
        plt.imshow(np.log10(ma_data[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
                    extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno')
    
        plt.xlabel("obstruction")
        plt.ylabel("Lyot stop diam.")
        plt.title(('FPM [lam/D] : ' + str(np.round(mB[i_min[2]],2))))
        plt.colorbar()
        fnm = ("/DLyotVsObs_radMas"+str(int(np.round(f_rad)))+"_t"
               +str(int(np.round(1000*thr)))+"m.")
        plt.savefig(save_dir+fnm+"pdf")
        plt.savefig(save_dir+fnm+"svg")
        plt.show()
        
    
        plt.figure(7)
    
        ma_tp_data = np.ma.masked_where(thrp_data<thr,thrp_data.copy(),1.)
    
        plt.imshow((ma_tp_data),
                    vmin=np.min((thrp_data)),
                    vmax=np.max((thrp_data)),
                    extent=(obs_min,obs_max,dL_max,dL_min), cmap='gray')
    
        plt.colorbar()
        plt.xlabel("obstruction")
        plt.ylabel("Lyot stop diam.")
        fnm = ("/throughput_DLyoVsobs_radMas"+str(int(np.round(f_rad)))+"_t"
               +str(int(np.round(1000*thr)))+"m.")
        plt.savefig(save_dir+fnm+"pdf")
        plt.savefig(save_dir+fnm+"svg")
        plt.show()
        
        print("\t", np.round(thr,3), "\t",
              np.round(thrp_data[i_min[0],i_min[1]],3), "\t",
              np.round(np.min(ma_data),9), "\t",
              np.round([dL[i_min[0]],obs[i_min[1]],mB[i_min[2]]],2))

        thr = thrp_data[i_min[0],i_min[1]].copy()
        thrVsInt.append([[np.round(thrp_data[i_min[0],i_min[1]],3)],
                         [np.round(np.min(ma_data),9)],
                         [np.round(dL[i_min[0]],2)],
                         [np.round(obs[i_min[1]],2)],
                         [np.round(mB[i_min[2]],2)]])

    thrVsInt = np.array(thrVsInt)
    plt.figure(8, (8,5))
    plt.plot(thrVsInt[:,0,0], thrVsInt[:,1,0])
    plt.xlabel("throughput")
    plt.ylabel("contrast @ " + str(int(f_rad)) + " mas radius")
    plt.title("wvl : " + str(int(lam*1e9)) + " nm")
    fnm = ("/throughput_DLyoVsobs_allThresholds_wvl"+ str(int(lam*1e9)) +
           "nm_radMas"+str(int(np.round(f_rad))))
    plt.savefig(save_dir+fnm+".pdf")
    plt.savefig(save_dir+fnm+".svg")
    fits.writeto(Path(res_dirs[i] + fnm + ".fits"), thrVsInt,
                 header=head_data, overwrite=True)
    plt.show()
    
