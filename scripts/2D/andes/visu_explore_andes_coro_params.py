# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:14:11 2024

@author: asp
"""
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
res_dirs=("d:/Andes/Data_corono/results/20240705161700",
          "d:/Andes/Data_corono/results/20240705162606",
          "d:/Andes/Data_corono/results/20240705163115",
          "d:/Andes/Data_corono/results/20240705163602")

#%%

print("work. dir. / vane width / throughput / min. intensity / min params")

for i in range(len(res_dirs)):
    
    donow_dir = os.path.basename(res_dirs[i]).split('.')[0]

    fnm = Path(res_dirs[i] + '/Parameters_resultat_w_coro_no_turb_H.fits')
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
    mD = 58.393*(nImg/1600)
    i_f_rad = np.argmin(abs(np.arange(nImg//2) * mD * lamD2mas / nImg - f_rad))

    coro_data = coro_data[1,:,:,:,i_f_rad]
    data_shape = coro_data.shape
    i_min = np.unravel_index(np.argmin(coro_data), data_shape)
    cd_min = np.log10(np.min(coro_data))
    cd_max = np.log10(np.max(coro_data))

#%%
    
    save_dir = ("d:/Andes/Data_corono/plots/"+donow_dir)
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(0)
    plt.imshow(np.log10(coro_data[i_min[0],:,:]), vmin=cd_min, vmax=cd_max,
               extent=(mB_min,mB_max,obs_max,obs_min),
               aspect=(mB_max-mB_min)/(obs_max-obs_min),cmap='inferno')
    plt.xlabel("FPM [lam/D]")
    plt.ylabel("obscuration")
    plt.title(('lyot diam. frac. :' + str(np.round(dL[i_min[0]],2))))
    plt.colorbar()
    plt.savefig(save_dir+"/obsVsFPM.pdf")
    plt.savefig(save_dir+"/obsVsFPM.svg")
    plt.show()
    
    plt.figure(1)
    plt.imshow(np.log10(coro_data[:,i_min[1],:]), vmin=cd_min, vmax=cd_max,
               extent=(mB_min,mB_max,dL_max,dL_min),
               aspect=(mB_max-mB_min)/(dL_max-dL_min), cmap='inferno')
    plt.xlabel("FPM [lam/D]")
    plt.ylabel("Lyot stop diam.")
    plt.title(('obscuration :' + str(np.round(obs[i_min[1]],2))))
    plt.colorbar()
    plt.savefig(save_dir+"/DLyotVsFPM.pdf")
    plt.savefig(save_dir+"/DLyotVsFPM.svg")
    plt.show()
    
    plt.figure(2)
    plt.imshow(np.log10(coro_data[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
               extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno')
    plt.xlabel("obscuration")
    plt.ylabel("Lyot stop diam.")
    plt.title(('FPM [lam/D] :' + str(np.round(mB[i_min[2]],2))))
    plt.colorbar()
    plt.savefig(save_dir+"/DLyoVsobs.pdf")
    plt.savefig(save_dir+"/DLyoVsobs.svg")
    plt.show()
    
    fnm = Path(res_dirs[i] + '/Parameters_throughput_H.fits')
    thrp_data = fits.getdata(fnm)
    plt.figure(3)
    plt.imshow((thrp_data[:,:]),
               vmin=np.min((thrp_data)),
               vmax=np.max((thrp_data)),
               extent=(obs_min,obs_max,dL_max,dL_min), cmap='gray')
    
    # plt.xlabel("Lyot diam. frac.")
    # plt.ylabel("obscuration")
    plt.colorbar()
    plt.xlabel("obscuration")
    plt.ylabel("Lyot stop diam.")
    plt.savefig(save_dir+"/throughput_DLyoVsobs.pdf")
    plt.savefig(save_dir+"/throughputDLyoVsobs.svg")
    plt.show()

    print(donow_dir, np.round(v_width),
          np.round(thrp_data[i_min[0],i_min[1]],3),
          np.round(np.min(coro_data),9),
          np.round([dL[i_min[0]],obs[i_min[1]],mB[i_min[2]]],2))


    