# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:14:11 2024

@author: asp
"""

# visualize outputs of explore_andes_corono_parameters.py 


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import os

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'hatch.color':'w'})
# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# field radius of interest in mas
f_rad = 25

# throughput threshold 0.8^2 = .64
# t_t = [1., 0.73, 0.64]


#%%
res_dirs=("d:/Andes/Data_corono/results/20250117175452",) # K
res_dirs=("d:Andes/Data_corono/results/20250117175444",) # YJH

# wvl in Y, J, H, K
wvl = 'K'
wvl = 'YJH'


#%%

print("threshold  throughput  intensity  [parameters DLyot obst. FPM]")

for i in range(len(res_dirs)):
    
    donow_dir = os.path.basename(res_dirs[i]).split('.')[0]
    print(donow_dir)

    fnm = Path(res_dirs[i] +
               '/Parameters_results_w_coro_no_turb_'+wvl+'.fits')
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
    i_f_rad = np.argmin(abs(np.arange(nImg//2) * mD * lamD2mas / nImg - f_rad*1.))

    coro_data = coro_data[1,:,:,:,i_f_rad]
    data_shape = coro_data.shape
    i_min = np.unravel_index(np.argmin(coro_data), data_shape)
    cd_min = np.log10(np.min(coro_data))
    cd_max = np.log10(np.max(coro_data))

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
    for i in range(data_shape[2]):
        thr_cube[:,:,i] = thrp_data.copy()
    
    thrVsInt = []
    thr = 0.75

    test = True
    while test:
        
        ma_data = np.ma.where(thr_cube > thr, coro_data.copy(), 1.)
        if ma_data.mean() == 1.:
            break
        
        i_min = np.unravel_index(np.argmin(ma_data), data_shape)
        cd_min = np.log10(np.min(ma_data)) - 0.1
        cd_max = np.log10(np.max(ma_data[ma_data<1.])) + 0.1
        # ma_data[np.where(ma_data==1)]=0.
        # (coro_data.copy()[np.where(ma_data==1)]*0.9)

        plt.figure(4)
        plt.tight_layout()
    
        # plt.imshow(np.log10(ma_data[i_min[0],:,:]), vmin=cd_min, vmax=cd_max,
        #             extent=(mB_min,mB_max,obs_max,obs_min),
        #             aspect=(mB_max-mB_min)/(obs_max-obs_min),cmap='inferno')

        y, x = np.mgrid[slice(obs_min,obs_max+obs_stp,obs_stp),
                        slice(mB_min,mB_max+mB_stp,mB_stp)]
        zm = -np.ma.masked_less(-thr_cube[i_min[0],:,:],-thr)
        plt.pcolor(x,y,zm,hatch='/',alpha=0)
        plt.imshow(np.log10(coro_data[i_min[0],:,:]), vmin=cd_min, vmax=cd_max,
                    extent=(mB_min-mB_stp,mB_max+mB_stp,
                            obs_min-obs_stp,obs_max+obs_stp*2),
                    aspect=(mB_max-mB_min)/(obs_max-obs_min),
                    cmap='inferno',origin='lower')
        plt.plot(mB[i_min[2]], obs[i_min[1]], marker='o', color='y')
                                          
        plt.axhline(obs[i_min[1]], color='w', ls='--')
        plt.axvline(mB[i_min[2]], color='w', ls='--')

        plt.xlabel(r'FPM diameter [$\lambda_c$/D]', fontsize=14)
        plt.ylabel("Lyot stop inner diameter [D]", fontsize=14)
        plt.title((f'Lyot stop outer diameter: {dL[i_min[0]]:.2f}D'),
                  fontsize=14)
        plt.colorbar(label=f'Contrast@{f_rad}mas in log scale')
        fnm = ("/obsVsFPM_AngSep"+str(int(np.round(f_rad)))+"mas_thr"+
               str(int(np.round(1000*thr)))+f"_{wvl}_250123_2")
        plt.savefig(save_dir+fnm+".pdf")
        plt.savefig(save_dir+fnm+".svg")
        plt.show()
        
        plt.figure(5)
        plt.tight_layout()

        # plt.imshow(np.log10(ma_data[:,i_min[1],:]), vmin=cd_min, vmax=cd_max,
        #             extent=(mB_min,mB_max,dL_max,dL_min),
        #             aspect=(mB_max-mB_min)/(dL_max-dL_min), cmap='inferno')

        y, x = np.mgrid[slice(dL_min,dL_max+dL_stp,dL_stp),
                        slice(mB_min,mB_max+mB_stp,mB_stp)]
        zm = -np.ma.masked_less(-thr_cube[:,i_min[1],:],-thr)
        plt.pcolor(x,y,zm,hatch='/',alpha=0.)
        plt.imshow(np.log10(coro_data[:,i_min[1],:]), vmin=cd_min, vmax=cd_max,
                    extent=(mB_min-mB_stp,mB_max+mB_stp,
                            dL_min-dL_stp,dL_max+dL_stp),
                    aspect=(mB_max-mB_min)/(dL_max-dL_min),
                    origin='lower',cmap='inferno')
        plt.plot(mB[i_min[2]], dL[i_min[0]],  marker='o', color='y')
            
        plt.axhline(dL[i_min[0]], color='w', ls='--')
        plt.axvline(mB[i_min[2]], color='w', ls='--')

        # plt.imshow(np.log10(ma_data[:,i_min[1],:]), vmin=cd_min, vmax=cd_max,
        #             extent=(mB_min,mB_max,dL_max,dL_min),
        #             aspect=(mB_max-mB_min)/(dL_max-dL_min), cmap='inferno',alpha=0.2)

        plt.xlabel(r'FPM diameter [$\lambda_c$/D]', fontsize=14)
        plt.ylabel("Lyot stop outer diameter [D]", fontsize=14)
        plt.title((f'Lyot stop inner diameter: {obs[i_min[1]]:.2f}D'),
                  fontsize=14)
        plt.colorbar(label=f'Contrast@{f_rad}mas in log scale')
        fnm = ("/DLyotVsFPM_AngSep"+str(int(np.round(f_rad)))+"mas_thr"+
               str(int(np.round(1000*thr)))+f"_{wvl}_250123_2")
        plt.savefig(save_dir+fnm+".pdf")
        plt.savefig(save_dir+fnm+".svg")
        plt.show()
    
        plt.figure(6)
        plt.tight_layout()

        # plt.imshow(np.log10(ma_data[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
        #             extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno')

        
        y, x = np.mgrid[slice(dL_min,dL_max+dL_stp,dL_stp),
                        slice(obs_min,obs_max+obs_stp,obs_stp)]
        zm = -np.ma.masked_less(-thr_cube[:,:,i_min[2]],-thr)
        plt.pcolor(x,y,zm,hatch='/',alpha=0.)
        plt.imshow(np.log10(coro_data[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
                    extent=(obs_min-obs_stp,obs_max+obs_stp*2,
                            dL_min-dL_stp,dL_max+dL_stp,),
                    origin='lower',cmap='inferno')
        plt.plot( obs[i_min[1]],dL[i_min[0]], marker='o', color='y')

        plt.axhline(dL[i_min[0]], color='w', ls='--')
        plt.axvline(obs[i_min[1]], color='w', ls='--')

        # plt.imshow(np.log10(ma_data[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
        #             extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno',alpha=0.2)

        plt.xlabel("Lyot stop inner diameter [D]", fontsize=14)
        plt.ylabel("Lyot stop outer diameter [D]", fontsize=14)
        plt.title((f'FPM diameter: {mB[i_min[2]]:.2f}'r'$\lambda_c$/D'),
                  fontsize=14)
        plt.colorbar(label=f'Contrast@{f_rad}mas in log scale')
        fnm = ("/DLyotVsObs_AngSep"+str(int(np.round(f_rad)))+"mas_thr"
               +str(int(np.round(1000*thr)))+f"_{wvl}_250123_2")
        plt.savefig(save_dir+fnm+".pdf")
        plt.savefig(save_dir+fnm+".svg")
        plt.show()
        
        # plt.figure(7)
    
        # ma_tp_data = np.ma.masked_where(thrp_data<thr,thrp_data.copy(),1.)
    
        # plt.imshow((ma_tp_data),
        #             vmin=np.min((thrp_data)),
        #             vmax=np.max((thrp_data)),
        #             extent=(obs_min,obs_max,dL_max,dL_min), cmap='gray')
    
        # plt.colorbar()
        # plt.xlabel("obstruction")
        # plt.ylabel("Lyot stop diameter")
        # fnm = ("/throughput_DLyoVsobs_radMas"+str(int(np.round(f_rad)))+"_t"
        #        +str(int(np.round(1000*thr)))+"m.")
        # plt.savefig(save_dir+fnm+"pdf")
        # plt.savefig(save_dir+fnm+"svg")
        # plt.show()
        
        # print("\t", np.round(thr,3), "\t",
        #       np.round(thrp_data[i_min[0],i_min[1]],3), "\t",
        #       np.round(np.min(ma_data),9), "\t",
        #       np.round([dL[i_min[0]],obs[i_min[1]],mB[i_min[2]]],2))

        # thr = thrp_data[i_min[0],i_min[1]].copy()
        # thrVsInt.append([[np.round(thrp_data[i_min[0],i_min[1]],3)],
        #                  [np.round(np.min(ma_data),9)]])

        test = False


    # thrVsInt = np.array(thrVsInt)
    # plt.figure(8)
    # plt.plot(thrVsInt[:,0,0], thrVsInt[:,1,0])
    # plt.xlabel("throughput")
    # plt.ylabel("intensity @ " + str(int(f_rad)) + " mas radius")
    # plt.title("wvl : " + str(int(lam*1e9)) + " nm")
    # fnm = "/throughput_DLyoVsobs_allThresholds."
    # plt.savefig(save_dir+fnm+"pdf")
    # plt.savefig(save_dir+fnm+"svg")
    # plt.show()
    
    
