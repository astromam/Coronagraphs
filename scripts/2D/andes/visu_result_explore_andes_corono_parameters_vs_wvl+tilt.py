#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@author: mndiaye, asimmonnin, asp
"""

# visualize exploration of coronagraph parameters ranges versus metrics on
# throughput, integrated energy of the star and planet psf in the field, with 
# or without coro, at some wavelengths for a given spaxel size

#%%
"""
### Initialization
"""
import numpy as np
# import slow_fourier_transform as sft
# from uniform_disk import uniform_disk
from astropy.io import fits

import os
from pathlib import Path
import matplotlib.pyplot as plt
plt.close()
plt.rcParams.update({'font.size': 12})  #♦  mdiaye 15!
# plt.rcParams.update({'figure.titlesize': 12})  #♦  mdiaye 15!

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

#%%
"""
input parameters:
    directory with files: was_donow value
    band choice within definded wvls
"""

was_donow = '20250924113208'

# spaxel (mas) directory
#  7.0 20250924113208
# 10.0 20250924113340
# 16.0 20250925152805

band='full_YJH'
wvls = {'J':[1.15e-06, 1.20e-06, 1.25e-06, 1.30e-06, 1.35e-06],
         
         'H':[1.40e-06, 1.45e-06, 1.50e-06, 1.55e-06, 1.60e-06,
              1.65e-06, 1.70e-06, 1.75e-06, 1.80e-06],
         
         'full_JH':[1.15e-06, 1.20e-06, 1.25e-06, 1.30e-06,
               1.35e-06, 1.40e-06, 1.45e-06,
               1.50e-06, 1.55e-06, 1.60e-06,
               1.65e-06, 1.70e-06, 1.75e-06],
         
         'full_YJH':[9.50e-07, 1.00e-06, 1.05e-06, 1.10e-06, 1.15e-06,
                1.20e-06, 1.25e-06, 1.30e-06, 1.35e-06, 1.40e-06, 
                1.45e-06, 1.50e-06, 1.55e-06, 1.60e-06, 1.65e-06,
                1.70e-06, 1.75e-06],

         'YJH_atm':[9.50e-07, 1.00e-06, 1.05e-06, 1.10e-06,
                1.20e-06, 1.25e-06, 1.30e-06, 1.35e-06, 
                1.50e-06, 1.55e-06, 1.60e-06, 1.65e-06,
                1.70e-06, 1.75e-06],
         'full_K':[1.95e-06, 2.00e-06, 2.05e-06, 2.10e-06,
                   2.15e-06, 2.20e-06, 2.25e-06, 2.30e-06,
                   2.35e-06, 2.40e-06, 2.45e-06],
         
         # 'yjh_atm':[1.00e-06, 1.05e-06, 1.15e-06, 1.20e-06, 1.25e-06,
         #        1.30e-06, 1.45e-06, 1.50e-06, 1.60e-06, 1.65e-06,
         #        1.70e-06, 1.75e-06],
         # # 'yjh':[1.00e-06, 1.05e-06, 1.10e-06, 1.15e-06, 1.20e-06, 1.25e-06,
         #        1.30e-06, 1.35e-06, 1.40e-06, 1.45e-06, 1.50e-06, 1.55e-06,
         #        1.60e-06, 1.65e-06, 1.70e-06, 1.75e-06],
         
         'jh_atm':[1.15e-06, 1.20e-06, 1.25e-06, 1.30e-06, 1.45e-06,
               1.50e-06, 1.55e-6, 1.60e-06, 1.65e-06, 1.70e-06, 1.75e-06]}

# 'yjh' near to legacy YJH [9.80e-07, 1.00e-06, 1.02e-06, 1.04e-06, 1.06e-06,
#                           1.16e-06, 1.20e-06, 1.24e-06, 1.28e-06, 1.32e-06,
#                           1.44e-06, 1.52e-06, 1.60e-06, 1.68e-06, 1.76e-06]
# 'jh' near to legacy JH [1.16e-06, 1.20e-06, 1.24e-06, 1.28e-06, 1.32e-06,
#                         1.44e-06, 1.52e-06, 1.60e-06, 1.68e-06, 1.76e-06]

#%%
"""
### Working directory
"""
user = 'Alain'
if user == 'Alain':
    fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()  # opd's seed value
    fdir_res = Path('D:/Andes/Data_corono/results/').resolve()  #  fits files
    fdir_res = (fdir_res / was_donow)
    fdir_plot = Path('D:/Andes/Data_corono/plots/').resolve()
    fdir_plot = (fdir_plot / was_donow)
    os.makedirs(fdir_plot, exist_ok=True)


#%%
"""
### Retrieve data
"""

file_lst = os.listdir(fdir_res)
file_res = [x for x in file_lst if 'results_vs_params_' in x]
file_coro_thr = [x for x in file_lst if 'Parameters_coro_thr_' in x]
file_plnt_thr = [x for x in file_lst if 'Parameters_planet_thr_' in x]
file_eelt_int = [x for x in file_lst if 'eelt_intensities_vs_params_' in x]
file_lyot_int = [x for x in file_lst if 'lyot_intensities_vs_params_' in x]
file_eta_tel = [x for x in file_lst if 'eta_tel_vs_params_' in x]

sim_data = fits.getdata(fdir_res/file_res[0])
thr_coro_data = fits.getdata(fdir_res/file_coro_thr[0])
thr_plnt_data = fits.getdata(fdir_res/file_plnt_thr[0])
eelt_int_data = fits.getdata(fdir_res/file_eelt_int[0])
lyot_int_data = fits.getdata(fdir_res/file_lyot_int[0])
eta_tel_data  = fits.getdata(fdir_res/file_eta_tel[0])

file_lyot_max = [x for x in file_lst if 'lyot_max_intensity_vs_params' in x]
lyot_max_psf_data = fits.getdata(fdir_res/file_lyot_max[0])


#%%
"""
parameters
"""

hdr = fits.getheader(fdir_res/file_res[0])

# Read ELT pupil 
elt_pup_fnm = hdr['EPUP_FNM'] # 'ELT_pupil_400.fits' # 2024
fdir_pupil = fdir_dat / 'Pupil'
fpath_elt = fdir_pupil / elt_pup_fnm
Pupil = fits.getdata(fpath_elt,)

nPup = hdr['NPUP']
nFPM = hdr['NFPM']
nImg = hdr['NIMG']
lamC = hdr['LMBD']
D = hdr['DIAM']
pscale = hdr['PSCL']

# field of view in mas
fov_mas = nImg * pscale
# fov in radians
fov_rdn = fov_mas * rad2mas
# in multiple of reference lambda (lamC) over D, 05/2025 -> 
mD_ref = fov_rdn / ( lamC / D )

lam_min = hdr['LMIN']
lam_itv = hdr['LITV']
lam_stp = hdr['LSTP']
lam_lst = np.round(np.arange(lam_min,lam_min+lam_itv*lam_stp+1e-9,lam_stp),8)
# lam_lst = np.array([1.55e-06, 1.60e-06, 1.65e-06]) # debug
nL = len(lam_lst)

mB_min = hdr['SFPM_MIN']
mB_itv = hdr['SFPM_ITV']
mB_stp = hdr['SFPM_STP']
mB_lst = np.round(np.arange(mB_min,mB_min+mB_stp*(mB_itv+0.5),mB_stp),3)

obs_min = hdr['OBST_MIN']
obs_itv = hdr['OBST_ITV']
obs_stp = hdr['OBST_STP']
obs_lst = np.round(np.arange(obs_min,obs_min+obs_stp*(obs_itv+0.5),obs_stp),3)

dL_min = hdr['DLYO_MIN']
dL_itv = hdr['DLYO_ITV']
dL_stp = hdr['DLYO_STP']
dL_lst = np.round(np.arange(dL_min,dL_min+dL_stp*(dL_itv+.5),dL_stp),3)

ap_mas = hdr['D_SPAXEL']
as_oi  = hdr['SPXL_ROI']
# as_oi = 25 # 25 as up to 05/2025
ap_min = hdr['SPXL_MIN']
ap_itv = hdr['SPXl_ITV']
ap_stp = hdr['SPXL_STP']
ap_lst = np.arange(ap_min,ap_min+ap_stp*ap_itv+1,ap_stp)
nA = len(ap_lst)
ap_area = np.pi * (ap_mas/2.)**2.


#%%
"""
areas of interest: aperture encircled energy and ring encircled energy
"""

a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
b_ = a_.copy()
aa, bb = np.meshgrid(a_, b_)
rad_mas = np.zeros((nImg,nImg,nA+1))
rad_mas[:,:,0] = np.sqrt(aa**2 + bb**2) * pscale
for a in range(nA):
    rad_mas[:,:,a+1] = np.sqrt(((aa - ap_lst[a]) * pscale)**2 +
                               (bb * pscale)**2)

aee = {a:np.where(rad_mas[:,:,a+1]<=ap_mas/2.) for a in range(nA)}
ree = {a:np.where(
    np.abs(rad_mas[:,:,0]-ap_lst[a]*1.)<=ap_mas/2) for a in range(nA)}


#%%
"""
data scaling
"""

eta_data = sim_data.copy()
star_int = sim_data[0,:].copy()


# lyot_max_psf_data = lyot_int_data.copy() * 0.


# for i in range(nL):

#     mD = mD_ref * lamC / lam_lst[i]
    
#     for Ld in range(len(dL_lst)): 

#         diam = dL_lst[Ld]

#         for ob in range(len(obs_lst)):

#             obst=obs_lst[ob]
#             LyotStop2d = (Pupil.copy() * (uniform_disk(nPup, diam*nPup/2) -
#                                           uniform_disk(nPup, obst*nPup/2)))
#             lyot_max_psf_data[Ld,ob,i] = np.max(np.abs(
#                 sft.sft(Pupil * LyotStop2d, nImg, mD*diam))**2.)
# fits.writeto(fdir_res / ('lyot_max_intensity_vs_params.fits'), 
#              np.array(lyot_max_psf_data),overwrite=True)

for i in range(nA):

    # eta star integrated over ring surface scaled down to the aperture surface
    eta_data[0,i,:] = (eta_data[0,i,:] * float(len(aee[i][0])) /
                       float(len(ree[i][0])))


    star_int[i,:] = (( star_int[i,:] / float(len(ree[i][0])) ) *
                     eelt_int_data[None,:,:,None,:]/
                     lyot_max_psf_data[None,:,:,None,:])


#%%
"""
wavelengths of interest
"""
i_wvl = np.where(np.array([ np.abs(lam_lst[:,None] -
                                   np.array(wvls[band])[None,:])
                           <= 2e-8]).any(axis=2)[0])[0]


#%%
'''
data to compute metrics at angular separation of interest
'''
star_int_oi = star_int[:,:,:,:,i_wvl].copy()
star_int_oi = star_int_oi[np.where(ap_lst == as_oi)[0],:,:,:,:][0,:]

eta_pla_abs = eta_data[1,:][:,:,:,:,i_wvl].copy()
eta_pla_abs = eta_pla_abs[np.where(ap_lst == as_oi)[0],:,:,:,:][0,:]

eta_sta_abs = eta_data[0,:][:,:,:,:,i_wvl].copy()
eta_sta_abs = eta_sta_abs[np.where(ap_lst == as_oi)[0],:,:,:,:][0,:]

eta_tel_norm = (eta_tel_data.copy() / eelt_int_data.copy())
eta_tel_norm = eta_tel_norm[:,:,i_wvl]


#%%

print('\n', "band: ", band, ", spaxel size: ",ap_mas, '\t', 
      ", angular separation: ", as_oi, '\t', ", working directory: ", was_donow)

# legacy contrast evaluation method
m_sta_int_leg = np.mean(star_int_oi, axis=3)

# ruane 2018 metrics 
m_eta_sta_abs = np.mean(eta_sta_abs,axis=3)
m_eta_pla_abs = np.mean(eta_pla_abs,axis=3)

m_con_raw = m_eta_sta_abs / m_eta_pla_abs
# s_con_raw = s_eta_sta_abs

m_eta_pla_rel = (np.mean(eta_pla_abs,axis=3) /
                 np.mean(eta_tel_norm[:,:,None,:],axis=3))

m_snr = (np.mean(eta_sta_abs,axis=3)/
         np.mean((eta_pla_abs)**2.,axis=3))

m_join_metric = m_sta_int_leg * m_snr

i_ref = (np.int64(10), np.int64(7), np.int64(10))
rc_ref = m_con_raw[i_ref[0],i_ref[1],i_ref[2]]
lc_ref = m_sta_int_leg[i_ref[0],i_ref[1],i_ref[2]]
eta_pla_abs_ref =m_eta_pla_abs[i_ref[0],i_ref[1],i_ref[2]]
eta_pla_rel_ref = m_eta_pla_rel[i_ref[0],i_ref[1],i_ref[2]]
snr_ref = m_snr[i_ref[0],i_ref[1],i_ref[2]]

print('\n', "OD", '\t', "ID", '\t', "FPM", '\t', "LS thr", '\t',
      "s/p", '\t', "<s>/max(LS)", '\t',
      "eta p", '\t', "p/tel", '\t', "<s>/max(LS)*s/p**2", "s/p**2")

print('\n', dL_lst[i_ref[0]], '\t', 
      obs_lst[i_ref[1]], '\t', 
      mB_lst[i_ref[2]],'\t',
      np.round(thr_coro_data[i_ref[0],i_ref[1],0],3), '\t',
      np.round(rc_ref,5), '\t',
      np.round(lc_ref,5), '\t',
      np.round(eta_pla_abs_ref,5), '\t', np.round(eta_pla_rel_ref,5), '\t',
      np.round(m_join_metric[i_ref[0],i_ref[1],i_ref[2]]*1e6,5), '\t',
      np.round(snr_ref,5), '\n')

wc_ref = 0.
thr_ref = 0.75
thr = 0.
once = True

while True:
    
    # minimize
    # w_c = m_con_raw.copy()
    # w_c[m_con_raw <= wc_ref] = 1.
    # w_c = m_sta_int_leg.copy()
    # w_c[m_sta_int_leg <= wc_ref] = 1.
    # w_c = m_snr.copy()
    # w_c[m_snr <= wc_ref] = 1.
    w_c = m_join_metric.copy()
    w_c[m_join_metric <= wc_ref] = 1.

    if np.mean(w_c) == 1:
        break

    # obsolete, "in memoriam"
    # iko = thr_coro_data[:,:,0] <= thr
    # for i in range(len(mB_lst)):
    #     w_c[:,:,i][iko] = 1.

    i_min = np.unravel_index(np.argmin(w_c), w_c.shape)
    # i_min = np.unravel_index(np.argmax(w_c), w_c.shape) # maximize!
        
    thr = thr_coro_data[i_min[0],i_min[1],0]
    wc_ref = w_c[i_min[0],i_min[1],i_min[2]]
    
    # if (m_con_raw[i_min[0],i_min[1],i_min[2]] <=rc_ref
    #     and mB_lst[i_min[2]] <= 4.  and thr >= thr_ref):
    if (m_join_metric[i_min[0],i_min[1],i_min[2]] <= lc_ref*snr_ref
        and m_sta_int_leg[i_min[0],i_min[1],i_min[2]] <= lc_ref
        and mB_lst[i_min[2]] <= 4.): 
        # and thr >= thr_ref):
    # if (m_sta_int_leg[i_min[0],i_min[1],i_min[2]] <= lc_ref
    #     and m_con_raw[i_min[0],i_min[1],i_min[2]] <=rc_ref
    #     and m_snr[i_min[0],i_min[1],i_min[2]] <=snr_ref
    #     and mB_lst[i_min[2]] <= 4. and thr >= thr_ref):
    # if (m_snr[i_min[0],i_min[1],i_min[2]] <=snr_ref 
        # and m_con_raw[i_min[0],i_min[1],i_min[2]] <=rc_ref
        # and m_sta_int_leg[i_min[0],i_min[1],i_min[2]] <= lc_ref
        # and mB_lst[i_min[2]] <= 4. and thr >= thr_ref):
            
        print(dL_lst[i_min[0]], '\t',
              obs_lst[i_min[1]], '\t',
              mB_lst[i_min[2]], '\t',
              np.round(thr,3), '\t',
              np.round(m_con_raw[i_min[0],i_min[1],i_min[2]],5), '\t',
              np.round(m_sta_int_leg[i_min[0],i_min[1],i_min[2]],5), '\t',
              np.round(m_eta_pla_abs[i_min[0],i_min[1],i_min[2]],5), '\t',
              np.round(m_eta_pla_rel[i_min[0],i_min[1],i_min[2]],5), '\t',
              np.round(m_join_metric[i_min[0],i_min[1],i_min[2]]*1e6,5), '\t',
              np.round(m_snr[i_min[0],i_min[1],i_min[2]],5), '\t')

        if once:
            
            best = i_min
            print(best)
            once = False
            # break
            

#%%
# 0.9/0.37/4.0 <--> (10,7,10) pour étude 2025/092
# i_min = (np.int64(10), np.int64(7), np.int64(10))
i_min = best
print('\n', dL_lst[i_min[0]], '\t', 
      obs_lst[i_min[1]], '\t', 
      mB_lst[i_min[2]])    
# cd_min = np.log10(np.min(m_sta_int_leg[i_min[0],i_min[1],i_min[2]]))
# cd_max = np.log10(np.max(m_sta_int_leg[i_min[0],i_min[1],i_min[2]]))
mB_min = mB_lst[0]
mB_max = mB_lst[-1]
obs_min = obs_lst[0]
obs_max = obs_lst[-1]
dL_min = dL_lst[0]
dL_max = dL_lst[-1]

# plt.figure(0)
# plt.imshow(np.log10(mean_contrast[i_min[0],:,:]), vmin=cd_min, vmax=cd_max,
#            extent=(mB_min,mB_max,obs_max,obs_min),
#            aspect=(mB_max-mB_min)/(obs_max-obs_min),cmap='inferno')
# plt.xlabel("FPM [lam/D]")
# plt.ylabel("obstruction")
# plt.title(('lyot stop diam. : ' + str(np.round(dL_lst[i_min[0]],2))))
# plt.colorbar()
# plt.savefig(fdir_plot / ('rawContrastInCoroLyotAs2025WithIDVsFPM.pdf'))
# plt.show()

# plt.figure(1)
# plt.imshow(np.log10(mean_contrast[:,i_min[1],:]), vmin=cd_min, vmax=cd_max,
#            extent=(mB_min,mB_max,dL_max,dL_min),
#            aspect=(mB_max-mB_min)/(dL_max-dL_min),cmap='inferno')
# plt.xlabel("FPM [lam/D]")
# plt.ylabel("Lyot stop diam.")
# plt.title(('obstruction : ' + str(np.round(obs_lst[i_min[1]],2))))
# plt.colorbar()
# plt.savefig(fdir_plot / ('rawContrastInCoroLyotAs2025WithODVsFPM.pdf'))
# plt.show()

cd_min = -4 # for YJH & doc. 24102025
cd_min = np.log10(np.min(m_sta_int_leg[i_min[0],i_min[1],i_min[2]]))
cd_max = -3 # for YJH & doc. 24102025
cd_max =  np.log10(np.max(m_sta_int_leg[i_min[0],i_min[1],i_min[2]]))

plt.figure(2)
plt.imshow(np.log10(m_sta_int_leg[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
           extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno')
# plt.imshow(np.log10(mean_contrast[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
#            extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno')
plt.plot(obs_lst[i_min[1]], dL_lst[i_min[0]], marker='P', color='1')
plt.xlabel("Lyot stop Inner Diameter [D]")
plt.ylabel("Lyot stop Outer Diameter [D]")
plt.title(('FPM diameter: ' + str(np.round(mB_lst[i_min[2]],2))+f"$\lambda$/D @ $\lambda$={lamC*1e6:.1f}$\mu$m"),
          fontsize=12)
plt.colorbar(label=r"Intensity $I_S$ at 20 mas in log scale")
# plt.savefig(fdir_plot / ('starContrastMean_Is_coroLyot_89_35_39_ODVsID.pdf'),bbox_inches='tight')
plt.show()

cd_min = -2.4 # for YJH & doc. 24102025
cd_min = np.log10(np.min(m_snr[i_min[0],i_min[1],i_min[2]]))
cd_max = -1.4 # for YJH & doc. 24102025
cd_max =  np.log10(np.max(m_snr[i_min[0],i_min[1],i_min[2]]))
   
plt.figure(3)
plt.imshow(np.log10(m_snr[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
           extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno')
plt.plot(obs_lst[i_min[1]], dL_lst[i_min[0]], marker='P', color='1')
plt.xlabel("Lyot stop Inner Diameter [D]")
plt.ylabel("Lyot stop Outer Diameter [D]")
plt.title(('FPM diameter: ' + str(np.round(mB_lst[i_min[2]],2))+f"$\lambda$/D @ $\lambda$={lamC*1e6:.1f}$\mu$m"),
          fontsize=12)
plt.colorbar(label=r"$\eta_S$/$\eta_P^2$ at 20 mas in log scale")
# plt.savefig(fdir_plot / ('etaSToEtaPsquared_coroLyot_89_35_39_ODVsID.pdf'),bbox_inches='tight')
plt.show()


cd_min = -6 # for YJH & doc. 24102025
cd_min = np.log10(np.min(m_join_metric[i_min[0],i_min[1],i_min[2]]))
cd_max = -5 # for YJH & doc. 24102025
cd_max = np.log10(np.max(m_join_metric[i_min[0],i_min[1],i_min[2]]))

plt.figure(4)
plt.imshow(np.log10(m_join_metric[:,:,i_min[2]]), vmin=cd_min, vmax=cd_max,
           extent=(obs_min,obs_max,dL_max,dL_min,), cmap='inferno')
plt.plot(obs_lst[i_min[1]], dL_lst[i_min[0]], marker='P', color='1')
plt.xlabel("Lyot stop Inner Diameter [D]")
plt.ylabel("Lyot stop Outer Diameter [D]")
plt.title(('FPM diameter: ' + str(np.round(mB_lst[i_min[2]],2))+f"$\lambda$/D @ $\lambda$={lamC*1e6:.1f}$\mu$m"),
          fontsize=12)
plt.colorbar(label=r"$I_S$ x ($\eta_S$/$\eta_P^2$) at 20 mas in log scale")
# plt.savefig(fdir_plot / ('joinMetric-coroLyot_89_35_39_ODVsID.pdf'),bbox_inches='tight')
plt.show()

# search for a LS transmission limit guess
lc_ref = 1. # 
wc_ref = 0.
LS_t_ref = 0.
thr = 0.
once = True
lc_vs_t = []

while True:
    
    # minimize
    w_c = m_sta_int_leg.copy()
    w_c[m_sta_int_leg <= wc_ref] = 1.

    if np.mean(w_c) == 1:
        break

    iko = thr_coro_data[:,:,0] <= thr
    for i in range(len(mB_lst)):
        w_c[:,:,i][iko] = 1.

    i_min = np.unravel_index(np.argmin(w_c), w_c.shape)
    if i_min == (0,0,0):
        break
        
    thr = thr_coro_data[i_min[0],i_min[1],0]
    wc_ref = w_c[i_min[0],i_min[1],i_min[2]]
        
    print(dL_lst[i_min[0]], '\t',
          obs_lst[i_min[1]], '\t',
          mB_lst[i_min[2]], '\t',
          np.round(thr,3), '\t',
          np.round(m_con_raw[i_min[0],i_min[1],i_min[2]],5), '\t',
          np.round(m_sta_int_leg[i_min[0],i_min[1],i_min[2]],5), '\t',
          np.round(m_eta_pla_abs[i_min[0],i_min[1],i_min[2]],5), '\t',
          np.round(m_eta_pla_rel[i_min[0],i_min[1],i_min[2]],5), '\t',
          np.round(m_join_metric[i_min[0],i_min[1],i_min[2]]*1e6,5), '\t',
          np.round(m_snr[i_min[0],i_min[1],i_min[2]],5), '\t')

    lc_vs_t.append([[thr], [m_sta_int_leg[i_min[0],i_min[1],i_min[2]]]])
    
    if once:
        
        best = i_min
        print(best)
        once = False
            
lc_vs_t = np.array(lc_vs_t)

plt.figure(5)
plt.xlabel('LS transmission')#[$\lambda$/D]')
# plt.ylabel('intensity')
plt.ylabel('intensity (log)')
plt.yscale('log')
plt.plot(lc_vs_t[:,0], lc_vs_t[:,1], linestyle='-', marker='.')
plt.show()

