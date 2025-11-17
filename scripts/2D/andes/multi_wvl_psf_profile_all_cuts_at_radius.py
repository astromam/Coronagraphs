# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 08:32:52 2024

@author: asp
"""

# plot contrats and gain at given angular separation from contrast profile
# fits files produced by multi_wvl_psf_profile_cut.py

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from pathlib import Path
# from psf_profile import profile

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!


#%%
"""
### scaling
"""

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

as_oi = 20.
as_str = str(int(as_oi))
rW_mas = 7.
rW_str = str(int(rW_mas))

avoid_k = True

#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots
    fdir_dat = Path("D:/Andes/Data_corono/data/OPDs_PASSATA/OPD/WS/").resolve()
    # dir name where to find results and plots of a common script run date
   
# 

was_donow = '20251106160002'
fdir_res = fdir_res / was_donow
fdir_plt = fdir_plt / was_donow

# new ref 1 kHz 01/10/2025
# 0.89 0.35 3.9 20251001085445 ruane2018 YJH

# 0.89/0.35/3.9 + Lyot stop azimutal offset
# 20251107134453 2 pixels / 0.5% D 
# 20251107134538 4 pixels / 1.0% D 
# 20251107134615 8 pixels / 2.0% D 

# 0.89/0.35/3.9 + Lyot stop azimutal offset
# 20251106103751 2 pixels / 0.5% D
# 20251106103913 4 pixels / 1.0% D
# 20251106104005 8 pixels / 2.0% D

# 0.89/0.35/3.9 + Lyot stop vertical offset
# 20251104110334 2 pixels / 0.5% D
# 20251104110517 4 pixels / 1.0% D
# 20251104110556 8 pixels / 2.0% D

# 0.89/0.35/3.9 + fpm defocus
# 20251103103334  10 nm RMS
# 20251103103358  30 nm RMS
# 20251103103429  50 nm RMS
# 20251105104900  70 nm RMS

# 0.89/0.35/3.9 + offsets
# 20251013163430 1 mas
# 20251013163502 2 mas
# 20251013163616 3 mas
# 20251015134855 5 mas

# 0.89/0.35/3.9 + disp
# 20251014150042  5 mas/µ
# 20251014150115 10 mas/µ
# 20251015162408 15 mas/µ
# 20251020152303 20 mas/µ

# 0.89/0.35/3.9 + ncpa
# 20251016135714 10 nm RMS
# 20251016135818 30 nm RMS
# 20251016135837 50 nm RMS
# 20251020152206 70 nm RMS

# 0.89 0.35 3.9 + LS angular position error
# 20251017115345 0.5°
# 20251017115549 1.0°
# 20251017115610 1.5°
# 20251017120335 2.0°
# 20251017120423 3.0°
# 20251017120448 4.0°


#%%
"""
### working directory of the profile files
"""




#%%

# root = 'OPDs_PASSATA/OPD/WS/'
# p_dir=(root+'ocam2k','perfect')
# p_dir=(root+'alice','perfect')
# p_dir=(root+'cam_500us','perfect')

# root = 'OPDs_PASSATA/OPD/WS/500HzVarWS/'
# p_dir=(root+'20250704_105833.0',
#           root+'20250704_112007.0',
#           root+'20250704_114101.0',
#           root+'20250704_121550.0',
#           root+'20250704_123540.0',
#           root+'20250704_125507.0',
#           root+'20250704_131433.0',
#           root+'20250704_135325.0',
#           root+'20250704_141253.0',
#           root+'20250707_114321.0',
#           'perfect')
root = 'OPDs_PASSATA/OPD/WS/1kHzVarWS/'
p_dir=(root+'1',
          root+'2',
          root+'3',
          root+'4',
          root+'5',
          root+'6',
          root+'7',
          root+'8',
          root+'9',
          root+'10','perfect')


#%%

ptrn = ''

if True:
    
    file_lst = os.listdir(fdir_res / ('perfect'))

    psf_match = ['contrast_profile_', '_wonoise_', (as_str+'mas_'+rW_str)]
    nncr_cnt = [s for s in file_lst if all([m in s for m in psf_match])]

    # nncr_cnt = [x for x in file_lst if ('contrast_profile') in x]
    nncr_cnt = nncr_cnt[np.argmax(
        (lambda x:[len(i) for i in x])(nncr_cnt))]
    
    nncr_data  = fits.getdata(fdir_res / ('perfect') / nncr_cnt)
    nncr_head = fits.getheader(fdir_res / ('perfect') / nncr_cnt)
        
    lam_min = nncr_head['LMIN']
    lam_stp = nncr_head['LSTP']
    lam_itv = nncr_head['LITV']
    lam_lst = np.arange(lam_min,lam_min+(lam_itv+0.5)*lam_stp,lam_stp)
    if avoid_k:
        lam_lst = lam_lst[np.where(lam_lst < 1860e-9)]
    nL = len(lam_lst)
        
    nImg = nncr_head['NIMG']
    D = nncr_head['DIAM']
    mB = nncr_head['SFPM']
    pscale = nncr_head['PSCL']
    hlf_fov = nImg * pscale / 2.

    aS = np.arange(nImg//2)*(mas2rad * hlf_fov / (D *1e9) ) # angular sep.
    pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))
    contrast_nncr =  nncr_data[:,pos_as_oi]


once=True
for dir_nb in range(len(p_dir)):
        
    cur_dir = Path(os.path.basename(p_dir[dir_nb])).stem
    res_dir = (fdir_res / cur_dir)
    print(cur_dir, res_dir)
    
    if os.path.isdir(res_dir) and cur_dir!='perfect':
        
        file_lst = os.listdir(res_dir)

        psf_match = ['contrast_profile_', '_ao_corr_psf_',
                     (as_str+'mas_'+rW_str)]
        file_psf = [s for s in file_lst if all([m in s for m in psf_match])]

        cro_match = ['contrast_profile_', '_ao_corr_coro_psf_',
                     (as_str+'mas_'+rW_str)]
        file_cro = [s for s in file_lst if all(m in s for m in cro_match)]
     
        print("psf :", file_psf)
        print("psf coro :", file_cro)
        
        Int_D0_prf_avg = fits.getdata(res_dir / file_psf[0])
        Int_D_prf_avg  = fits.getdata(res_dir / file_cro[0])
        Int_D_prf_std  = fits.getdata(res_dir / file_cro[0], ext=1)
        Int_D_prf_min  = fits.getdata(res_dir / file_cro[0], ext=2)
        Int_D_prf_max  = fits.getdata(res_dir / file_cro[0], ext=3)

        head_psf = fits.getheader(res_dir / file_psf[0])
        
        if once:
            
            lamC = head_psf['LMBD']
            lam_min = head_psf['LMIN']
            lam_stp = head_psf['LSTP']
            lam_itv = head_psf['LITV']
            lam_lst = np.arange(lam_min,lam_min+(lam_itv+0.5)*lam_stp,lam_stp)
            if avoid_k:
                lam_lst = lam_lst[np.where(lam_lst < 1860e-9)]
            nL = len(lam_lst)
            
            nImg = head_psf['NIMG']
            D = head_psf['DIAM']
            mB = head_psf['SFPM']
            pscale = head_psf['PSCL']
            hlf_fov = nImg * pscale / 2.
       
            aS = np.arange(nImg//2)*(mas2rad * hlf_fov / (D *1e9) ) # ang. sep.
            pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))
             
            
            gain = np.zeros((nL, len(p_dir)))
            contrast = np.zeros((nL, len(p_dir)))
            cont_std = np.zeros((nL, len(p_dir)))
            cont_min = np.zeros((nL, len(p_dir)))
            cont_max = np.zeros((nL, len(p_dir)))
            maxi = np.zeros((nL, len(p_dir)))
            
            once = not(once)
    
        gain[:,dir_nb] = Int_D0_prf_avg[:,pos_as_oi]/Int_D_prf_avg[:,pos_as_oi]
        contrast[:,dir_nb] = Int_D_prf_avg[:,pos_as_oi]
        cont_std[:,dir_nb] = Int_D_prf_std[:,pos_as_oi]
        cont_min[:,dir_nb] = Int_D_prf_min[:,pos_as_oi]
        cont_max[:,dir_nb] = Int_D_prf_max[:,pos_as_oi]
        maxi[:,dir_nb] = Int_D0_prf_avg[:,pos_as_oi]

            
#%%
"""
plot profiles
"""
c_tab= plt.cm.inferno(np.linspace(0.15,.85,5)) # ['c','b','k','r','m','g','y']
# c_tab[2]=[0.,0.,0.,0.]

jqs = ['JQ1','JQ2','JQM','JQ3','JQ4']
jqs=jqs[::-1]
seeings = {'JQ1':'0.43"', 'JQ2':'0.58"','JQM':'0.65"','JQ3':'0.74"','JQ4':'1.06"'}

# mean_gain = np.array((5,len(gain)))
# mini_gain = mean_gain.copy()
# maxi_gain = mean_gain.copy()

# # plot of the azimutal average ratio profile
# plt.figure(1, (8, 4.5))
# plt.tight_layout()
# plt.xlabel(r'$\lambda$ (nm)')  #[$\lambda$/D]')
# plt.ylabel('Gain')
# plt.yscale('log')
# plt.title('Gain at '+as_str+' mas, windshake data')
# plt.grid(True)
# idx = []

# for i in range(5):
    
#     idx = [j for j,item in enumerate(np.array(p_dir)) if jqs[i] in item]
    
#     if idx != []:

#         g_avg = np.mean(gain[:,idx], axis=1)
#         g_std = np.std(gain[:,idx], axis=1)
#         plt.plot(lam_lst*1e9, g_avg, label=jqs[i]+': '+seeings[jqs[i]])
#         plt.fill_between(lam_lst*1e9, g_avg - g_std, g_avg + g_std, alpha=0.2)

# plt.ylim(0.5,1500)
# # plt.legend(fontsize='small')

# # fname_gain_25mas = (
# #     'all_windshake_data_gain_25mas_asRatioOf_lbd2D_ringAvgdPrfs_vs_wvl_'
# #     + os.path.basename(file_cro[0]).split('.')[0])
# #if actual lambda/D
# fname_gain_25mas = ('all_windshake_data_gain_' + ptrn +'_'+as_str+'mas_'+ was_donow)
# fpath_gain_25mas_pdf = fdir_plt / (fname_gain_25mas + '.pdf')
# plt.savefig(fpath_gain_25mas_pdf)


# plt.show()

# #%%

mean_contrast = np.array((5,len(contrast)))
mini_contrast = mean_contrast.copy()
maxi_contrast = mean_contrast.copy()

# plot of the azimutal average ratio profile
plt.figure(2, (8, 4.5))
plt.tight_layout()
plt.xlabel(r'Wavelength $\lambda$ [nm]')#[$\lambda$/D]')
plt.ylabel(f'Contrast @ {int(as_oi)} mas')
plt.yscale('log')
# tilt = np.round(offset*mas2rad,1)
# plt.title(f'ls off. 0.0°, tilt {tilt} mas, ncpa {int(ncpa)} nm rms, disp. {int(disp*1e-6)} mas/µm')
plt.title('Coronagraph configuration for YJH band')
plt.grid(True)
i = 2
if i==2:
# for i in range(5):
    
    i=2
    idx = i # [j for j,item in enumerate(np.array(p_dir)) if jqs[i] in item]
    
    if idx != []:
        
        idx = np.arange(len(p_dir)-1)
    
        c_avg = np.mean(contrast[:,idx], axis=1)
        c_std = np.max(cont_std[:,idx], axis=1)
        # c_10 = np.percentile(contrast[:,idx], 10, axis=1)
        # c_90 = np.percentile(contrast[:,idx], 90, axis=1)
        c_min = np.min(contrast[:,idx], axis=1)
        c_max = np.max(contrast[:,idx], axis=1)

        plt.plot(lam_lst*1e9, c_avg,
                 # label=seeings[jqs[i]] + ' (' + jqs[i] + ')',
                 color=c_tab[i])
        plt.fill_between(lam_lst*1e9, c_min, c_max, alpha=0.2, color=c_tab[i])
        
        if i!=0:
            
            c_std_avg = np.mean(cont_std[:,idx], axis=1)
            c_std_std = np.std(cont_std[:,idx], axis=1)
            idxm = idx.copy()
            plt.plot(lam_lst*1e9, np.mean(gain[:,idxm]*contrast[:,idxm],axis=1),
                      ls='--', color='black')

            fnm = (fdir_res / 
                   ('contrast_'+str(int(as_oi))+'mas_'+str(int(rW_mas))+'mas_'+
                    jqs[i]+'_'+was_donow+'.fits'))
            fits.writeto(fnm, np.array((c_avg,lam_lst)), head_psf, overwrite=True)
            fits.append(fnm, np.array((c_min, c_max)), overwrite=True)
            fits.append(fnm, np.array((np.mean(gain[:,idx], axis=1),
                                        np.std(gain[:,idx], axis=1))),
                        overwrite=True)         

plt.plot(lam_lst*1e9, contrast_nncr, color='black', ls=':')
# plt.plot(lam_lst*1e9, np.mean(gain[:,idxm]*contrast[:,idxm],axis=1),
#           ls='--', color='black')
# , label='median atm., no coro.')
bbox = dict(boxstyle='square', fc='w', alpha=0.75)
plt.text(int(lam_lst[-1]*1e9 - 250),2e-2,'---- atmo., no coro.', bbox=bbox)
plt.text(1000,9e-5,'... no atmo., coro.', bbox=bbox)

plt.ylim(1e-5,1e-1)
# plt.legend(title='Seeing',fontsize='small', loc=4)

fname_contrast_as_oi = (
    f'all_windshake_data_contrast_{int(as_oi)}mas_as_{int(rW_mas)}masRingAvgdPrfs_vs_wvl_' +
                    os.path.basename(file_cro[0]).split('.')[0])
fname_contrast_as_oi = (
    f'JQM_data_contrast_{int(as_oi)}mas_as_{int(rW_mas)}masRingAvgdPrfs_vs_wvl_' +
                    os.path.basename(file_cro[0]).split('.')[0])

# fname_contrast_as_oi = ('median_condition_data_contrast_' + as_str +'mas_'+ was_donow)

fpath_contrast_25mas_pdf = fdir_plt / (fname_contrast_as_oi + '.pdf')
plt.savefig(fpath_contrast_25mas_pdf, bbox_inches='tight', pad_inches=0.1)

plt.show()

