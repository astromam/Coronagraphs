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

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!


#%%
"""
### scaling
"""
# lam_c = 1600e-9  #  "central" lambda (of interest) in meters
# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1./rad2mas

as_oi = 25.
as_str = str(int(as_oi))

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
   
# elif user == 'toto':

was_donow = '20250317172539'
fdir_res = fdir_res / was_donow
fdir_plt = fdir_plt / was_donow
rmsValue = str(int(100*0.9))

# 20250317172539 perfect 2nd order with lyotstop
# 20250318154752 perfect 2nd order with lyotstop + TT, JQ1 scaled 90%
# 20250318154814 perfect 2nd order with lyotstop + TT, JQ1 scaled 80%
# 20250318154832 perfect 2nd order with lyotstop + TT, JQ1 scaled 70%
# 20250319083152 perfect 2nd order with lyotstop + TT, JQ1 scaled 60%
# 20250319083224 perfect 2nd order with lyotstop + TT, JQ1 scaled 50%
# 20250317172808 perfect 2nd order with lyotstop + petalling corr

# 20250310155522 20250312085754 perfect coro 2nd order full elt pupil
# 20250312094239 psf no TT corr, coro TT corr
# 20250313135902 perfect 2nd order coro + petalling corr

# 20250121142815 yjh 0.9/0.37/4.0  0 disp,0 ncpa,0,tilt perfect: 20250122140220


#%%
"""
### working directory of the profile files
"""

p_dir=('OPDs_PASSATA/OPD/WS/ASI',)


#%%

# add noiseless plot
if False:
    
    file_lst = os.listdir(fdir_res / ('perfect'))
    
    nncr_cnt = [x for x in file_lst if ('wonoise_psf') in x]
    nncr_cnt = nncr_cnt[np.argmax(
        (lambda x:[len(i) for i in x])(nncr_cnt))]
    
    print('toto :', nncr_cnt)
    nncr_data  = fits.getdata(fdir_res / ('perfect') / nncr_cnt)
    nncr_head = fits.getheader(fdir_res / ('perfect') / nncr_cnt)
        
    lam_min = nncr_head['LMIN']
    lam_stp = nncr_head['LSTP']
    lam_itv = nncr_head['LITV']
    lam_lst = np.arange(lam_min,lam_min+(lam_itv+1)*lam_stp,lam_stp)
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
    
    if os.path.isdir(res_dir):
        
        file_lst = os.listdir(res_dir)
        file_psf = [x for x in file_lst if (
            'contrast_profile_'+cur_dir+'_ao_corr_psf_') in x]
        file_cro = [x for x in file_lst if (
            'contrast_profile_'+cur_dir+'_ao_corr_coro_psf_') in x]
                
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
            lam_lst = np.arange(lam_min,lam_min+(lam_itv+1)*lam_stp,lam_stp)
            nL = len(lam_lst)
            
            nImg = head_psf['NIMG']
            D = head_psf['DIAM']
            mB = head_psf['SFPM']
            pscale = head_psf['PSCL']
            hlf_fov = nImg * pscale / 2.
       
            aS = np.arange(nImg//2)*(mas2rad * hlf_fov / (D *1e9) ) # ang. sep.
            pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))
             
            try:
                disp = head_psf['DISP']
            except KeyError:
                disp=0
            
            try:
                ncpa = head_psf['NCPA']
            except KeyError:
                ncpa=0
        
            try:
                offset = head_psf['FDEC']
            except KeyError:
                offset=0
         
            try:
                ls_ape = head_psf['LSAE']
            except KeyError:
                ls_ape=0
                   
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

# fname_gain_25mas = ('all_windshake_data_gain_' + '_'+as_str+'mas_'+ was_donow)
# fpath_gain_25mas_pdf = fdir_plt / (fname_gain_25mas + '.pdf')
# plt.savefig(fpath_gain_25mas_pdf)

# plt.show()

#%%

mean_contrast = np.array((5,len(contrast)))
mini_contrast = mean_contrast.copy()
maxi_contrast = mean_contrast.copy()

# plot of the azimutal average ratio profile
plt.figure(2, (8, 4.5))
plt.tight_layout()
plt.xlabel(r'Wavelength $\lambda$ [nm]')#[$\lambda$/D]')
plt.ylabel(f'Contrast @ {int(as_oi)} mas')
plt.yscale('log')
tilt = np.round(offset*mas2rad,1)
# plt.title(f'ls off. 0.0°, tilt {tilt} mas, ncpa {int(ncpa)} nm rms, disp. {int(disp*1e-6)} mas/µm')
plt.title('Ideal coronagraph')
plt.grid(True)
for i in range(5):
    
    idx = [j for j,item in enumerate(np.array(p_dir)) if jqs[i] in item]
    
    if idx != []:
    
        c_avg = np.mean(contrast[:,idx], axis=1)
        c_std = np.max(cont_std[:,idx], axis=1)
        # c_10 = np.percentile(contrast[:,idx], 10, axis=1)
        # c_90 = np.percentile(contrast[:,idx], 90, axis=1)
        c_min = np.min(contrast[:,idx], axis=1)
        c_max = np.max(contrast[:,idx], axis=1)

        plt.plot(lam_lst*1e9, c_avg,
                 label=seeings[jqs[i]] + ' (' + jqs[i] + ')',
                 color=c_tab[i])

        # plt.plot(lam_lst*1e9, c_avg,
        #          label=rmsValue+' nm RMS')

        plt.fill_between(lam_lst*1e9, c_min, c_max, alpha=0.2, color=c_tab[i])
        
        if i!=0:
            
            c_std_avg = np.mean(cont_std[:,idx], axis=1)
            c_std_std = np.std(cont_std[:,idx], axis=1)
            idxm = idx.copy()
            fnm = (fdir_res / ('contrast_'+jqs[i]+'_'+was_donow+
                               '_'+as_str+'.fits'))
            fits.writeto(fnm, np.array((c_avg,c_std)), overwrite=True)
            fits.append(fnm, np.array((c_min, c_max)), overwrite=True)
            fits.append(fnm, np.array((np.mean(gain[:,idx], axis=1),
                                        np.std(gain[:,idx], axis=1))),
                        overwrite=True)
           
# plt.plot(lam_lst*1e9, contrast_nncr, color='black', ls=':')
# plt.plot(lam_lst*1e9, np.mean(gain[:,idxm]*contrast[:,idxm],axis=1),
#           ls='--', color='black')
# , label='median atm., no coro.')
# bbox = dict(boxstyle='square', fc='w', alpha=0.75)
# plt.text(1980,9e-2,'---- atmo., no coro.', bbox=bbox)
# plt.text(1000,2e-5,'... no atmo., coro.', bbox=bbox)
# plt.text(1000,2e-5,'... no atmo., no coro.', bbox=bbox)

plt.ylim(1e-5,1e-1)
plt.legend(title='AO residuals',fontsize='small', loc=4)

# fname_contrast_25mas = (
#     rmsValue+f'nmRMS_data_contrast_{int(as_oi)}mas_as_10masRingAvgdPrfs_vs_wvl_' +
#                     os.path.basename(file_cro[0]).split('.')[0])
# fname_contrast_25mas = (
#     f'JQs_data_contrast_{int(as_oi)}mas_as_10masRingAvgdPrfs_vs_wvl_' +
#                     os.path.basename(file_cro[0]).split('.')[0])

# fname_contrast_25mas = ('median_condition_data_contrast_' + as_str +'mas_'+ was_donow)

# fpath_contrast_25mas_pdf = fdir_plt / (fname_contrast_25mas + '.pdf')

# plt.savefig(fpath_contrast_25mas_pdf, bbox_inches='tight', pad_inches=0.1)

plt.show()

