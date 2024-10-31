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
lam_c = 1600e-9  #  "central" lambda (of interest) in meters
# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

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
    was_donow = '20241031105129'
    # 20241009164421 , 20241007141137
    fdir_res = fdir_res / was_donow
    fdir_plt = fdir_plt / was_donow
    
elif user == 'Adrien':
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/results/').resolve()
    fdir_plt   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/plots/').resolve()
    fdir_dat = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/OPDs_PASSATA/OPD/WS/').resolve()

elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path( fdir_base / 'results' ).resolve()
    fdir_plt   = Path( fdir_base / 'plots' ).resolve()
    fdir_plt   = Path( fdir_base / 'data/OPDs_PASSATA/OPD/WS' ).resolve()

# 20241007141137 20241024101416 : 10mas/µ
# 20241009164421

# 20241007141603
# 20241009084549
# 20241009084821
# 20241009165029
# 20241010080453
# 20241010140433
# 20241010140448
# 20241011173119

#♠ ncpa nm rms with yjh 0mas/µ disp
# 20241025140035 10
# 20241028101351 20
# 20241028161658 30
# 20241029090244 40
# 20241029131750 50
# 20241029160850 60

# 20241030082457  100
# 20241030161024  200
# 20241031105129  300
# 20241030161219  400

#%%
"""
### working directory of the profile files
"""

p_dir=( 'OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',
        'OPDs_PASSATA/OPD/WS/JQ1/20240517_091216',
        'OPDs_PASSATA/OPD/WS/JQ1/20240517_100705',
        'OPDs_PASSATA/OPD/WS/JQ1/20240517_103452',
        'OPDs_PASSATA/OPD/WS/JQ1/20240517_105822',
        'OPDs_PASSATA/OPD/WS/JQ1/20240517_111658',
        'OPDs_PASSATA/OPD/WS/JQ1/20240517_113534',
        'OPDs_PASSATA/OPD/WS/JQ1/20240517_121247',
        'OPDs_PASSATA/OPD/WS/JQ2/20240517_181033',
        'OPDs_PASSATA/OPD/WS/JQ2/20240517_183817',
        'OPDs_PASSATA/OPD/WS/JQ2/20240517_190418',
        'OPDs_PASSATA/OPD/WS/JQ2/20240517_192251',
        'OPDs_PASSATA/OPD/WS/JQ2/20240517_194126',
        'OPDs_PASSATA/OPD/WS/JQ2/20240517_195959',
        'OPDs_PASSATA/OPD/WS/JQ2/20240517_201835',
        'OPDs_PASSATA/OPD/WS/JQ2/20240517_203708',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0',
        'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-001/JQ3/20240521_200540',
        'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-004/JQ3/20240521_181105',
        'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-002/JQ3/20240521_213115',
        'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-005/JQ3/20240521_222747',
        'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-003/JQ3/20240521_210334',
        'OPDs_PASSATA/OPD/WS/JQ3/20240527_190439/JQ3/20240527_190439',
        'OPDs_PASSATA/OPD/WS/JQ3/20240527_195648/JQ3/20240527_195648',
        'OPDs_PASSATA/OPD/WS/JQ3/20240527_204843/JQ3/20240527_204843',
        'OPDs_PASSATA/OPD/WS/JQ3/20240527_214043/JQ3/20240527_214043',
        'OPDs_PASSATA/OPD/WS/JQ3/20240527_223245/JQ3/20240527_223245',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_161522/JQ4/20240528_161522',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_163416/JQ4/20240528_163416',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_165414/JQ4/20240528_165414',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_171307/JQ4/20240528_171307',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_173157/JQ4/20240528_173157',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_175246/JQ4/20240528_175246',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_181244/JQ4/20240528_181244',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_183130/JQ4/20240528_183130',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_185018/JQ4/20240528_185018',
        'OPDs_PASSATA/OPD/WS/JQ4/20240528_190906/JQ4/20240528_190906')
p_dir=( 'OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
        'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0')


#%%

slc = False
if slc:
    ptrn = 'Lbd2D_'
else:
    ptrn = 'L0toD_'

# add noiseless plot
if True:
    
    file_lst = os.listdir(fdir_res / ('perfect'))
    
    nncr_cnt = [x for x in file_lst if ('contrast_profile') in x]
    nncr_cnt = nncr_cnt[np.argmax(
        (lambda x:[len(i) for i in x])(nncr_cnt))]
    
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
    
    aS = np.arange(nImg//2)*(mas2rad * 58.393 / (D *1e9) ) # angular sep.
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
            'contrast_profile_'+ptrn+cur_dir+'_ao_corr_psf_') in x]
        file_cro = [x for x in file_lst if (
            'contrast_profile_'+ptrn+cur_dir+'_ao_corr_coro_psf_') in x]
                
        print("psf :", file_psf)
        print("psf coro :", file_cro)
    
        Int_D0_prf_avg = fits.getdata(res_dir / file_psf[0])
        Int_D_prf_avg  = fits.getdata(res_dir / file_cro[0])
        Int_D_prf_std  = fits.getdata(res_dir / file_cro[0], ext=1)

        head_psf = fits.getheader(res_dir / file_psf[0])
        
        if once:
            
            lam_min = head_psf['LMIN']
            lam_stp = head_psf['LSTP']
            lam_itv = head_psf['LITV']
            lam_lst = np.arange(lam_min,lam_min+(lam_itv+1)*lam_stp,lam_stp)
            nL = len(lam_lst)
            
            nImg = head_psf['NIMG']
            D = head_psf['DIAM']
            mB = head_psf['SFPM']
            
            aS = np.arange(nImg//2)*(mas2rad * 58.393 / (D *1e9) ) # ang. sep.
            pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))
            
            gain = np.zeros((nL, len(p_dir)))
            contrast = np.zeros((nL, len(p_dir)))
            cont_std = np.zeros((nL, len(p_dir)))
            maxi = np.zeros((nL, len(p_dir)))
            
            once = not(once)
    
        gain[:,dir_nb] = Int_D0_prf_avg[:,pos_as_oi]/Int_D_prf_avg[:,pos_as_oi]
        contrast[:,dir_nb] = Int_D_prf_avg[:,pos_as_oi]
        cont_std[:,dir_nb] = Int_D_prf_std[:,pos_as_oi]
        maxi[:,dir_nb] = Int_D0_prf_avg[:,pos_as_oi]

            
#%%
"""
plot profiles
"""

jqs = ['JQ1','JQ2','JQM','JQ3','JQ4']
seings = {'JQ1':'0"43', 'JQ2':'0"58','JQM':'0"65','JQ3':'0"74','JQ4':'1"06'}

mean_gain = np.array((5,len(gain)))
mini_gain = mean_gain.copy()
maxi_gain = mean_gain.copy()

# plot of the azimutal average ratio profile
plt.figure(1, (8, 4.5))
plt.tight_layout()
plt.xlabel(r'$\lambda$ (nm)')  #[$\lambda$/D]')
plt.ylabel('Gain')
plt.yscale('log')
plt.title('Gain at '+as_str+' mas, windshake data')
plt.grid(True)
idx = []

for i in range(5):
    
    idx = [j for j,item in enumerate(np.array(p_dir)) if jqs[i] in item]
    
    if idx != []:

        g_avg = np.mean(gain[:,idx], axis=1)
        g_std = np.std(gain[:,idx], axis=1)
        plt.plot(lam_lst*1e9, g_avg, label=jqs[i]+': '+seings[jqs[i]])
        plt.fill_between(lam_lst*1e9, g_avg - g_std, g_avg + g_std, alpha=0.2)

plt.ylim(0.5,1500)
# plt.legend(fontsize='small')

# fname_gain_25mas = (
#     'all_windshake_data_gain_25mas_asRatioOf_lbd2D_ringAvgdPrfs_vs_wvl_'
#     + os.path.basename(file_cro[0]).split('.')[0])
#if actual lambda/D
fname_gain_25mas = ('all_windshake_data_gain_' + ptrn +'_'+as_str+'mas_'+ was_donow)
fpath_gain_25mas_svg = fdir_plt / (fname_gain_25mas + '.svg')
fpath_gain_25mas_pdf = fdir_plt / (fname_gain_25mas + '.pdf')
plt.savefig(fpath_gain_25mas_svg)
plt.savefig(fpath_gain_25mas_pdf)


plt.show()

#%%

# mean_contrast = np.array((5,len(contrast)))
# mini_contrast = mean_contrast.copy()
# maxi_contrast = mean_contrast.copy()

# plot of the azimutal average ratio profile
plt.figure(2, (8, 4.5))
plt.tight_layout()
plt.xlabel(r'$\lambda$ (nm)')#[$\lambda$/D]')
plt.ylabel('Contrast')
plt.yscale('log')
plt.title('contrast at '+as_str+' mas, windshake data')
plt.grid(True)
for i in range(5):
    
    idx = [j for j,item in enumerate(np.array(p_dir)) if jqs[i] in item]
    
    if idx != []:
    
        g_avg = np.mean(contrast[:,idx], axis=1)
        g_std = np.std(contrast[:,idx], axis=1)
        plt.plot(lam_lst*1e9, g_avg, label=jqs[i]+': '+seings[jqs[i]])
        plt.fill_between(lam_lst*1e9, g_avg - g_std, g_avg + g_std, alpha=0.2)
        
        if i==2:
            
            g_std_avg = np.mean(cont_std[:,idx], axis=1)
            g_std_std = np.std(cont_std[:,idx], axis=1)
            idxm = idx.copy()
            fits.writeto(fdir_res / ('contrast_JQM_'+was_donow+'.fits'), np.array((g_avg,g_std)),
                          overwrite=True)
            fits.append(fdir_res / ('contrast_JQM_'+was_donow+'.fits'), np.array((g_std_avg,g_std_std)),
                          overwrite=True)
            fits.append(fdir_res / ('contrast_JQM_'+was_donow+'.fits'), np.array((np.mean(gain[:,idx], axis=1),np.std(gain[:,idx], axis=1))),
                          overwrite=True)
            

plt.plot(lam_lst*1e9, contrast_nncr, color='black', label='coro, no atm.')
plt.plot(lam_lst*1e9, np.mean(gain[:,idxm]*contrast[:,idxm],axis=1),
          ls='--', color='black', label='median atm., no coro.')


plt.ylim(2e-5,2e-1)
plt.legend(fontsize='small', loc=2, ncol=2)

fname_contrast_25mas = ('all_windshake_data_contrast_25mas_asRatioOf_lbd2D_ringAvgdPrfs_vs_wvl_' +
                    os.path.basename(file_cro[0]).split('.')[0])
# fname_contrast_25mas = ('all_windshake_data_contrast_' + ptrn +'_'+as_str+'mas_'+ was_donow)
fpath_contrast_25mas_svg = fdir_plt / (fname_contrast_25mas + '.svg')
fpath_contrast_25mas_pdf = fdir_plt / (fname_contrast_25mas + '.pdf')
plt.savefig(fpath_contrast_25mas_svg)
plt.savefig(fpath_contrast_25mas_pdf)


plt.show()

