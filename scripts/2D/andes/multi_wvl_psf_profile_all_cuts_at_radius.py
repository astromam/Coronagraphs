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
# lam_c = 1600e-9  #  "central" lambda (of interest) in meters
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

was_donow = '20250305134456'
fdir_res = fdir_res / was_donow
fdir_plt = fdir_plt / was_donow

# 20250305134456 perfect coro 2nd order full elt pupil

# 20250121142815 yjh 0.9/0.37/4.0  0 disp,0 ncpa,0,tilt perfect: 20250122140220

# lyot stop horizontal offset in pixels
# 20250214100537 1
# 20250214100614 2
# 20250214100649 4
# 20250214100716 8

# lyot stop angular position error
# 20250214100748 0.5
# 20250214100828 1.0
# 20250214100857 2.0

# lyot stop vertical offset in pixels
# 20250130104616 1
# 20250130104747 2
# 20250130104759 4
# 20250130104809 8

# fpm defocus in nm RMS
# 20250210175500 10
# 20250210175517 20
# 20250210175531 30
# 20250210175546 40
# 20250210175606 50

# ncpa nm rms with yjh 0mas/µ disp
# 20250122183859 10
# 20250122184044 20
# 20250122184053 30
# 20250122184101 40
# 20250122184109 50

# disp + yjh
# 20250124135415 5
# 20250124135527 10
# 20250124135557 15

# offset/tilt in mas for psf to fpm
# 20250217154848 1
# 20250217155004 2
# 20250217155048 3
# 20250217155138 4
# 20250217155355 20

# 20241007141137 yjh 0 disp, 0 ncpa, 0 tilt
# 20241009164421 k 0 disp, 0 ncpa, 0 tilt

# offset in lambda central/D for psf to fpm + yjh

# 20241112095915 0.05
# 20241112100035 0.10
# 20241112100137 0.15
# 20241112100549 0.20
# 20241112100648 0.25
# 20241121163003 0.25 colineaire!
# 20241112175847 0.50
# 20241112175935 0.75
# 20241112180014 1.00

# disp + yjh
# 20241106094210  5 mas/µ
# 20241024101416 10 mas/µ
# 20241106094322 15 mas/µ

# ncpa nm rms with yjh 0mas/µ disp
#  20241105085056 10
#  20241105131506 20
#  20241105155837 30
#  20241105155850 40
#  20241105155859 50

# 20241118143408 30 nm ncp, 0.25 lamC/D et 5 mas/µ bw
# 20241121161633 30 nm ncpa, 0.25 lamC/D et 5 mas/µ bw colineaires


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
# p_dir=( 'OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
#         'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
#         'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
#         'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
#         'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
#         'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
#         'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
#         'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
#         'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0')

# p_dir=('OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_123456.0',)

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
            'contrast_profile_'+ptrn+cur_dir+'_ao_corr_psf_') in x]
        file_cro = [x for x in file_lst if (
            'contrast_profile_'+ptrn+cur_dir+'_ao_corr_coro_psf_') in x]
                
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
                   
            # disp = head_psf['DISP']
            # ncpa_rms = head_psf['NCPA']
            # fpm_dec = head_psf['FDEC']
            
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
# fpath_gain_25mas_svg = fdir_plt / (fname_gain_25mas + '.svg')
# fpath_gain_25mas_pdf = fdir_plt / (fname_gain_25mas + '.pdf')
# plt.savefig(fpath_gain_25mas_svg)
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
tilt = np.round(offset*mas2rad,1)
# plt.title(f'ls off. 0.0°, tilt {tilt} mas, ncpa {int(ncpa)} nm rms, disp. {int(disp*1e-6)} mas/µm')
plt.title('Coronagraph configuration for YJH band')
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
        plt.fill_between(lam_lst*1e9, c_min, c_max, alpha=0.2, color=c_tab[i])
        
        if i!=0:
            
            c_std_avg = np.mean(cont_std[:,idx], axis=1)
            c_std_std = np.std(cont_std[:,idx], axis=1)
            idxm = idx.copy()
            fnm = (fdir_res / ('contrast_'+jqs[i]+'_'+was_donow+'.fits'))
            fits.writeto(fnm, np.array((c_avg,c_std)), overwrite=True)
            fits.append(fnm, np.array((c_min, c_max)), overwrite=True)
            fits.append(fnm, np.array((np.mean(gain[:,idx], axis=1),
                                        np.std(gain[:,idx], axis=1))),
                        overwrite=True)
           

plt.plot(lam_lst*1e9, contrast_nncr, color='black', ls=':')
# plt.plot(lam_lst*1e9, np.mean(gain[:,idxm]*contrast[:,idxm],axis=1),
#           ls='--', color='black')
# , label='median atm., no coro.')
bbox = dict(boxstyle='square', fc='w', alpha=0.75)
# plt.text(1980,9e-2,'---- atmo., no coro.', bbox=bbox)
plt.text(1000,2e-5,'... no atmo., coro.', bbox=bbox)

plt.ylim(1e-5,1e-1)
plt.legend(title='Seeing',fontsize='small', loc=4)

fname_contrast_25mas = (
    f'all_windshake_data_contrast_{int(as_oi)}mas_as_10masRingAvgdPrfs_vs_wvl_' +
                    os.path.basename(file_cro[0]).split('.')[0])

# fname_contrast_25mas = ('median_condition_data_contrast_' + as_str +'mas_'+ was_donow)

fpath_contrast_25mas_svg = fdir_plt / (fname_contrast_25mas + '.svg')
fpath_contrast_25mas_pdf = fdir_plt / (fname_contrast_25mas + '.pdf')
fpath_contrast_25mas_png = fdir_plt / (fname_contrast_25mas + '.png')

plt.savefig(fpath_contrast_25mas_svg)
plt.savefig(fpath_contrast_25mas_pdf, bbox_inches='tight', pad_inches=0.1)
plt.savefig(fpath_contrast_25mas_png)


plt.show()

