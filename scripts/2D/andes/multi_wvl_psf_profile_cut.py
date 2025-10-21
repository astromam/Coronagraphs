# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:59:22 2024

@author: asp, mndiaye, asimonnin
"""

# compute contrast profiles vs angular separation for multiple wavelength

#%%
"""
### Initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from pathlib import Path
# from psf_profile import profile

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!

avoid_k = True


#%%
"""
### scaling
"""
# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# angular separation and ring width of interest in mas
as_oi = 20.
rW_mas = 7.


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots
    
elif user == 'Adrien':
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/results/').resolve()
    fdir_plt   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/plots/').resolve()

elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path( fdir_base / 'results' ).resolve()
    fdir_plt   = Path( fdir_base / 'plots' ).resolve()

was_donow = '20251020152303'

fdir_res = fdir_res / was_donow
fdir_plt = fdir_plt / was_donow

# new ref 1 kHz 01/10/2025
# 0.89 0.35 3.9 20251001085445 ruane2018 YJH

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

# 0.89 0.35 3.9 alice  20251007190203
# 0.90 0.37 4.0 alice  20251007190322
# 0.89 0.35 3.9 ocam2k 20251007190218
# 0.90 0.37 4.0 ocam2k 20251007190259
# 0.89 0.35 3.9 cam_500us 20251013105528
# 0.9 0.37 4.0  cam_500us 20251013105414

# 0.96 0.34 3.3 20251001154101 ruane2018 YJH raw contrast optimization
# 0.96 0.3 3.3 20251001100920 ruane2018 JH raw contrast optimization
# 0.89 0.35 3.9 20251001085445 ruane2018 JH/YJH 10mas@25mas! raw contrast optimization

# 0.89 0.35 3.7 20250929170957 ruane2018 YJH raw contrast optimization
# 0.87 0.3 3.9 20250929171546 ruane2018 JH raw contrast optimization

# new params exploration follow. ruane2018 06/2025
# photometric aperture 7µ, 1kHz data
# J     4.6e-05     0.85    0.36   3.3 0.66501  20250618164513
# H     0.000392    0.86    0.3    4.1 0.72769  20250618164620
# JH    0.000428    0.86    0.3    4.0 0.72769  20250618164704
# YJH   0.000628    0.86    0.37   4.0 0.67608  20250618164749

# 500Hz 0.86 0.31 4.0 20250709155129
#  1kHz 0.86 0.31 4.0 20250709155051
# 20250711155826 0.89/0.35/3.9 1.kHz
# 20250711155801 0.89/0.35/3.9 .5kHz

# 20250711155932 0.88/0.3/4.0 1.kHz
# 20250711155911

# 20250711160041 0.88/0.3/3.5 1.kHz
# 20250711160019

#%%
# 500 Hz    0.9/0.37/4.0    20250604120819 lam_itv = 80
# 1 kH0 Hz  0.9/0.37/4.0    20250602181544 lam_itv = 80
# 20250122140220 "perfect"

# 500 Hz    0.9/0.37/4.0    20250708162615 lam_itv = 50
# 1 kH0 Hz  0.9/0.37/4.0    20250708171059 lam_itv = 50

# 500 Hz    0.9/0.37/4.0    20250610135805 lam_itv = 50
# 1 kH0 Hz  0.9/0.37/4.0    20250610140016 lam_itv = 50
# 20250611134316 "perfect"

# 0.804 	 0.000341188 	 [0.92 0.36 3.9 ]    20250523162033 20250523162130
# 0.811 	 0.000354267 	 [0.92 0.35 3.8 ]    20250523162217 20250523162247
# 0.824 	 0.000380266 	 [0.93 0.36 3.6 ]    20250523163019 20250523162955 
# 0.826 	 0.000388391 	 [0.92 0.33 3.5 ]    20250523163116 20250523163139

# RIZ
# 20250521172353 JQM 05/2025  500Hz
# 20250521172457 JQM 05/2025 1000Hz
# YJH
# 20250522161347 JQM 05/2025  500Hz
# 20250522160547 JQM 05/2025 1000Hz

#%%
# 20250506131205 coro UJH2024 @ 1 kHz
# 20250409111209 perfect 2nd order with lyotstop + TT - strehl correction
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

#%%
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

# 20241009164421 k 0 0.96/0.3/4.5  0 disp,0 ncpa,0 tilt perfect: 20241021142027

# erroneous
# 20241007141137 yjh 0.9/0.37/4.5! 0 disp,0 ncpa,0,tilt perfect: 20241016110318

# lyot stop angular position error
# 20241204133939 0.5
# 20241204134054 1.0
# 20241204134125 1.5
# 20241204134159 2.0
# 20241205132015 30.

# offset/tilt in lambda central/D for psf to fpm + yjh

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

#yjh et hk avec erreur dans disp: 80 est 80/2.pi
# 20241007141603
# 20241009084549
# 20241009084821
# 20241009165029
# 20241010080453
# 20241010140433
# 20241010140448
# 20241011173119

# ncpa nm rms with yjh 0mas/µ disp
# 20241105085056 10
# 20241105131506 20
# 20241105155837 30
# 20241105155850 40
# 20241105155859 50

# 20241118143408 30 nm ncpa, 0.25 lamC/D et 5 mas/µ bw
# 20241121161633 30 nm ncpa, 0.25 lamC/D et 5 mas/µ bw colineaires


#%%
"""
### working directory of the OPD files
"""
# New set of OPDs from PASSATA 

# opds_dir=('OPDs_PASSATA/OPD/20231124_090126.0/',
#               'OPDs_PASSATA/OPD/20231122_142204.0/',
#               'OPDs_PASSATA/OPD/20240227_234849-007/20240227_234849.0',
#               'OPDs_PASSATA/OPD/20240228_053027-001/20240228_053027.0',
#               'OPDs_PASSATA/OPD/20240302_000411-003/20240302_000411.0',
#               'OPDs_PASSATA/OPD/20240313_133532-004/20240313_133532.0',
#               'OPDs_PASSATA/OPD/20240228_112033-002/20240228_112033.0')
#%%
# opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',
#           'OPDs_PASSATA/OPD/WS/JQ1/20240517_091216',
#           'OPDs_PASSATA/OPD/WS/JQ1/20240517_100705',
#           'OPDs_PASSATA/OPD/WS/JQ1/20240517_103452',
#           'OPDs_PASSATA/OPD/WS/JQ1/20240517_105822',
#           'OPDs_PASSATA/OPD/WS/JQ1/20240517_111658',
#           'OPDs_PASSATA/OPD/WS/JQ1/20240517_113534',
#           'OPDs_PASSATA/OPD/WS/JQ1/20240517_121247',
#           'OPDs_PASSATA/OPD/WS/JQ2/20240517_181033',
#           'OPDs_PASSATA/OPD/WS/JQ2/20240517_183817',
#           'OPDs_PASSATA/OPD/WS/JQ2/20240517_190418',
#           'OPDs_PASSATA/OPD/WS/JQ2/20240517_192251',
#           'OPDs_PASSATA/OPD/WS/JQ2/20240517_194126',
#           'OPDs_PASSATA/OPD/WS/JQ2/20240517_195959',
#           'OPDs_PASSATA/OPD/WS/JQ2/20240517_201835',
#           'OPDs_PASSATA/OPD/WS/JQ2/20240517_203708',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0')
#,
          # 'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-001/JQ3/20240521_200540',
          # 'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-004/JQ3/20240521_181105',
          # 'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-002/JQ3/20240521_213115',
          # 'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-005/JQ3/20240521_222747',
          # 'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-003/JQ3/20240521_210334',
          # 'OPDs_PASSATA/OPD/WS/JQ3/20240527_190439/JQ3/20240527_190439',
          # 'OPDs_PASSATA/OPD/WS/JQ3/20240527_195648/JQ3/20240527_195648',
          # 'OPDs_PASSATA/OPD/WS/JQ3/20240527_204843/JQ3/20240527_204843',
          # 'OPDs_PASSATA/OPD/WS/JQ3/20240527_214043/JQ3/20240527_214043',
          # 'OPDs_PASSATA/OPD/WS/JQ3/20240527_223245/JQ3/20240527_223245',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_161522/JQ4/20240528_161522',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_163416/JQ4/20240528_163416',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_165414/JQ4/20240528_165414',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_171307/JQ4/20240528_171307',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_173157/JQ4/20240528_173157',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_175246/JQ4/20240528_175246',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_181244/JQ4/20240528_181244',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_183130/JQ4/20240528_183130',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_185018/JQ4/20240528_185018',
          # 'OPDs_PASSATA/OPD/WS/JQ4/20240528_190906/JQ4/20240528_190906')

# opds_dir=('OPDs_PASSATA/OPD/WS/ASI',)


#,
#          'perfect')
# opds_dir=('OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0',
#           'perfect',)
# opds_dir=('OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0','perfect',)
# opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822','perfect',)

# remove .split[0] at opd_set = ...
# 20241016110318 diam_0.9-obst_0.37_FPM_lcToD_4.5
# 20241016111736 diam_0.96-obst_0.3_FPM_lcToD_4.5
# opds_dir=('perfect',)
#%%
# opds_dir=('OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_123456.0',)
# opds_dir=('OPDs_PASSATA/OPD/WS/1kHz/02042025/20250325_145800.0_phase_screens-001/20250325_145800.0','perfect')
# opds_dir=('OPDs_PASSATA/OPD/WS/500Hz/20250508_163051.0_phase_screens/20250508_163051.0_oaCUBEs','perfect')
#%%
# root = 'OPDs_PASSATA/OPD/WS/'
# opds_dir=(root+'ocam2k','perfect')
# opds_dir=(root+'alice','perfect')
# opds_dir=(root+'cam_500us','perfect')

# root = 'OPDs_PASSATA/OPD/WS/500HzVarWS/'
# opds_dir=(root+'20250704_105833.0',
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
opds_dir=(root+'1',
          root+'2',
          root+'3',
          root+'4',
          root+'5',
          root+'6',
          root+'7',
          root+'8',
          root+'9',
          root+'10','perfect')
# 1 kHz
# root_1kHz = 'OPDs_PASSATA/OPD/WS/1kHz/'
# opds_dir=(root_1kHz+'20250325_145800.0_phase_screens/20250325_145800.0',
#           root_1kHz+'20250522_150800.0_phase_screens/20250522_150800.0',
#           root_1kHz+'20250522_154952.0_phase_screens/20250522_154952.0_phase_screens',
#           root_1kHz+'20250522_163144.0_phase_screens/20250522_163144.0_phase_screens',
#           root_1kHz+'20250522_171340.0_phase_screens/20250522_171340.0_phase_screens',
#           root_1kHz+'20250523_120216.0_phase_screens/20250523_120216.0_phase_screens',
#           root_1kHz+'20250523_124400.0_phase_screens/20250523_124400.0_phase_screens',
#           root_1kHz+'20250523_140738.0_phase_screens/20250523_140738.0_phase_screens',
#           root_1kHz+'20250523_144927.0_phase_screens/20250523_144927.0_phase_screens',
#           root_1kHz+'20250527_145633.0_phase_screens/20250527_145633.0_phase_screens',
#           'perfect')

# 500 Hz
# root_500Hz = 'OPDs_PASSATA/OPD/WS/500Hz/'
# opds_dir=(root_500Hz + '20250508_163051.0_phase_screens/20250508_163051.0_oaCUBEs',
#           root_500Hz + '20250521_145220.0_phase_screens/20250521_145220.0_oaCUBEs',
#           root_500Hz + '20250521_151332.0_phase_screens/20250521_151332.0_oaCUBEs',
#           root_500Hz + '20250521_153444.0_phase_screens/20250521_153444.0_oaCUBEs',
#           root_500Hz + '20250521_155558.0_phase_screens/20250521_155558.0_oaCUBEs',
#           root_500Hz + '20250521_161711.0_phase_screens/20250521_161711.0_oaCUBEs',
#           root_500Hz + '20250521_163823.0_phase_screens/20250521_163823.0_oaCUBEs',
#           root_500Hz + '20250521_172051.0_phase_screens/20250521_172051.0_oaCUBEs',
#           root_500Hz + '20250521_174203.0_phase_screens/20250521_174203.0_oaCUBEs',
#           root_500Hz + '20250527_122032.0_phase_screens/20250527_122032.0_oaCUBEs',
#           'perfect')

#%%

# """
# ### Read psf files & create profiles
# """

for dir_nb in range(len(opds_dir)):
        
    opd_set = os.path.basename(fdir_res / opds_dir[dir_nb] ).split('.')[0]

    print(fdir_res / opd_set)
    
    cur_dir = Path(os.path.basename(opds_dir[dir_nb])).stem
    res_dir = (fdir_res / cur_dir)
    print(cur_dir, res_dir)
    
    if os.path.isdir(res_dir):
        
        os.makedirs(fdir_plt / opd_set, exist_ok=True)

        file_lst = os.listdir(res_dir)

        if opd_set != 'perfect':
            
            file_psf = [x for x in file_lst if 'ao_corr_psf_' in x]
            file_cro = [x for x in file_lst if 'ao_corr_coro_psf_' in x]
        
        else:
            
            file_psf = [x for x in file_lst if 'wonoise_psf_' in x]
            file_cro = [x for x in file_lst if 'wonoise_coro_psf_' in x]
            # file_elt = [x for x in file_lst if 'wo_noise_psf_elt_pupil' in x]
               
        print("psf :", file_psf)
        print("psf coro :", file_cro)

        # if opd_set == 'perfect':
            
            # file_elt = file_elt[np.argmin(
            #     (lambda x:[len(i) for i in x])(file_elt))]
            # base_elt = os.path.basename(file_elt).split('.')[0]
            # Int_elt_avg  = fits.getdata(fdir_res / opd_set / file_elt)

        
        file_psf = file_psf[np.argmin(
            (lambda x:[len(i) for i in x])(file_psf))]
        file_cro = file_cro[np.argmin(
            (lambda x:[len(i) for i in x])(file_cro))]

        base_psf = os.path.basename(file_psf).split('.')[0]
        base_cro = os.path.basename(file_cro).split('.')[0]
        
        Int_D0_psf_avg = fits.getdata(fdir_res / opd_set / file_psf)
        Int_D_psf_avg  = fits.getdata(fdir_res / opd_set / file_cro)
        
        head_psf = fits.getheader(fdir_res / opd_set / file_psf)
        
        lam_min = head_psf['LMIN']
        lam_stp = head_psf['LSTP']
        lam_itv = head_psf['LITV']
        lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)
        if avoid_k:
            lam_lst = lam_lst[np.where(lam_lst < 1860e-9)]
        lmx = str(int(np.rint(lam_lst[-1]*1e9)))
        lmn = str(int(np.rint(lam_lst[0]*1e9)))
        nL = len(lam_lst)

        if np.median(lam_lst) > 1e-6:
            wvl = 'YJH'
        else:
            wvl = 'RIZ'
        
        nImg = head_psf['NIMG']
        pscale = head_psf['PSCL']
        D = head_psf['DIAM']
        mB = head_psf['SFPM'] # - 0.5
        lam_c = head_psf['LMBD']
        lamCD2mas = (lam_c / D) * mas2rad

        # print(lam_c)
        hlf_fov = nImg * pscale / 2.
        # stackoveflow...
        a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
        b_ = a_.copy()
        aa, bb = np.meshgrid(a_, b_)
        rad_pix = np.sqrt(aa**2 + bb**2)
        rad_mas = rad_pix * (mas2rad * hlf_fov / (D * 1e9) )
        
        #  angular separation
        aS = np.arange(nImg//2) * (mas2rad * hlf_fov / (D * 1e9) )
        
        # if lambda of interest / D
        # rW_mas = 10.  #  (lam_c / D) * mas2rad
        
        ratio_prf = np.zeros([nL, nImg//2])
        Int_D0_prf_avg = ratio_prf.copy()
        Int_D_prf_avg = ratio_prf.copy()
        Int_D0_prf_std = ratio_prf.copy()
        Int_D_prf_std = ratio_prf.copy()
        Int_D0_prf_min = ratio_prf.copy()
        Int_D_prf_min = ratio_prf.copy()
        Int_D0_prf_max = ratio_prf.copy()
        Int_D_prf_max = ratio_prf.copy()

        Int_elt_prf_avg = ratio_prf.copy()
        Int_elt_prf_std = ratio_prf.copy()
        Int_elt_prf_min = ratio_prf.copy()
        Int_elt_prf_max = ratio_prf.copy()

        for i in range(nL):
            
            # if slc:
            #     rW_mas = rW_mas # 10.  #  (lam_lst[i] / D) * mas2rad
            
            for p in range(nImg//2):
                
                ring_val = np.where(np.abs(rad_mas-aS[p])<=rW_mas/2)
                Int_D0_prf_avg[i,p] = np.mean(Int_D0_psf_avg[i,:,:][ring_val])
                Int_D_prf_avg[i,p] = np.mean(Int_D_psf_avg[i,:,:][ring_val])
                Int_D0_prf_std[i,p] = np.std(Int_D0_psf_avg[i,:,:][ring_val])
                Int_D_prf_std[i,p] = np.std(Int_D_psf_avg[i,:,:][ring_val])
                Int_D0_prf_min[i,p] = np.min(Int_D0_psf_avg[i,:,:][ring_val])
                Int_D_prf_min[i,p] = np.min(Int_D_psf_avg[i,:,:][ring_val])
                Int_D0_prf_max[i,p] = np.max(Int_D0_psf_avg[i,:,:][ring_val])
                Int_D_prf_max[i,p] = np.max(Int_D_psf_avg[i,:,:][ring_val])
                
                # if opd_set == 'perfect':
                    
                #     nrm = np.max(Int_elt_avg[i,:,:])
                #     Int_elt_prf_avg[i,p] = np.mean(Int_elt_avg[i,:,:][ring_val])/nrm
                #     Int_elt_prf_std[i,p] = np.std(Int_elt_avg[i,:,:][ring_val])/nrm
                #     Int_elt_prf_min[i,p] = np.min(Int_elt_avg[i,:,:][ring_val])/nrm
                #     Int_elt_prf_max[i,p] = np.max(Int_elt_avg[i,:,:][ring_val])/nrm

        # if slc:
            
        #     fname_psf_contrast = ('contrast_profile_Lbd2D_' + opd_set + '_' +
        #                           base_psf +'_'+lmn+'_'+lmx+'.fits')
        #     fname_cro_contrast = ('contrast_profile_Lbd2D_' + opd_set + '_' +
        #                           base_cro +'_'+lmn+'_'+lmx+ '.fits')
    
        # else:
            
        fname_psf_contrast = ('contrast_profile_L0toD_' + opd_set + '_' +
                              base_psf +'_'+
                              str(int(as_oi))+'mas_'+str(int(rW_mas))+'mas_'+
                              lmn+'_'+lmx+ 'nm.fits')
        fname_cro_contrast = ('contrast_profile_L0toD_' + opd_set + '_' +
                              base_cro +'_'+
                              str(int(as_oi))+'mas_'+str(int(rW_mas))+'mas_'+
                              lmn+'_'+lmx+ 'nm.fits')
        # if opd_set == 'perfect':

            # fname_elt_contrast = ('contrast_profile_L0toD_' + opd_set + '_' +
            #                   base_elt +'_'+
            #                   str(int(as_oi))+'mas_'+str(int(rW_mas))+'mas_'+
            #                   lmn+'_'+lmx+ 'nm.fits')
        
        fpath_psf_contrast =  fdir_res / opd_set / fname_psf_contrast 
        fits.writeto(fpath_psf_contrast, Int_D0_prf_avg, head_psf,
                     overwrite=True)
        fits.append(fpath_psf_contrast, Int_D0_prf_std, head_psf)
        fits.append(fpath_psf_contrast, Int_D0_prf_min, head_psf)
        fits.append(fpath_psf_contrast, Int_D0_prf_max, head_psf)
        
        fpath_cro_contrast =  fdir_res / opd_set / fname_cro_contrast 
        fits.writeto(fpath_cro_contrast, Int_D_prf_avg, head_psf,
                     overwrite=True)
        fits.append(fpath_cro_contrast, Int_D_prf_std, head_psf)
        fits.append(fpath_cro_contrast, Int_D_prf_min, head_psf)
        fits.append(fpath_cro_contrast, Int_D_prf_max, head_psf)

        prf_as_oi_mas = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))
       
        # if opd_set == 'perfect':
    
        #     fpath_elt_contrast =  fdir_res / opd_set / fname_elt_contrast 
        #     fits.writeto(fpath_elt_contrast, Int_elt_prf_avg, head_psf,
        #                  overwrite=True)
        #     fits.append(fpath_elt_contrast, Int_elt_prf_std, head_psf)
        #     fits.append(fpath_elt_contrast, Int_elt_prf_min, head_psf)
        #     fits.append(fpath_elt_contrast, Int_elt_prf_max, head_psf)

        prf_as_oi_mas = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))
        
        if opd_set != 'perfect':
            
            file_psf_prf = [x for x in file_lst if 'ao_corr_psf_profile_' in x]
            file_cro_prf = [x for x in file_lst
                            if 'ao_corr_coro_psf_profile_' in x]
    
            if opd_set == 'perfect':
                
                file_psf_prf = [x for x in file_lst if 'wonoise_psf_profile_' in x]
                file_cro_prf = [x for x in file_lst
                                if 'wonoise_coro_psf_profile_' in x]
    
            
            print("psf profile:", file_psf_prf)
            print("psf coro profile:", file_cro_prf)
        
            Int_D0_psf_prf = fits.getdata(fdir_res / opd_set / file_psf_prf[0])
            Int_D_psf_prf  = fits.getdata(fdir_res / opd_set / file_cro_prf[0])
        
                
        #%%
        """
        plot profiles
        """
        
        # # plot of the gain vs wvl
        # plt.figure(1, (8, 4.5))
        # plt.tight_layout()
        # plt.xlabel('wavelength (nm)')#[$\lambda$/D]')
        # plt.ylabel('gain (log)')
        # plt.yscale('log')
        # plt.title('gain at 25 mas vs wvl')
        # plt.grid(True)
        # plt.ylim(9e-1, 2e3)
    
        # g_val = Int_D0_prf_avg[:,prf_as_oi_mas]/Int_D_prf_avg[:,prf_as_oi_mas]
        # print(np.min(g_val), np.max(g_val))    
        # plt.plot(lam_lst*1e9, g_val)
        
        # if slc:
        #     fname_gain_as_oi_mas = (
        #         'gain_'+str(int(as_oi))+'mas_Lbd2D_' + base_cro)
        # else:
        #     fname_gain_as_oi_mas = (
        #         'gain_'+str(int(as_oi))+'mas_L0toD_' + base_cro)
        
        # fpath_gain_as_oi_mas_svg = (fdir_plt / opd_set /
        #                             (fname_gain_as_oi_mas + '.svg'))
        # fpath_gain_as_oi_mas_pdf = (fdir_plt / opd_set /
        #                             (fname_gain_as_oi_mas + '.pdf'))
        # plt.savefig(fpath_gain_as_oi_mas_svg)
        # plt.savefig(fpath_gain_as_oi_mas_pdf)
        # plt.show()
        
        #%%
        # fix mask radius before plots
        
        # # plot of the contrast vs radial distance
        # colors = plt.cm.rainbow(np.linspace(0,1,nL))
        # plt.figure(2, (8, 4.5))
        # plt.tight_layout()
        # plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
        # plt.ylabel('contrast')
        # plt.yscale('log')
        # plt.title('contrasts(wvl) vs radial distance')
        # plt.grid(True)
        # plt.ylim(1e-5, 2e0)
    
        # for i in range(nL):
        
        #     # AO corrected psf 
        #     plt.plot(aS, Int_D0_prf_avg[i,:], color=colors[i], alpha=0.5)
            
        #     # AO corrected coronagraphic psf 
        #     plt.plot(aS, Int_D_prf_avg[i,:], label=str(int(lam_lst[i]*1e9))+'nm',
        #               color=colors[i])
            
        # # Focal plane mask boundary
        # x = np.arange(0.0, mB/2, 0.01)
        # plt.axvline(as_oi, color='k', ls='--')
        # plt.legend(fontsize='xx-small', ncols=5)
        # # Focal plane mask grey area
        # plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lam_c/rad2mas, color='gray',
        #                   alpha=0.3)
        
        # if slc:
        #     fname_contrast_profile = ('contrast_profile_Lbd2D_' + base_cro)
            
        # else:
        #     fname_contrast_profile = ('contrast_profile_L0toD_' + base_cro)
        # fpath_contrast_profile_svg = (fdir_plt / opd_set /
        #                               (fname_contrast_profile + '.svg'))
        # fpath_contrast_profile_pdf = (fdir_plt / opd_set /
        #                               (fname_contrast_profile + '.pdf'))
        # plt.savefig(fpath_contrast_profile_svg)
        # plt.savefig(fpath_contrast_profile_pdf)
        # plt.show()

        #%%
        # index of wvl to display
        iD = [0,nL//4-1,nL//2-1,nL*3//4-1,nL-1]

        if opd_set != 'perfect':
            
            # plot of the radial profiles
            colors = plt.cm.rainbow(np.linspace(0,1,nL))
            plt.figure(3, (8, 4.5))
            plt.tight_layout()
            plt.ylim(1e-5, 2e0)
            plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
            plt.ylabel('Intensity in log scale')
            plt.yscale('log')
            plt.title(f'Coronagraph configuration in {wvl} band')
            plt.grid(True)
           
            for i in range(5):
            # for i in range(0,nL,2):
            # for i in range(6):

                # AO corrected psf 
                # plt.plot(aS, Int_D0_psf_prf[iD[i],1,:], color=colors[iD[i]], alpha=0.5,
                #           ls='--')
                
                # AO corrected coronagraphic psf 
                plt.plot(aS, Int_D_psf_prf[iD[i],1,:],
                         label=str(int(np.rint(lam_lst[iD[i]]*1e9)))+'nm',
                          color=colors[iD[i]])
                
            # Focal plane mask boundary
            x = np.arange(0.0, mB/2, 0.01)
            plt.axvline(as_oi, color='k', ls='--')
            plt.legend(fontsize='xx-small', ncols=6)
            # Focal plane mask grey area
            plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lam_c/rad2mas, color='gray',
                              alpha=0.3)
            
            fname_contrast_profile = ('intensities_profiles_' + base_cro)
            fpath_contrast_profile_svg = (fdir_plt / opd_set /
                                          (fname_contrast_profile +'_'+lmn+'_'+lmx+ '.svg'))
            fpath_contrast_profile_pdf = (fdir_plt / opd_set /
                                          (fname_contrast_profile +'_'+lmn+'_'+lmx+ '.pdf'))
            plt.savefig(fpath_contrast_profile_svg)
            plt.savefig(fpath_contrast_profile_pdf, bbox_inches='tight', pad_inches=0.1)
            plt.show()
    
    
            # plot of the radial profiles
            colors = plt.cm.rainbow(np.linspace(0,1,nL))
            plt.figure(4, (8, 8))
            # plt.title('intensity@1600nm with/without coro. vs radial distance')
            plt.tight_layout()
            plt.xlim(1e0,1e2)
            plt.ylim(1e-5, 2e0)
            plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
            plt.ylabel('intensity')
            plt.yscale('log')
            plt.xscale('log')
            plt.title('intensities(wvl) vs radial distance')
            plt.grid(True)
           
            if True:
                
                i=np.argmin(np.abs(lam_lst-lam_c))
                # AO corrected psf 
                plt.plot(aS, Int_D0_psf_prf[i,1,:],
                          label='no coro', color=colors[0])
                
                # AO corrected coronagraphic psf 
                plt.plot(aS, Int_D_psf_prf[i,1,:],
                          label='with coro', color=colors[-1])
                
            # Focal plane mask boundary
            x = np.arange(0.0, mB/2, 0.01)
            plt.axvline(as_oi, color='k', ls='--')
            plt.legend(fontsize='xx-small', ncols=5)
            # Focal plane mask grey area
            fill_max=mB/2/ 38.54*lam_c/rad2mas
            plt.fill_between(x*lamCD2mas, 1e-5, fill_max, color='gray', alpha=0.3)
            
            fname_contrast_profile = ('intensities_profiles_1600nmOnly_'+base_cro)
            fpath_contrast_profile_svg = (fdir_plt / opd_set /
                                          (fname_contrast_profile +'_'+lmn+'_'+lmx+ '.svg'))
            fpath_contrast_profile_pdf = (fdir_plt / opd_set /
                                          (fname_contrast_profile +'_'+lmn+'_'+lmx+ '.pdf'))
            plt.savefig(fpath_contrast_profile_svg)
            plt.savefig(fpath_contrast_profile_pdf)
            plt.show()
        
