# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:36:25 2024

@author: asp
"""
# plot psf maps from contrast profile
# fits files produced by test_andes_corono_multi_wvl*.py

#%%
"""
### Initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
import os
from pathlib import Path
from mpl_toolkits.axes_grid1 import AxesGrid

# from psf_profile import profile

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!


#%%
"""
### scaling
"""

lam_c = 1.6e-6  #  "central" reference lambda, "of interest", in meters

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# angular separation of interest in mas
as_oi = 20.

# wvl = 'YJH'


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_base  = Path('D:/Andes/Data_corono/').resolve()
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

was_donow = '20251020152206'
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

# 0.89 0.39 3.9 date of now :  20251003104537
# 0.88 0.35 3.9 date of now :  20251003104346
# 0.87 0.36 3.8 date of now :  20251003104236
# 0.88 0.34 3.9 date of now :  20251003104427

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

# 20250711155932
# 20250711155911

# 20250711160041
# 20250711160019

#%%
# 500 Hz    0.9/0.37/4.0    20250708162615 lam_itv = 50
# 1 kH0 Hz  0.9/0.37/4.0    20250708171059 lam_itv = 50

# 500 Hz    0.9/0.37/4.0    20250604120819 lam_itv = 80
# 1 kH0 Hz  0.9/0.37/4.0    20250602181544 lam_itv = 80
# 500 Hz    0.9/0.37/4.0    20250610135805 lam_itv = 50
# 1 kH0 Hz  0.9/0.37/4.0    20250610140016 lam_itv = 50

# 0.804 	 [0.92 0.36 3.9 ]    20250523162033 20250523162130
# 0.811 	 [0.92 0.35 3.8 ]    20250523162217 20250523162247
# 0.824 	 [0.93 0.36 3.6 ]    20250523163019 20250523162955 
# 0.826  	 [0.92 0.33 3.5 ]    20250523163116 20250523163139

# RIZ 0.9 / 0.37 / 4.
# 20250521172353 JQM 05/2025  500Hz
# 20250521172457 JQM 05/2025 1000Hz
# YJH 0.9 / 0.37 / 4.
# 20250522161347 JQM 05/2025  500Hz
# 20250522160547 JQM 05/2025 1000Hz

#%%
# ideal with lyotstop + petaling20250321160937 0,1,2 20250321161111 1,2,3
# 20250321144751 perfect 2nd order with lyotstop + petaling JQM
# 20250321093529 perfect 2nd order with lyotstop + TT + petaling 2..4 vs 5..1
# 20250321093502 perfect 2nd order with lyotstop + TT + petaling 1..3 vs 4..0
# 20250321085350 perfect 2nd order with lyotstop + TT + petaling 0..2 vs 3..5

# 20250320164637 perfect 2nd order with lyotstop + TT + petaling 2..3 vs 4..1
# 20250320164551 perfect 2nd order with lyotstop + TT + petal 1..2 vs 3..0
# 20250320163707 perfect 2nd order with lyotstop + TT + petals 3..5 vs 0..2?
# 20250317172539 perfect 2nd order with lyotstop + TT
# 20250318154752 perfect 2nd order with lyotstop + TT, JQ1 scaled 90%
# 20250318154814 perfect 2nd order with lyotstop + TT, JQ1 scaled 80%
# 20250318154832 perfect 2nd order with lyotstop + TT, JQ1 scaled 70%
# 20250319083152 perfect 2nd order with lyotstop + TT, JQ1 scaled 60%
# 20250319083224 perfect 2nd order with lyotstop + TT, JQ1 scaled 50%
# 20250317172808 perfect 2nd order with lyotstop + petalling corr + TT

# 20250313135902 perfect 2nd order coro + petalling corr
# 20250312094239 psf no TT corr, coro TT corr
# 20250313130757 20250313130437 corr. petal. after TT / first then TT
# 20250312160834 perfect coro no petalling OPDs_PASSATA/OPD/WS/ASI
# 20250310155522 20250312085754 perfect coro 2nd order full elt pupil

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
# opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',
          # 'OPDs_PASSATA/OPD/WS/JQ1/20240517_091216',
          # 'OPDs_PASSATA/OPD/WS/JQ1/20240517_100705',
          # 'OPDs_PASSATA/OPD/WS/JQ1/20240517_103452',
          # 'OPDs_PASSATA/OPD/WS/JQ1/20240517_105822',
          # 'OPDs_PASSATA/OPD/WS/JQ1/20240517_111658',
          # 'OPDs_PASSATA/OPD/WS/JQ1/20240517_113534',
          # 'OPDs_PASSATA/OPD/WS/JQ1/20240517_121247')#,
          # 'OPDs_PASSATA/OPD/WS/JQ2/20240517_181033',
          # 'OPDs_PASSATA/OPD/WS/JQ2/20240517_183817',
          # 'OPDs_PASSATA/OPD/WS/JQ2/20240517_190418',
          # 'OPDs_PASSATA/OPD/WS/JQ2/20240517_192251',
          # 'OPDs_PASSATA/OPD/WS/JQ2/20240517_194126',
          # 'OPDs_PASSATA/OPD/WS/JQ2/20240517_195959',
          # 'OPDs_PASSATA/OPD/WS/JQ2/20240517_201835',
          # 'OPDs_PASSATA/OPD/WS/JQ2/20240517_203708',
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0')#,
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

# opds_dir=('OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',)

# opds_dir=('OPDs_PASSATA/OPD/WS/ASI',)

# # ,
# #           'perfect')

# opds_dir=('OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0')
# ,
#           'perfect')
#%%
# opds_dir=('OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_123456.0',)
# opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',)
# opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0')
# opds_dir=('OPDs_PASSATA/OPD/WS/ASI_staticVib',)
# opds_dir=('OPDs_PASSATA/OPD/WS/ASI_M1M4err',)
# opds_dir=('OPDs_PASSATA/OPD/WS/1kHz/02042025/20250325_145800.0_phase_screens-001/20250325_145800.0',)
# opds_dir=('OPDs_PASSATA/OPD/WS/500Hz/20250508_163051.0_phase_screens/20250508_163051.0_oaCUBEs',)

# root = 'OPDs_PASSATA/OPD/WS/'
# opds_dir=(root+'ocam2k',)
# opds_dir=(root+'alice',)
# opds_dir=(root+'cam_500us',)


#%%
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
#           root+'20250707_114321.0')
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
          root+'10')
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
#           root_1kHz+'20250527_145633.0_phase_screens/20250527_145633.0_phase_screens')

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
#           root_500Hz + '20250527_122032.0_phase_screens/20250527_122032.0_oaCUBEs')

#%%
for dir_nb in range(len(opds_dir)):
            
    opd_set = os.path.basename(fdir_res / opds_dir[dir_nb] ).split('.')[0]
    # opd_set='toto'  #  ne pas oublier de creer toto avant
    print(fdir_res / opd_set)
    
    #%%
    """
    ### Read psf files & create profiles
    """
    
    # file_lst = sorted(
    #     recursive_search(fdir_res / opd_set), key=os.path.getmtime)
    # , reverse=True
    
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
    
    print("psf :", file_psf)
    print("psf coro :", file_cro)

    file_psf = file_psf[np.argmin(
        (lambda x:[len(i) for i in x])(file_psf))]
    file_cro = file_cro[np.argmin(
        (lambda x:[len(i) for i in x])(file_cro))]

    base_psf = os.path.basename(file_psf).split('.')[0]
    base_cro = os.path.basename(file_cro).split('.')[0]
    
    Int_D0 = fits.getdata(fdir_res / opd_set / file_psf)
    # print(np.min(Int_D0), np.max(Int_D0))
    Int_D  = fits.getdata(fdir_res / opd_set / file_cro)
    # print(np.min(Int_D), np.max(Int_D))
    
    head_psf = fits.getheader(fdir_res / opd_set / file_psf)
    
    lam_min = head_psf['LMIN']
    lam_stp = head_psf['LSTP']
    lam_itv = head_psf['LITV']
    # lam_lst = np.arange(lam_min,lam_min+(lam_itv+1)*lam_stp,lam_stp)
    lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)
    lam_lst = lam_lst[np.where(lam_lst < 1860e-9)]
    
    if np.median(lam_lst) > 1e-6:
        wvl = 'YJH'
    else:
        wvl = 'RIZ'
    
    nL = len(lam_lst)
    
    nImg = head_psf['NIMG']
    pscale = head_psf['PSCL']
    D = head_psf['DIAM']
    mB = head_psf['SFPM']
    # lam_c = head_psf['LMBD']
    hlf_fov = nImg * pscale / 2.

    # if lambda of interest / D
    rW_mas = (lam_c / D) * mas2rad

    # stackoveflow...
    a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
    b_ = a_.copy()
    aa, bb = np.meshgrid(a_, b_)
    rad_pix = np.sqrt(aa**2 + bb**2)
    rad_mas = rad_pix * (mas2rad * hlf_fov / (D * 1e9) )
    
    #  angular separation
    aS = np.arange(nImg//2) * (mas2rad * hlf_fov / (D * 1e9) )
    
          
    # filename of the plot
    fnm =  f'ao_corrected_coro_psf_{wvl}band'
    # fname_images_svg = fnm+'.svg'
    fname_images_pdf = fnm+'.pdf'
    fname_images_png = fnm+'.png'

    # filepath for the direct and coronagraphic images
    # fpath_images_svg = fdir_plt / opd_set / fname_images_svg
    fpath_images_pdf = fdir_plt / opd_set / fname_images_pdf
    fpath_images_png = fdir_plt / opd_set / fname_images_png

    # index of wvl to display
    iD = [0,nL//4+1,nL//2-1,nL*3//4-1,nL-1]
    
    # boundaries for the images in log scale
    vmin0 = -5
    vmax0 = 0
    
    fig = plt.figure(4, figsize=(16,6))
    plt.clf()
    plt.tight_layout()
    # plt.suptitle('ao corrected psf (top) vs ao corrected coro. psf (bottom)')
    
    grid = AxesGrid(fig, 111,
            nrows_ncols=(2, 5),
            axes_pad=0.3,
            cbar_mode='single',
            cbar_location='right',
            cbar_pad=0.2,
            label_mode='L'
            )
    
    for i in range(5):
        
        print('position maxi in flattened psf array: ',
              np.argmax(Int_D0[iD[i],:]))
        im = grid[i].imshow(
            np.log10(Int_D0[iD[i],:]),vmin=vmin0,vmax=vmax0,cmap='inferno',
            extent=(-aS[-1],aS[-1],-aS[-1],aS[-1]))
        grid[i].set_ylabel("sep. [mas]")
        grid[i].set_title(str(int(lam_lst[iD[i]]*1e9+.1))+'nm')
         
        im = grid[i+5].imshow(
            np.log10(Int_D[iD[i],:]),vmin=vmin0,vmax=vmax0,cmap='inferno',
            extent=(-aS[-1],aS[-1],-aS[-1],aS[-1]))
        grid[i+5].add_patch(
            Ellipse( (0,0),rW_mas*mB,rW_mas*mB,
                    color='w',ls=':',hatch='xxx',fill=False,alpha=0.7))
        grid[i+5].set_xlabel("sep. [mas]")
        grid[i+5].set_ylabel("sep. [mas]")
        # grid[i+5].add_patch(
        #     Ellipse( (0,0), as_oi*2, as_oi*2,color='w',ls=':',lw=2,fill=False))
        
    # colorbar
    cbar = grid[0].cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Intensity in log scale', rotation=270)
    
    # plt.savefig(fpath_images_svg)
    plt.savefig(fpath_images_pdf, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(fpath_images_png)

    # if i==5:
    plt.show()
    # else : 
    # plt.close()
    # plt.close()

