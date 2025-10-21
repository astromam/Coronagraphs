#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@authors: mndiaye, asimonnin, asp
"""
# for multiple wavelengths compute normalized psf profiles intensity for coro 
# and no coro, with or without residual opds/windshake including 
# tilt/atmospheric dispersion
# normalized to the peak intensity of the no coro, no opds pupil with lyot stop

#%%
"""
### Initialization
"""

import numpy as np
from scipy import ndimage

# pythonpath to update possibly...
import slow_fourier_transform as sft
from uniform_disk import uniform_disk
from psf_profile import profile

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid

from astropy.io import fits

import os
import time
from pathlib import Path
from datetime import datetime  #  asp for datetime of now
#  import pdb

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!


#%%
"""
### Parameters
"""
# Pupil size
# nPup = 400  #  even/pair!

# Sampling of the coronagraph focal plane mask
nFPM = 100

# "field of view" in mas, not the real fov but some legacy nIMG*pscale/2...
# f_o_v = 58.393

# Image size
nImg = 400  #*1.6 #  #  even/pair!

# angular separation of interest in mas
as_oi = 20.

# # of OPD phase screens
nOPD = 4000

# wavelengths in m
lamC = 1600e-9  #  some reference wvl unique value

# yjhk --> 2025-03
# lam_min = 960e-9  #  min value in range
# lam_max = 2450e-9  #  max value in range  #  2450e-9 // 1800e-9
# lam_itv = 18  #  nb of intervals in range --> nb+1 wvl's !  #  18 // 10
# lam_stp = np.floor(np.ceil((lam_max-lam_min)*1e9/lam_itv)/10)*1e-8  # wvl step

# riz
# lam_min = 630e-9  #  min value in range
# lam_max = 960e-9  #  max value in range  #  2450e-9 // 1800e-9
# lam_itv = 8  #  nb of intervals in range --> nb+1 wvl's !  #  18 // 10
# lam_stp = 40e-9 # np.floor(np.ceil((lam_max-lam_min)*1e9/lam_itv)/10)*1e-8  # wvl step

# yjhk  2025-05 -->
lam_min = 950e-9  #  min value in range
lam_itv = 30
lam_stp = 50e-9

lam_lst = np.arange(lam_min,lam_min+(lam_itv+0.5)*lam_stp,lam_stp)
nL = len(lam_lst)
print(nL, lam_stp, lam_lst)

# Pupil diameter in m 
D = 38.54

# conversion l radian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# plate scale in mas per pixel
pscale = 0.3

lamCD2mas = (lamC/D)*mas2rad
# field of view
# in mas
fov_mas = nImg * pscale
# in radians
fov_rdn = fov_mas * rad2mas
# in multiple of reference lambda (lamC) over D
mD_ref = fov_rdn / ( lamC / D )

"""
### Coronagraphic components
"""
# Focal plane mask
mask2d = uniform_disk(nFPM, nFPM/2.)
# diam = 0.90 obs = 0.37, mB = 4. with 'ELT_pupil_400.fits' YJH
# diam = 0.96 obs = 0.30, mB = 4.5 with 'ELT_pupil_400.fits' K
# diam = 0.90 obs = 0.38 with 'Tel-Pupil.fits'

# 'YJH2024' 0.9 / 0.37 / 0.4 @ 25 mas / 75% thr

# from JH optimum research @ 25 mas: throughput, <contrast>, config, comments
# 0.754 	 0.000253336 	 [0.9  0.37 4.  ]    'YJH2024'
# ...
# 0.804 	 0.000341188 	 [0.92 0.36 3.9 ]    20250523162033 20250523162130
# 0.805 	 0.000349114 	 [0.91 0.33 3.9 ]
# 0.811 	 0.000354267 	 [0.92 0.35 3.8 ]    20250523162217 20250523162247
# 0.819 	 0.000370672 	 [0.92 0.34 3.8 ]
# 0.824 	 0.000380266 	 [0.93 0.36 3.6 ]    20250523162955 20250523163019
# 0.826 	 0.000388391 	 [0.92 0.33 3.5 ]    20250523163116 20250523163139

# new params exploration follow. ruane2018 06/2025
# photometric aperture 7µ
# J     4.6e-05     0.85    0.36   3.3 0.66501  20250618164513
# H     0.000392    0.86    0.3    4.1 0.72769  20250618164620
# JH    0.000428    0.86    0.3    4.0 0.72769  20250618164704
# YJH   0.000628    0.86    0.37   4.0 0.67608  20250618164749
# 0.85 0.32 3.9 

diam = 0.89 # diameter of the pupil in fraction of the pupil size
obst = 0.35 # diameter of the central obscuration in fraction of the pupil size
# FPM size in lam/D in the focal plane B
# 4. with 'ELT_pupil_400.fits'
# 3.8 with 'Tel-Pupil.fits', old!
mB = 3.9

# dispersion mas/m
disp = 2e7  #  e.g 8e7 ou 80e6 = 80 mas / 1e-6 m
# psf to fpm decentering in mas then for legacy in radians
fpm_dec_mas = 0.
fpm_dec = fpm_dec_mas * rad2mas
# lyot stop angular position error in degres
ls_ape = 0.
# lyot stop vertical - elevation - offset error in pixels
ls_voe = 0
# lyot stop horizontal - azimut - offset error in pixels
ls_hoe = 0
# ncpa phase screens
ncpa_rms = 0  #  nm
# defocus at fpm in nm RMS for reference wvl
fpm_dfe_elt = 0
# fpm_dfe = -1 if fpm_dfe_elt = 0., computed dynamicaly otherwise
fpm_dfe = fpm_dfe_elt - 1.

# compute & keep elt psf & profiles
simu_elt = False

# datetime of script execution
donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now

print('\n', diam, obst, mB)
print('date of now : ', donow)


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()  # opd's seed value
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots

elif user == 'Adrien':
    # File directory
    fdir_dat = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/plots/').resolve()

elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    # File directory
    fdir_dat = Path( fdir_base / 'data' ).resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path( fdir_base / 'results' ).resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path( fdir_base / 'plots' ).resolve()

# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'

fdir_res = fdir_res / donow
fdir_prfct = fdir_res / 'perfect'
os.makedirs(fdir_prfct, exist_ok=True)

fdir_plt = fdir_plt / donow


#%%
"""
data cubes for all psfs and profiles
"""

Int_DD0 = np.zeros([nL, nImg, nImg])
Int_DD = np.zeros([nL, nImg, nImg])
Int_D0 = np.zeros([nL, nImg, nImg])
Int_D = np.zeros([nL, nImg, nImg])
Int_D0_prf_avg = np.zeros([nL, 2, nImg//2])
Int_D_prf_avg = np.zeros([nL, 2, nImg//2])
Int_elt = np.zeros([nL, nImg, nImg])


#%%
"""
ncpa's, pupils, ncpa, defocus
"""
# Filename and path for the ELT pupil  //  'Tel-Pupil.fits' <-- OLD
fname_elt = 'ELT_pupil_400.fits' # New pupil with new spider
fpath_elt = fdir_pupil / fname_elt
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)
nPup = Pupil.shape[0]
ipup = np.nonzero(Pupil)

# 2D array pupil slope for tilt
sf_x = np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5
sf_y = np.transpose(sf_x.copy())

#2D array for distance to center pixel in pupil, in [0,1]
kx = (np.arange(nPup)-nPup//2)/(nPup/2)
ky = (np.arange(nPup)-nPup//2)/(nPup/2)
kx2, ky2 = np.meshgrid(kx, ky)
rho = np.sqrt(kx2**2 + ky2**2)

ncpa_d = np.zeros((nOPD,nPup,nPup))
if ncpa_rms != 0:
    print("NCPA [nm RMS]:",ncpa_rms)
    # ncpa_d = np.zeros((nOPD,nImg,nImg))
    # fnm = ('ncpa_pupil_new_'+str(ncpa_rms)+'nm.fits')
    fnm = ('ncpa_ELT_pupil_400_'+str(ncpa_rms)+'nm_4096screens.fits')
    ncpa_d = fits.getdata(fdir_dat/fnm)


dfc = np.zeros((nPup,nPup))
if fpm_dfe != 0:
    
    dfc = uniform_disk(nPup,nPup//2)
    iok = np.nonzero(dfc)
    dfc[iok] = np.sqrt(3.)*(rho[iok]*rho[iok]-1.)

    dfc_temp = dfc.copy()
    dfc_temp *= Pupil.copy()
    mp = np.mean(dfc_temp[ipup])
    dfc_temp -= mp
    sp = np.std(dfc_temp[ipup])
    dfc_temp /= sp
    dfc_temp *= fpm_dfe_elt * 1e-9
    dfc -= mp
    dfc /= sp
    dfc *= fpm_dfe_elt * 1e-9
    dfc -= np.mean(dfc[iok])
    fpm_dfe = np.std(dfc[iok]) * 1e9
    dfc = dfc_temp.copy()    
    
    print('elt pupil circumcircle input defocus:', np.round(fpm_dfe,3),\
          ',\nelt pupil input defocus:', np.round(fpm_dfe_elt,3))


#%%
'''
rotate then shift Lyot Stop and defocus
'''
# Lyot stop
LyotStop2d = Pupil*(uniform_disk(nPup, diam*nPup/2) -
                    uniform_disk(nPup, obst*nPup/2))

if ls_ape != 0:
    
    pup_rot = ndimage.rotate(Pupil,ls_ape, reshape=False)
    pup_rot = pup_rot > 0.5
    LyotStop2d = pup_rot*(uniform_disk(nPup, diam*nPup/2) -
                          uniform_disk(nPup, obst*nPup/2))

if ls_voe != 0 or ls_hoe != 0:
    
    LyotStop2d = np.roll(LyotStop2d,(ls_hoe,ls_voe),(1,0))

    
#%%
"""
### working directory of the OPD files
"""

#%%
#tests cameras
# root = 'OPDs_PASSATA/OPD/WS/'
# # opds_dir=(root+'ocam2k',)
# # opds_dir=(root+'alice',)
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

#%%
# Directory for the OPDs with the corresponding seed value 
# (from HARMONI simulation)
# fdir_opd   = fdir_dat / 'OPD_Harmoni' / str(seed)

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
          # 'OPDs_PASSATA/OPD/WS/JQ1/20240517_121247',
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
          # 'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0',
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
# opds_dir=('OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0')
# opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',
#           'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0')
# opds_dir=('OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',)
# opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',)
# opds_dir=('OPDs_PASSATA/OPD/WS/ASI_staticVib',)

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

# opds_dir=('OPDs_PASSATA/OPD/WS/500Hz/20250508_163051.0_phase_screens/20250508_163051.0_oaCUBEs',)

#%%
for dir_nb in range(len(opds_dir)):

    new_spider_flare = fdir_dat / opds_dir[dir_nb]
    fdir_opd   = new_spider_flare
    
    opd_set = os.path.basename(fdir_opd).split('.')[0]
    print(opd_set)
    
    # opd_set='toto'
    os.makedirs(fdir_res / opd_set, exist_ok=True)
    os.makedirs(fdir_plt / opd_set, exist_ok=True)
    
    
    #%%
    
    # Filename and path for the OPD maps
    flist_opd = []
    for fnm in os.listdir(fdir_opd):
        if fnm.endswith(".fits"):
            flist_opd.append(fnm)
    
    flist_opd = sorted(flist_opd,key=len)
    # flist_opd = sorted(os.listdir(fdir_opd),key=len) 
    nof = len(flist_opd)
    print("# of files in phase screens dir:", nof)
    fpath_opd = [fdir_opd / flist_opd[i] for i in range(nof)]
    # fpath_opd = sorted(fpath_opd)
    nW = int(np.floor((nof/nOPD)+1))
    if nW>2:
        fpath_opd = [fpath_opd[i] for i in range(1,nof,nW)]
    nOPD = len(fpath_opd)
    print('sample size of OPD files:', nOPD)
    
    
    #%%
    """
    ### Read OPD files
    """
    
    t0 = time.time()
    # Read OPD maps for the nOPD files
    OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])

    # pour jeu fichier fits unique, e.g: OPDs_PASSATA/OPD/WS/ASI_*
    if nof < 2:
        start = int((OPD_arr.shape)[3]/5)
        print("skip:", start)
        OPD_arr = OPD_arr[0,:,:,start:]
        OPD_arr = np.transpose(OPD_arr,(2,0,1))
        nOPD = OPD_arr.shape[0]
        print("phase screen array shape from file: ", OPD_arr.shape)
    else:
        print("phase screen array shape from files: ", OPD_arr.shape)

    t1 = time.time()
    print(f'OPD reading file time: {t1-t0:.3f}s')
    
    OPD_arr *= 1e-9 # convert OPD from nm `to m if new OPD with new pupil

    
        #%%
    
    for i in np.arange(nL):
        
        lam = lam_lst[i]
        
        dLam = lam - lamC
        tilt = dLam * disp * rad2mas
        
        # conversion lam/D to mas
        lamD2mas = (lam/D)*mas2rad
        
        # FoV in lam/D in the final image plane D
        # mD = f_o_v*(nImg/(lam*1e9))
        mD = mD_ref * lamC / lam
        
        print('lambda (nm):' ,np.round(lam*1e9,0), '. Field of view (lam/D):',
              np.round(mD,3))
        
            
        #%%
        """
        ### Compute perfect PSF
        """
        
        # Field in the entrance pupil plane A
        Fld_AA0 = Pupil * 1.* LyotStop2d
        
        # Field in the image plane D (no coronagraph)
        Fld_DD0 = sft.sft(Fld_AA0, nImg, mD*diam)


        # Intensity 
        Int_DD0[i,:] = np.abs(Fld_DD0)**2
        
        # Normalized intensity
        norm_peakDD0 = 1/np.max(Int_DD0[i,:])
        Int_DD0[i,:] *= norm_peakDD0

        
        #%%
        """
        ### Compute perfect coronographic image
        """
        
        # pupil plane A
        Fld_AA = Pupil * 1.
        # focal plane B 
        Fld_BB = mask2d*sft.sft(Fld_AA, nFPM, mB*lamC/lam)
        
        # pupil plane C before Lyot stop
        Fld_CC = Fld_AA - sft.isft(Fld_BB, nPup, mB*lamC/lam)
        
        # # pupil plane C after Lyot stop
        Fld_LL = Fld_CC * LyotStop2d
        
        # image plane D 
        Fld_DD = sft.sft(Fld_LL, nImg, mD*diam)
        
        # Intensity
        Int_DD[i,:] = np.abs(Fld_DD)**2
        
        # Normalized intensity
        Int_DD[i,:] *= norm_peakDD0
        
        
    #%%    
        # computation of the averaged intensity profiles of the images  
        Int_DD0_prf_avg, rad_DD0_prf_avg = profile(Int_DD0[i,:], ptype='mean')
        Int_DD_prf_avg, rad_DD_prf_avg = profile(Int_DD[i,:], ptype='mean')
        
        if simu_elt:
            Fld_elt = sft.sft(Pupil*1., nImg, mD*diam)
            Int_elt[i,:] = np.abs(Fld_elt)**2
            Int_elt[i,:] *= norm_peakDD0

        # computation of the standard deviation intensity profiles of the images
        # Int_DD0_prf_std, rad_DD0_prf_std = profile(Int_DD0[i,:], ptype='std')
        # Int_DD_prf_std, rad_DD_prf_std = profile(Int_DD[i,:], ptype='std')
        
        
        #%%
        """
        ### Compute PSF (with errors)
        """
        
        t0 = time.time()
        
        for iOPD in range(nOPD):
            
            # Field in the entrance pupil plane A
            Fld_A0 = (Pupil 
                      * np.exp(1j*2*np.pi *
                               (OPD_arr[iOPD] +
                                ncpa_d[iOPD] +
                                tilt * D * diam * sf_y / nPup +
                                fpm_dec * D * diam * sf_y / nPup +
                                dfc)/lam)
                      * LyotStop2d)
            
            # Field in the image plane D (no coronagraph)
            Fld_D0 = sft.sft(Fld_A0, nImg, mD*diam) ## Avec Lyot stop
            
            # Intensity 
            Int_D0[i,:] += np.abs(Fld_D0)**2 ## Avec Lyot stop
        
        t1 = time.time()       
        print(f'PSF computation time: {t1-t0:.3f}s')  
        
        # Normalized intensity
        Int_D0[i,:] /= nOPD ## Avec Lyot stop
        # Int_D0_2 /= flare[i][1]-flare[i][0]# Sans Lyot stop
        
        # Normalized intensity
        norm_peakD0 = 1/np.max(Int_D0[i,:])
        # norm_peakD0_2 = 1/np.max(Int_D0_2)
        
        Int_D0[i,:] *= norm_peakD0
        
        """
        ### Compute coronographic image (with errors)
        """
        t0 = time.time()
        # Int_D = np.zeros((nImg, nImg))
        
        for iOPD in range(nOPD):
            # pupil plane A
            Fld_A0 = (Pupil 
                      * np.exp(1j*2*np.pi * 
                               (OPD_arr[iOPD] +
                                ncpa_d[iOPD] +
                                tilt * D * diam * sf_y / nPup +
                                fpm_dec * D * diam * sf_y / nPup +
                                dfc)/lam))
            
            # focal plane B 
            Fld_B = mask2d*sft.sft(Fld_A0, nFPM, mB*lamC/lam)
            
            # pupil plane C before Lyot stop
            Fld_C = Fld_A0 - sft.isft(Fld_B, nPup, mB*lamC/lam)
            
            # pupil plane C after Lyot stop
            Fld_L = Fld_C*LyotStop2d
            
            # image plane D 
            Fld_D = sft.sft(Fld_L, nImg, mD*diam)
            
            # Intensity
            Int_D[i,:] += np.abs(Fld_D)**2    
    
        t1 = time.time()       
        
        print(f'Coro image computation time: {t1-t0:.3f}s')  
        
        # Normalized intensity
        Int_D[i,:] /= nOPD
        
        # Normalized intensity
        # Int_D *= norm_peakDD0 #norm_peakD0 DD0 to have common scale
        Int_D[i,:] *= norm_peakD0
        
        """
        ### Compute the radial intensity profiles of the images
        """
        # computation of the averaged intensity profiles of the images   
        # Int_D0_prf_avg2, rad_D0_prf_avg = profile(Int_D0_2, ptype='mean')
        Int_D0_prf_avg[i,1,:], rad_D0_prf_avg = profile(Int_D0[i,:], ptype='mean')
        Int_D_prf_avg[i,1,:], rad_D_prf_avg = profile(Int_D[i,:], ptype='mean')

        # computation of the standard deviation intensity profiles of the images
        # Int_D0_prf_std2, rad_D0_prf_std = profile(Int_D0_2, ptype='std')
        # Int_D0_prf_std, rad_D0_prf_std = profile(Int_D0, ptype='std')
        # Int_D_prf_std, rad_D_prf_std = profile(Int_D, ptype='std')
        
        # convert pixel scale into lam/D scale for the x-axis
        rad_D0_prf_avg_lamD = rad_D0_prf_avg * mD/nImg
        rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg
        
        # rad_D0_prf_avg_mas = rad_D0_prf_avg_lamD * lamD2mas
        Int_D0_prf_avg[i,0,:] = rad_D0_prf_avg_lamD * lamD2mas
        rad_D_prf_avg_mas = rad_D_prf_avg_lamD * lamD2mas
        Int_D_prf_avg[i,0,:] = rad_D_prf_avg_lamD * lamD2mas
        # rad_DD0_prf_avg_mas = rad_DD0_prf_avg_lamD * lamD2mas 
    
    
    #%%
    """
    save data cubes in fits files with keywords
    """
    
    # filename for the direct and coronagraphic images and profiles
    fname_Int_D0 = 'ao_corr_psf_'+donow+'.fits'
    fname_Int_D = 'ao_corr_coro_psf_'+donow+'.fits'
    fname_Prf_D0 = 'ao_corr_psf_profile_'+donow+'.fits'
    fname_Prf_D = 'ao_corr_coro_psf_profile_'+donow+'.fits'
    
    # filepath for the direct and coronagraphic images
    fpath_Int_D0 = fdir_res / opd_set / fname_Int_D0
    fpath_Int_D  = fdir_res / opd_set / fname_Int_D
    fpath_Prf_D0 = fdir_res / opd_set / fname_Prf_D0
    fpath_Prf_D  = fdir_res / opd_set / fname_Prf_D
    
    # save the direct and coronagraphic images
    fits.writeto(fpath_Int_D0, Int_D0, overwrite=True)
    fits.writeto(fpath_Int_D, Int_D, overwrite=True)
    fits.writeto(fpath_Prf_D0, Int_D0_prf_avg, overwrite=True)
    fits.writeto(fpath_Prf_D, Int_D_prf_avg, overwrite=True)

    # fpath_psf_lst=(fpath_Int_D0, fpath_Int_D,fpath_Prf_D0, fpath_Prf_D)
    fpath_psf_lst=(fpath_Int_D0, fpath_Int_D, fpath_Prf_D0, fpath_Prf_D)

    if len(os.listdir(fdir_prfct))==0:
        fname_Int_DD0 = 'wonoise_psf_'+donow+'.fits'
        fname_Int_DD = 'wonoise_coro_psf_'+donow+'.fits'
        fpath_Int_DD0 = fdir_prfct/ fname_Int_DD0
        fpath_Int_DD  = fdir_prfct / fname_Int_DD

        if not os.path.isfile(fpath_Int_DD0):
            fits.writeto(fpath_Int_DD0, Int_DD0, overwrite=True)
            fpath_psf_lst += (fpath_Int_DD0,)
            
        if not os.path.isfile(fpath_Int_DD):
            fits.writeto(fpath_Int_DD, Int_DD, overwrite=True)
            fpath_psf_lst += (fpath_Int_DD,)
    
        if simu_elt:
            fname_elt = 'wo_noise_psf_elt_pupil.fits'
            fpath_elt = fdir_prfct / fname_elt
            if not os.path.isfile(fpath_elt):
                fits.writeto(fpath_elt, Int_elt, overwrite=True)
                fpath_psf_lst += (fpath_elt,)
    
    for fpath in fpath_psf_lst:
        fits.setval(fpath,'NPUP',value=nPup,comment='pupil size')
        fits.setval(fpath,'NFPM',value=nFPM,comment='FP coro. sampling')
        fits.setval(fpath,'NIMG',value=nImg,comment='image size')
        fits.setval(fpath,'NOPD',value=nOPD,comment='number of OPD files')
        fits.setval(fpath,'FOVS',value=fov_mas,comment='field of view in mas')
        fits.setval(fpath,'LMIN',value=lam_min,comment='wavelength in meters')
        fits.setval(fpath,'LITV',value=lam_itv,comment='# of wvl intervals')
        fits.setval(fpath,'LSTP',value=lam_stp,comment='wvl step in meters')
        fits.setval(fpath,'LMBD',value=lamC,comment='reference wvl in meters')
        fits.setval(fpath,'DIAM',value=D,comment='pupil dimater in meters')
        fits.setval(fpath,'PSCL',value=pscale,comment='plate scale in mas')
        fits.setval(fpath,'SFPM',value=mB,
                    comment='FPM (LMBD/D), first focal plane')
        fits.setval(fpath,'FDIA',value=diam,comment='fractional pup. diameter')
        fits.setval(fpath,'OBST',value=obst,comment='fractional obscuration')
        fits.setval(fpath,'OPDS',value=opd_set,comment='opd set creation date')
        fits.setval(fpath,'DISP',value=disp,comment='achr. disp. in mas/m bw')
        fits.setval(fpath,'NCPA',value=ncpa_rms,comment='ncpa rms in meters')
        fits.setval(fpath,'FDEC',value=fpm_dec,
                    comment='psf to fpm offset in radians')
        fits.setval(fpath,'FTLT',value=fpm_dec_mas,
                    comment='psf to fpm offset in mas')
        fits.setval(fpath,'LSAE',value=ls_ape,
                    comment='lyot stop angular position error in degrees')
        fits.setval(fpath,'LSVE',value=ls_voe,
                    comment='lyot stop vertical offset error in pixels')
        fits.setval(fpath,'LSHE',value=ls_hoe,
                    comment='lyot stop horizontal offset error in pixels')
        fits.setval(fpath,'ELT_DFOC',value=fpm_dfe_elt,
                    comment='defocus at elt pupil in nm RMS at reference wvl')
        fits.setval(fpath,'DIAM_DFC',value=np.round(fpm_dfe,3),
                    comment='pup. circumcirc. input defocus nm RMS ref. wvl')
        fits.setval(fpath,'EPUP_FNM',value=fname_elt,
                comment='ELT pupil filename')
        fits.setval(fpath,'DATE_NOW',value=donow,
            comment='daye of now, i.e. script execution')


    #%%
    """
    display images
    # """
        
    # # filename of the plot
    # fname_images_svg = 'ao_corrected_coro_psf_'+donow+'.svg'
    # fname_images_pdf = 'ao_corrected_coro_psf_'+donow+'.pdf'
    
    # # filepath for the direct and coronagraphic images
    # fpath_images_svg = fdir_plt / opd_set / fname_images_svg
    # fpath_images_pdf = fdir_plt / opd_set / fname_images_pdf
    
    # index of wvl to display
    # iD = [0,nL//4,nL//2,nL*3//4,nL-1]
    
    # boundaries for the images in log scale
    # vmin0 = -7
    # vmax0 = 0
    
    # fig = plt.figure(4, figsize=(16,6))
    # plt.clf()
    # plt.tight_layout()
    # plt.suptitle('ao corrected psf (top) vs ao corrected coro. psf (bottom)')
    
    # grid = AxesGrid(fig, 111,
    #         nrows_ncols=(2, 5),
    #         axes_pad=0.3,
    #         cbar_mode='single',
    #         cbar_location='right',
    #         cbar_pad=0.2
    #         )
    
    # for i in range(5):
        
    #     im = grid[i].imshow(
    #         np.log10(Int_D0[iD[i],:]), vmin=vmin0, vmax=vmax0, cmap='inferno')
    #     grid[i].set_title(str(int(lam_lst[iD[i]]*1e9+.1))+'nm')
        
    #     im = grid[i+5].imshow(
    #         np.log10(Int_D[iD[i],:]), vmin=vmin0, vmax=vmax0, cmap='inferno')
    #     # grid[i+1+5].set_title('coro. psf')
        
    # # colorbar
    # cbar = grid[0].cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)
    # cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('Intensity in log scale', rotation=270)
    
    # plt.savefig(fpath_images_svg)
    # plt.savefig(fpath_images_pdf)
    
    # if i==5:
    # plt.show()
    # else : 
    # plt.close()
    # plt.close()
    
    
    #%%
    """
    plot profiles
    """
        
    # filepath for the direct and coronagraphic images
    fname_prf_svg = 'intensities_profiles_'+donow+'.svg'
    fname_prf_pdf = 'intensities_profiles_'+donow+'.pdf'
    fpath_prf_svg = fdir_plt / opd_set / fname_prf_svg
    fpath_prf_pdf = fdir_plt / opd_set / fname_prf_pdf
    
    # plot of the radial profiles
    colors = plt.cm.rainbow(np.linspace(0,1,nL))
    plt.figure(5, (8, 4.5))
    plt.clf()
    plt.tight_layout()
    plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
    plt.ylabel('intensity (log)')
    plt.yscale('log')
    plt.grid(True)
    
    for i in range(0,nL,2):
        
        # AO corrected coronagraphic image
        plt.plot(Int_D_prf_avg[i,0,:], Int_D_prf_avg[i,1,:],
                label=str(int(lam_lst[i]*1e9+.1))+'nm', color=colors[i])
        
    # Focal plane mask boundary
    x = np.arange(0.0, mB/2, 0.01)
    plt.axvline(as_oi, color='k', ls='--')
    plt.legend(fontsize='small', ncols=4)
    # Focal plane mask grey area
    plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lamC/rad2mas, color='gray',
                     alpha=0.3)
    plt.xlim(-0.05,np.max(rad_D_prf_avg_mas)+0.05)
    plt.ylim(1e-5, 2e0)  #  (2e-5, 2e0)

    plt.savefig(fpath_prf_svg)
    plt.savefig(fpath_prf_pdf)
        
    plt.show()

print(diam,obst,mB)
print('date of now : ', donow)

