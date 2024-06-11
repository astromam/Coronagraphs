# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:30:48 2024

@author: asp
"""

#%%
"""
### Initialization
"""

import numpy as np
from uniform_disk import uniform_disk
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from pathlib import Path

plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!


#%%
"""
### Parameters
"""
# Pupil size
nPup = 400  #  even/pair!

nOPD = 2000

"""
### Coronagraphic components
"""
diam = 0.9 # diameter of the pupil in fraction of the pupil size
obst = 0.38 # diameter of the central obscuration in fraction of the pupil size


#%%
"""
### Working directories
"""
fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()
fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
fdir_pupil = fdir_dat / 'Pupil'
fname_elt = 'ELT_pupil_400.fits' # New pupil with new spider
fpath_elt = fdir_pupil / fname_elt

"""
### Read file
"""
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)
nzp = np.where(Pupil!=0)

LyotStop2d = Pupil*(
    uniform_disk(nPup, diam*nPup/2)-uniform_disk(nPup, obst*nPup/2))
nzl = np.where(LyotStop2d!=0)

#%%
"""
### working directory of the OPD files
"""
# Directory for the OPDs with the corresponding seed value 

opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',
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

res = np.zeros((4, len(opds_dir)))

for dir_nb in range(len(opds_dir)):

    new_spider_flare = fdir_dat / opds_dir[dir_nb]
    fdir_opd   = new_spider_flare
    opd_set = os.path.basename(fdir_opd).split('.')[0]
    
    # Filename and path for the OPD maps
    flist_opd = os.listdir(fdir_opd) 
    nof = len(flist_opd)
    fpath_opd = [fdir_opd / flist_opd[i] for i in range(nof)]
    fpath_opd = sorted(fpath_opd)
    nW = int(np.floor((nof/nOPD)+1))
    if nW>2:
        fpath_opd = [fpath_opd[i] for i in range(1,nof,nW)]
    nOPD = len(fpath_opd)
    print('sample size of OPD files:', nOPD)
    
    OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])
    temp = np.std((OPD_arr*Pupil)[:,nzp[0],nzp[1]],axis=1)
    
    res[0,dir_nb] = np.mean(temp)
    res[1,dir_nb] = np.std(temp)
    temp = np.std((OPD_arr*LyotStop2d)[:,nzl[0],nzl[1]],axis=1)
    
    res[2,dir_nb] = np.mean(temp)
    res[3,dir_nb] = np.std(temp)
    
jqs = ['JQ1','JQ2','JQM','JQ3','JQ4']

for i in range(5):
    
    idx = [j for j,item in enumerate(np.array(opds_dir)) if jqs[i] in item]
    print("jeu ; ", jqs[i])
    print('pupil:')
    print(np.round(np.transpose([res[0,idx],res[1,idx]])))
    print('lyot:')
    print(np.round(np.transpose([res[2,idx],res[3,idx]])))
    print('pupil:')
    print(np.round(np.transpose([np.median(res[0,idx]),np.median(res[1,idx])])))
    print('lyot:')
    print(np.round(np.transpose([np.median(res[2,idx]),np.median(res[3,idx])])))
 


