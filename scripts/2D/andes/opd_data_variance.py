#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mes pieds, asp. 07/2025
"""

#%%
"""
### Initialization
"""
import numpy as np
from astropy.io import fits

import os
from pathlib import Path
import matplotlib.pyplot as plt

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas


#%%
"""
### Working directories
"""
# dirs without trailing /
roots = ("OPDs_PASSATA/OPD/WS/JQM_cube",
         "OPDs_PASSATA/OPD/WS/500HzVarWS",
         "OPDs_PASSATA/OPD/WS/1kHzVarWS")

fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()

all_dirs = []
r_dirs = []
for r, root in enumerate(roots):
    r_dirs.append(os.path.basename(root))
    with os.scandir(fdir_dat / root) as entries:
        for entry in entries:
            if entry.is_dir():
               all_dirs.append(os.path.join(root,entry.name))

all_dirs = sorted(all_dirs,key=len)


#%%
fdir_pupil = fdir_dat / 'Pupil'
fpath_elt = fdir_pupil / 'ELT_pupil_400.fits'
Pupil = fits.getdata(fpath_elt,)
width = Pupil.shape[0]

# plt.imshow(Pupil)
# plt.show()

iok = np.nonzero(Pupil)

res = {}
for f, dnm in enumerate(all_dirs):
    
    file_res = []
    for root, dirs, files in os.walk(fdir_dat / dnm):
        for d, fnm in enumerate(files):
            if fnm.endswith(".fits"):
                hdr = fits.getheader(os.path.join(root, fnm))
                if hdr['NAXIS1'] == width or hdr['NAXIS2'] == width:
                    file_res.append(os.path.join(root, fnm))
                
    file_res = sorted(file_res,key=len)
    cube = np.zeros((width,width,len(file_res)))
        
    for g, fnm in enumerate(file_res):
        
        opds = fits.getdata(fdir_dat / dnm / fnm)
        
        if len(opds.shape) == 3:
            
            # print(opds.shape)
            if opds.shape[2] >= 2400:
                # only last 80% of all frames
                opds = opds[:,:,int(opds.shape[2]*0.2):]
            else:
                opds = np.transpose(opds,(1,2,0))
            
            spatialStd = np.zeros((opds.shape[2]))
            
            for i in range(opds.shape[2]):
                
                spatialStd[i] = np.std(opds[:,:,i][iok])
        
            tempStd = np.std(opds, axis=2)
            # plt.imshow(tempStd)
            # plt.show()
            tempStd = tempStd[iok]
            r_d = [x for x in r_dirs if x in dnm ]
            res[dnm] = (r_d, np.rint(np.mean(spatialStd)),
                        np.rint(np.std(spatialStd)))
                                     # np.rint(np.mean(tempStd)))
            
            print(r_d[0],
                  np.rint(np.mean(spatialStd)), np.rint(np.std(spatialStd)))
            #       np.rint(np.mean(tempStd)))      

nb_val = len(res)
d_c = {0:'g',1:'b',2:'r'} #♥ follows roots tuple order
for r, bnm in enumerate(r_dirs):
    
    p = 0
    
    plt.figure(0)
    plt.ylim(0, 25e1)
    plt.xlabel('set #')
    plt.ylabel('nm RMS')
    plt.title('temporal mean of spatial std dev')
    
    plt.figure(1)
    plt.ylim(-5e1, 5e1)
    plt.xlabel('set #')
    plt.ylabel('nm RMS')
    plt.title('temporal std dev of spatial std dev')
    
    for i in range(nb_val):
        
        if bnm in list(res)[i]:
            
            plt.figure(0)
            plt.plot(p, res[list(res)[i]][1], marker='+', color=d_c[r])
            plt.figure(1)
            plt.plot(p, res[list(res)[i]][2], marker='+', color=d_c[r])
            p = p + 1

plt.figure(0)
bbox = dict(boxstyle='square', fc='w', alpha=0.75)
plt.text(0, 50,'green: 500Hz 2024, blue: 500Hz 2025, red: 1kHz 2025', bbox=bbox)
plt.savefig((fdir_dat / "OPDs_PASSATA/OPD/WS/tempMeanOfSpatialStdDev.pdf"))

plt.figure(1)
bbox = dict(boxstyle='square', fc='w', alpha=0.75)
plt.text(0, -25,'green: 500Hz 2024, blue: 500Hz 2025, red: 1kHz 2025', bbox=bbox)
plt.savefig((fdir_dat / "OPDs_PASSATA/OPD/WS/TempStdDevOfSpatialStdDev.pdf"))

