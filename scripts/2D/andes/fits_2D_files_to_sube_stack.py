#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mndiaye, asimmonnin, asp. 07/2025
"""

#%%
"""
### Initialization
"""
import numpy as np
from astropy.io import fits

import os
from pathlib import Path
# import matplotlib.pyplot as plt

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas


#%%
"""
### Working directories
"""
roots = ("OPDs_PASSATA/OPD/WS/JQM",)
fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()

all_dirs = []
for r, root in enumerate(roots):
    with os.scandir(fdir_dat / root) as entries:
        for entry in entries:
            if entry.is_dir():
               all_dirs.append(os.path.join(root,entry.name))

all_dirs = sorted(all_dirs,key=len)


#%%

for f, dnm in enumerate(all_dirs):
    
    file_res = []
    for root, dirs, files in os.walk(fdir_dat / dnm):
        for d, fnm in enumerate(files):
            if fnm.endswith(".fits"):
                    file_res.append(os.path.join(root, fnm))
    
        file_res = sorted(file_res,key=len)
        OPD_arr = np.asarray(
            [fits.getdata(file_res[i]) for i in range(len(file_res))])
    
        fits.writeto((fdir_dat / dnm / ('cube.fits')), OPD_arr,
                     overwrite=True)

        
# mkdir 20240509_182041 20240509_183915 20240509_191620 20240509_193453 20240509_195327 20240509_201200 20240509_203033 20240509_204907 20240509_210742