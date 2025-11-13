# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:51:38 2024

@author: asp
"""
import os
from astropy.io import fits

# ['20241007141137', '20241007141603', '20241009084549', '20241009084821']

rdirs = os.listdir("D:/Andes/Data_corono/results")
gdirs = [x for x in rdirs if '20241007141137' in x]

for gdir in gdirs:

    fits_files = []

    for root, dirs, files in os.walk("D:/Andes/Data_corono/results/"+gdir):

        for file in files:
            
            if file.endswith(".fits"):
            
                fits_files.append(os.path.join(root, file))
                fits.setval(os.path.join(root, file),
                            'NCPA',
                            value=0,comment='ncpa rms in meters')
                # fits.setval(os.path.join(root, file),
                #             'LMBD',
                #             value=1600e-9,comment='reference wvl in meters')
                # fits.setval(os.path.join(root, file), 
                #             'DISP',
                #             value=4e7,comment='achr. disp. in mas/m')
