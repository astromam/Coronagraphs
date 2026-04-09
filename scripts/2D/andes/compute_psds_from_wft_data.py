# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:02:59 2026

@author: asp, mndiaye
"""

import numpy as np
from astropy.io import fits
from pyzelda.ztools import compute_fft_opd, integrate_psd

# from PYZELDA


if __name__ == "__main__":
    
    fnm = r'D:\Andes\Data_corono\data\ncpa_ELT_pupil_400_30nm_4096screens.fits'
    
    ncpa = fits.getdata(fnm)[0:3999,:,:]
    ncpa *= 1e9
    mask = ((ncpa[0,:,:] - ncpa[0,0,0]) != 0 ) * 1.

    ncpa_shp = ncpa.shape
    res = np.zeros((4,ncpa_shp[0]))
                
    for i in range(ncpa.shape[0]):
        
        psd = compute_fft_opd(ncpa[i,:,:], mask)
        psd = np.abs(psd)**2.
        
        res[:,i] = [integrate_psd(psd, 199.5, 0, 2),
                    integrate_psd(psd, 199.5, 2, 7.5),
                    integrate_psd(psd, 199.5, 7.5, 199.5),
                    integrate_psd(psd, 199.5, 0, 199.5)]

    print('\n', np.mean(res[0,:]), '\n', np.mean(res[1,:]),
          '\n', np.mean(res[2,:]), '\n', np.mean(res[3,:]))

