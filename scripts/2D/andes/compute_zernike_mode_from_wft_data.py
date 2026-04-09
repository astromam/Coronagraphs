# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:02:59 2026

@author: asp
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pyzelda.utils.zernike as zernike

if __name__ == "__main__":
    
    fnm = r'D:\Andes\Data_corono\data\ncpa_ELT_pupil_400_30nm_4096screens.fits'
    
    ncpa = fits.getdata(fnm)[0:3999,:,:] * 1e9
    mask = ((ncpa[0,:,:] - ncpa[0,0,0]) != 0 ) * 1.

    ncpa_shp = ncpa.shape
    nPup = ncpa_shp[-1]
    
    nZern = 10
    nDim = nPup + 2
    
    # crop the pupil to have array size equal to the pupil diameter
    ini = (nDim-nPup)//2
    end = (nDim+nPup)//2
    
    """
    ### Generation of the Zernike modes
    """
    
    Zern_arr0 = np.zeros((nZern, nDim, nDim))
    for i in range(nZern):
        Zern_arr0[i] = zernike.zernike1(i+2,npix=nDim)
        Zern_arr0[i, np.isnan(Zern_arr0[i])] = 0.
        
    Zern_arr = np.zeros((nZern, nPup, nPup))
    # crop the pupil to have array size equal to the pupil diameter
    Zern_arr = Zern_arr0[:, ini:end, ini:end]
    plt.imshow(Zern_arr[-1])

    res = np.zeros((nZern,ncpa_shp[0]))
    
    for j in range(ncpa_shp[0]):
        
        for i in range(nZern):
        
            res[i,:] = np.sum(ncpa[j,:,:]*mask*Zern_arr[i]) / np.sum(Zern_arr[i]*Zern_arr[i])
        
    print(np.mean(res,axis=1))
    print(np.sqrt(np.sum(np.mean(res,axis=1)**2.)))
    print(np.sqrt(30**2. - np.sum(np.mean(res,axis=1)**2.))) # 30: nm RMS ncpa

