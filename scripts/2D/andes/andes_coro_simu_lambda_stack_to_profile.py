# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 08:58:11 2026

@author: asp
"""

import numpy as np
import psf_profile as pp
from astropy.io import fits
import matplotlib.pyplot as plt

rcn = 'd:/Andes/Data_corono/results/20251001085445/1-10/'

fnm = 'ao_corr_coro_psf_20251001085445.fits'
out = 'ao_corr_coro_psf_profile_20251001085445.fits'

# fnm = 'ao_corr_psf_20251001085445.fits'
# out = 'ao_corr_psf_profile_20251001085445.fits'


if __name__ == '__main__':

    data = fits.getdata(rcn+fnm)
    hdr = fits.getheader(rcn+fnm)
    
    nImg = hdr['NIMG']
    D = hdr['DIAM']
    pscale = hdr['PSCL']    # plate scale in mas per pixel
    
    lam_ref = hdr['LMBD']
    lam_min = hdr['LMIN']
    lam_itv = hdr['LITV']
    lam_stp = hdr['LITV']
    lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)
    nL = len(lam_lst)
    
    # conversion l radian to mas
    rad2mas = np.pi/(180.*3600*1000)
    mas2rad = 1/rad2mas

    lam_ref_2_2_mas = (lam_ref / D) * mas2rad
    # field of view
    # in mas
    fov_mas = nImg * pscale
    # in radians
    fov_rdn = fov_mas * rad2mas
    # in multiple of reference lambda (lamC) over D
    mD_ref = fov_rdn / ( lam_ref / D )
    # mD = fov_rdn * D / lam_lst

    Int_prf_avg = np.zeros([nL, 2, nImg//2])

    for i in range(len(lam_lst)):
        
        mD = mD_ref * lam_ref / lam_lst[i]
        lamD2mas = (lam_lst[i] / D) * mas2rad

        Int_prf_avg[i,1,:], rad_D_prf_avg = pp.radial_profile(data[i,:])
    
        # convert pixel scale into lam/D scale for the x-axis
        rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg
        rad_D_prf_avg_mas = rad_D_prf_avg_lamD * lamD2mas
        Int_prf_avg[i,0,:] = rad_D_prf_avg_lamD * lamD2mas
        
    fits.writeto(rcn+out, Int_prf_avg, hdr, overwrite=True)
    
    
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
        plt.plot(Int_prf_avg[i,0,:], Int_prf_avg[i,1,:],
                label=str(int(lam_lst[i]*1e9+.1))+'nm', color=colors[i])
        
    plt.xlim(-0.05,np.max(rad_D_prf_avg_mas)+0.05)
    plt.ylim(1e-5, 2e0)  #  (2e-5, 2e0)
    plt.show()

