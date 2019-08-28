#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:48:19 2019

@author: mndiaye
"""

import numpy as np
import pylab as pl
from pathlib import Path
from pyzelda.utils import imutils

import os
from astropy.io import fits
#import corono as coro

#from scipy.misc import imresize
import vip_hci as vip
from hciplot import plot_frames, plot_cubes

#%% parameters
"""
### Parameters
"""
pl.close('all')
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'vlt' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
    solver       = 'stdgrb' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    FirstDerGlobalLim = 1.
    
    #nPup = corono0.params['nPup']
    nPup = 384
    nFPM = 50
    Fmax2d = 50
    nImg2d = 500
    
    # central wavelength (H2 filter)
    wv = 1.593e-6
    
    # mask radius in lam0/D unit
    dAper     = 8
    mas2rad   = np.pi/(180.*3600)
    rMask_m   = 287e-6/2.#372e-6/2.#
    Fratio    = 40
    rMask  = rMask_m/(wv*Fratio)
    print('Mask radius: {0:.3f} lambda_0/D at {1:.3f}um'.format(rMask, wv*1e6))
    rMask_mas = 1000.*rMask * (wv/dAper)/mas2rad
    print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  2.0
    rho1 = 20.0
    
    # contrast in the dark region
    cDarkHole = 6.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.756
    
    # CtrBtwnPix2
    CtrBtwnPix  = True
    CtrBtwnPix2 = True
    Pupil2dSym  = False # set it True only for optimization
    
    #nlam
    bw   = 0.2
    nlam = 5

    
    do_fits = True

nlambis = 11  
nImg2dbis = 200  
Fmax2dbis = (nImg2dbis/2)*(950e-9/wv)
   

# total number of existing maps
qmap = 1000
# total number of used maps
nmap = 1000

# plot parameters
vmin0 = -7.5
vmax0 = -3.5

aplc_arr = ['aplc1', 'aplc2']
naplc = len(aplc_arr)

temp_freq = 1000

# PSD number
iPSD = 0
nPSD = 120

#%%
"""
### File directories to be read
"""
for i, iaplc in enumerate(aplc_arr): 
    fdir_data = Path('../../../results/2D/data/AO_tests/{0}/'.format(
            iaplc)).resolve()
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)

#%%
"""
### File reading path for the generated data
"""
fdir_pdf = Path('../../../results/2D/plots/AO_tests/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

direct_poly_img = np.empty((naplc, nImg2dbis, nImg2dbis))
corono_poly_img = np.empty((naplc, nImg2dbis, nImg2dbis))

direct_poly_img_AO = np.empty((naplc*nPSD, nImg2dbis, nImg2dbis))
corono_poly_img_AO = np.empty((naplc*nPSD, nImg2dbis, nImg2dbis))

direct_poly_avg = np.empty((naplc, nImg2dbis//2))
corono_poly_avg = np.empty((naplc, nImg2dbis//2))
direct_poly_std = np.empty((naplc, nImg2dbis//2))
corono_poly_std = np.empty((naplc, nImg2dbis//2))

direct_poly_avg_AO = np.empty((naplc*nPSD, nImg2dbis//2))
corono_poly_avg_AO = np.empty((naplc*nPSD, nImg2dbis//2))
direct_poly_std_AO = np.empty((naplc*nPSD, nImg2dbis//2))
corono_poly_std_AO = np.empty((naplc*nPSD, nImg2dbis//2))


#%%
"""
### Read data for all the images with no aberrations
"""
for i, iaplc in enumerate(aplc_arr): 
    fdir_data = Path('../../../results/2D/data/AO_tests/{0}/'.format(
            iaplc)).resolve()
    """
    ### Filepaths
    """
    fname_direct = ('{2}_direct_nPup={0}' + \
                    '_nImg={1}_img.fits').format(nPup, nImg2dbis, iaplc)
    fname_corono = ('{2}_corono_nPup={0}' + \
                    '_nImg={1}_img.fits').format(nPup, nImg2dbis, iaplc)
    fpath_direct = fdir_data / fname_direct
    fpath_corono = fdir_data / fname_corono

    # with no aberrations
    fname_direct_avg = ('{2}_direct_nPup={0}' + \
                        '_nImg={1}_avg.fits').format(nPup, nImg2dbis, iaplc)
    fname_corono_avg = ('{2}_corono_nPup={0}' + \
                        '_nImg={1}_avg.fits').format(nPup, nImg2dbis, iaplc)
    fname_direct_std = ('{2}_direct_nPup={0}' + \
                        '_nImg={1}_std.fits').format(nPup, nImg2dbis, iaplc)
    fname_corono_std = ('{2}_corono_nPup={0}' + \
                        '_nImg={1}_std.fits').format(nPup, nImg2dbis, iaplc)

    fpath_direct_avg = fdir_data / fname_direct_avg
    fpath_corono_avg = fdir_data / fname_corono_avg
    fpath_direct_std = fdir_data / fname_direct_std
    fpath_corono_std = fdir_data / fname_corono_std

    """
    ### Read image (no aberration)
    """
    direct_poly_img[i] = fits.getdata(fpath_direct)
    corono_poly_img[i] = fits.getdata(fpath_corono)
    """
    ### Read profiles (with AO residuals)
    """
    direct_poly_avg[i] = fits.getdata(fpath_direct_avg)
    corono_poly_avg[i] = fits.getdata(fpath_corono_avg)
    direct_poly_std[i] = fits.getdata(fpath_direct_std)
    corono_poly_std[i] = fits.getdata(fpath_corono_std)

#%% 
"""
### Read data for all the images with AO residuals
"""
for i, iaplc in enumerate(aplc_arr): 
    fdir_data = Path('../../../results/2D/data/AO_tests/{0}/'.format(
            iaplc)).resolve()
    for iPSD in range(nPSD):
        if (iPSD+1) % 10 == 0:
            print('\niPSD: {0:03d}/{1:03d}'.format(iPSD+1, nPSD))

        """
        ### Filepaths
        """    
        fname_direct_AO = ('{4}_direct_nPup={0}_nImg={1}_iPSD={2:04d}' + \
                           '_nmap={3:04d}_img.fits').format(nPup, nImg2dbis, 
                                  iPSD, nmap, iaplc)
        fname_corono_AO = ('{4}_corono_nPup={0}_nImg={1}_iPSD={2:04d}' + \
                           '_nmap={3:04d}_img.fits').format(nPup, nImg2dbis, 
                                  iPSD, nmap, iaplc)
        fpath_direct_AO = fdir_data / fname_direct_AO
        fpath_corono_AO = fdir_data / fname_corono_AO
        
        # with AO residuals
        fname_direct_avg_AO = ('{4}_direct_nPup={0}_nImg={1}_iPSD={2:04d}' + \
                               '_nmap={3:04d}_avg.fits').format(nPup, 
                                      nImg2dbis, iPSD, nmap, iaplc)
        fname_corono_avg_AO = ('{4}_corono_nPup={0}_nImg={1}_iPSD={2:04d}' + \
                               '_nmap={3:04d}_avg.fits').format(nPup, 
                                      nImg2dbis, iPSD, nmap, iaplc)
        fname_direct_std_AO = ('{4}_direct_nPup={0}_nImg={1}_iPSD={2:04d}' + \
                               '_nmap={3:04d}_std.fits').format(nPup, 
                                      nImg2dbis, iPSD, nmap, iaplc)
        fname_corono_std_AO = ('{4}_corono_nPup={0}_nImg={1}_iPSD={2:04d}' + \
                               '_nmap={3:04d}_std.fits').format(nPup, 
                                      nImg2dbis, iPSD, nmap, iaplc)
        
        fpath_direct_avg_AO = fdir_data / fname_direct_avg_AO
        fpath_corono_avg_AO = fdir_data / fname_corono_avg_AO
        fpath_direct_std_AO = fdir_data / fname_direct_std_AO
        fpath_corono_std_AO = fdir_data / fname_corono_std_AO

        """
        ### Read image (with AO residuals)
        """
        direct_poly_img_AO[i*nPSD+iPSD] = fits.getdata(fpath_direct_AO)
        corono_poly_img_AO[i*nPSD+iPSD] = fits.getdata(fpath_corono_AO)
            
        """
        ### Read profiles (with AO residuals)
        """        
        # with AO residuals
        direct_poly_avg_AO[i*nPSD+iPSD] = fits.getdata(fpath_direct_avg_AO)
        corono_poly_avg_AO[i*nPSD+iPSD] = fits.getdata(fpath_corono_avg_AO)
        direct_poly_std_AO[i*nPSD+iPSD] = fits.getdata(fpath_direct_std_AO)
        corono_poly_std_AO[i*nPSD+iPSD] = fits.getdata(fpath_corono_std_AO)

#%%
"""
### Post processing tests on APLC images
"""
cube1 = corono_poly_img_AO[:nPSD]
cube2 = corono_poly_img_AO[nPSD:]

#%%
"""
### Mean estimate
"""
mean1 = np.mean(cube1, axis=0)
mean2 = np.mean(cube2, axis=0)

corono_poly_mean1_avg, rad_corono = imutils.profile(mean1, type='mean')
corono_poly_mean1_std, rad_corono = imutils.profile(mean1, type='std')
corono_poly_mean2_avg, rad_corono = imutils.profile(mean2, type='mean')
corono_poly_mean2_std, rad_corono = imutils.profile(mean2, type='std')

#%%
"""
### Display of the coronagraphic images for both APLCs
"""
pl.figure(1, (10,5))
pl.clf() 
pl.subplot(1,2,1)
pl.imshow(np.log10(mean1), 
          vmin=vmin0, vmax=vmax0, cmap='inferno')
pl.title('Current APLC (mean)')

pl.subplot(1,2,2)
pl.imshow(np.log10(mean2), 
          vmin=vmin0, vmax=vmax0, cmap='inferno')
pl.title('New APLC (mean)')
pl.show()      


#%%
"""
### Median estimate
"""
med1 = np.median(cube1, axis=0)
med2 = np.median(cube2, axis=0)

corono_poly_med1_avg, rad_corono = imutils.profile(med1, type='mean')
corono_poly_med1_std, rad_corono = imutils.profile(med1, type='std')
corono_poly_med2_avg, rad_corono = imutils.profile(med2, type='mean')
corono_poly_med2_std, rad_corono = imutils.profile(med2, type='std')

#%%
pl.figure(2, (10,5))
pl.clf() 
pl.subplot(1,2,1)
pl.imshow(np.log10(med1), 
          vmin=vmin0, vmax=vmax0, cmap='inferno')
pl.title('Current APLC (median)')

pl.subplot(1,2,2)
pl.imshow(np.log10(med2), 
          vmin=vmin0, vmax=vmax0, cmap='inferno')
pl.title('New APLC (median)')
pl.show() 

#%%
"""
### Median subtraction
"""
cube1bis = np.zeros((nPSD,nImg2dbis, nImg2dbis))
cube2bis = np.zeros((nPSD,nImg2dbis, nImg2dbis))
for iPSD in range(nPSD):
    cube1bis[iPSD] = cube1[iPSD] - med1
    cube2bis[iPSD] = cube2[iPSD] - med2

mean1bis = np.mean(cube1bis, axis=0)
mean2bis = np.mean(cube2bis, axis=0)

corono_poly_mean1bis_avg, rad_corono = imutils.profile(mean1bis, type='mean')
corono_poly_mean1bis_std, rad_corono = imutils.profile(mean1bis, type='std')
corono_poly_mean2bis_avg, rad_corono = imutils.profile(mean2bis, type='mean')
corono_poly_mean2bis_std, rad_corono = imutils.profile(mean2bis, type='std')

#%%
pl.figure(3, (10,5))
pl.clf() 
pl.subplot(1,2,1)
pl.imshow(np.log10(mean1bis), 
          vmin=vmin0, vmax=vmax0, cmap='inferno')
pl.title('Current APLC (median-subtracted mean)')

pl.subplot(1,2,2)
pl.imshow(np.log10(mean2bis), 
          vmin=vmin0, vmax=vmax0, cmap='inferno')
pl.title('New APLC (median-subtracted mean)')
pl.show() 

#%%
"""
### Plot comparison
"""
fname_image_plane_plt = ('aplcs_corono_nPup={0}_nImg={1}_iPSD={2:04d}' + \
                         '_nmap={3:04d}_plt').format(nPup, nImg2dbis, 
                                iPSD, nmap)
fpath_image_plane_plt_pdf = (fdir_pdf / fname_image_plane_plt).with_suffix('.pdf')
fpath_image_plane_plt_png = (fdir_pdf / fname_image_plane_plt).with_suffix('.png')

rad_corono = np.arange(nImg2dbis//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,2))

lines_noturb = []
lines_siturb = []

fig = pl.figure(13, (8, 4.5))
pl.clf()
ax = fig.add_subplot(111)


ax.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_mean1bis_std,
        color = colors_cor[0], ls='-.')
#
ax.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_mean2bis_std,
       color = colors_cor[1], ls='-.')

ax.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_mean1_std,
        color = colors_cor[0], ls='-')
#
ax.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_mean2_std,
       color = colors_cor[1], ls='-')

dummy_lines = []
dummy_lines += ax.semilogy([], [], ls='-', color='k')
dummy_lines += ax.semilogy([], [], ls='-.', color='k')

dummy_lines2 = []
for i in range(naplc):
    dummy_lines2 += ax.semilogy([], [], ls='', color=colors_cor[i])

ax.axvspan(-1, rMask, alpha=0.25, color='b')
ax.set_xlabel(r'Angular separation in $\lambda_0$/D')
ax.set_ylabel(r'5$\sigma$ normalized intensity in log scale')
ax.set_xlim(-0.5, 30.5)
ax.set_ylim(3e-8, 3e-4)
ax.grid(True, which='both')
#ax.legend()

leg1 = ax.legend(dummy_lines, [r'AO residuals'.format(
        int(nmap*temp_freq/1000)),'Median sub'], 
             loc='lower right', frameon=False)
ax.add_artist(leg1);

leg2 = ax.legend(dummy_lines2, ['Current APLC', 'New APLC'],
             loc='upper right', frameon=False)
for i, text in enumerate(leg2.get_texts()):
    pl.setp(text, color = colors_cor[i])
ax.add_artist(leg2);

ax.set_title(r'Broadband intensity profile ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100,))
pl.tight_layout()
#pl.savefig(str(fpath_image_plane_plt_pdf), tight=True, transparent=True)
#pl.savefig(str(fpath_image_plane_plt_png), tight=True)
    

#%%
"""
### median subtraction using VIP package
"""
fwhm0 = 2*1.593/0.950
PA_ini = -15
PA_end = 65
angs = np.empty((nPSD, 1))
angs[:,0] = np.linspace(PA_ini, PA_end, nPSD)

#fr_adi1 = vip.medsub.median_sub(cube1, angs, mode='fullfr', fwhm = fwhm0)
#fr_adi2 = vip.medsub.median_sub(cube2, angs, mode='fullfr', fwhm = fwhm0)

cube_out1, cube_der1, fr_adi1 = vip.medsub.median_sub(cube1, angs, 
                                                      mode='fullfr', 
                                                      fwhm = fwhm0, 
                                                      full_output=True)
cube_out2, cube_der2, fr_adi2 = vip.medsub.median_sub(cube2, angs, 
                                                      mode='fullfr', 
                                                      fwhm = fwhm0, 
                                                      full_output=True)


#%%
"""
### Plot of the coronagraphic images for both APLCs
"""
pl.figure(5, (10,5))
pl.clf() 
pl.subplot(1,2,1)
pl.imshow(np.log10(cube_der1[0]), 
          vmin=vmin0, vmax=vmax0, cmap='inferno')
pl.title('Current APLC (median subtraction)')

pl.subplot(1,2,2)
pl.imshow(np.log10(cube_der2[0]), 
          vmin=vmin0, vmax=vmax0, cmap='inferno')
pl.title('New APLC (median subtraction)')
pl.show()      

