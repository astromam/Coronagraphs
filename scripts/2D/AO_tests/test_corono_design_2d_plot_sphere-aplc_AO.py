#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:50:30 2019

@author: mndiaye
"""
import numpy as np
import pylab as pl
from pathlib import Path

import os
from astropy.io import fits

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


print('reading ok')
#%%
"""
### Plot comparison
"""
for iPSD in range(nPSD):
    if (iPSD+1) % 10 == 0:
        print('\niPSD: {0:03d}/{1:03d}'.format(iPSD+1, nPSD))

    fname_image_plane_plot = 'aplcs_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_plt'.format(nPup, nImg2dbis, iPSD, nmap)
    fpath_image_plane_plot_png = (fdir_pdf / fname_image_plane_plot).with_suffix('.png')
    fpath_image_plane_plot_pdf = (fdir_pdf / fname_image_plane_plot).with_suffix('.pdf')                  
    
    rad_corono = np.arange(nImg2dbis//2)
    colors_cor = pl.cm.rainbow(np.linspace(0,1,2))
    
    lines_noturb = []
    lines_siturb = []
    
    """
    ### Plot comparison
    """
    
    fig = pl.figure(12, (8, 4.5))
    pl.clf()
    ax = fig.add_subplot(111)
    
    for i, iaplc in enumerate(aplc_arr):
        ax.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_std_AO[i*nPSD+iPSD],
                color = colors_cor[i], ls='-')
    
        ax.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_std[i],
                color = colors_cor[i], ls='--')
    
    dummy_lines = []
    dummy_lines += ax.semilogy([], [], ls='-', color='k')
    dummy_lines += ax.semilogy([], [], ls='--', color='k')
    
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
    
    leg1 = ax.legend(dummy_lines, [r'AO residuals, t={0:04d}ms'.format(int(nmap*temp_freq/1000)), 'No turbulence'], 
                 loc='lower right', frameon=False)
    ax.add_artist(leg1);
    
    leg2 = ax.legend(dummy_lines2, ['Current APLC', 'New APLC'],
                 loc='upper right', frameon=False)
    for i, text in enumerate(leg2.get_texts()):
        pl.setp(text, color = colors_cor[i])
    ax.add_artist(leg2);
    
    ax.set_title(r'Broadband intensity profile ($\Delta\lambda/\lambda_0$={0:.1f}%), t={1:03d}s'.format(bw*100,iPSD))
    pl.tight_layout()
    pl.savefig(str(fpath_image_plane_plot_pdf), transparent=True, tight=True)
    pl.savefig(str(fpath_image_plane_plot_png), tight=True)    

    #%%
    """
    image comparison
    """
    fname_image_plane_img = 'aplcs_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_img'.format(nPup, nImg2dbis, iPSD, nmap)
    fpath_image_plane_img_png = (fdir_pdf / fname_image_plane_img).with_suffix('.png')
    
    pl.figure(13, (10,5))
    pl.clf()
    ax1 = pl.subplot(121)
    pl.imshow(np.log10(corono_poly_img_AO[iPSD]), cmap = 'inferno', 
              vmin=vmin0, vmax=vmax0)
    ax1.text(0.5, 0.95, 'Current APLC', fontsize=14, 
            horizontalalignment='center',
            verticalalignment='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.05, 't={0:03d}s'.format(iPSD), fontsize=14, 
            horizontalalignment='center',
            verticalalignment='center', transform=ax1.transAxes)
    
    ax2 = pl.subplot(122)
    pl.imshow(np.log10(corono_poly_img_AO[i*nPSD+iPSD]), cmap = 'inferno', 
              vmin=vmin0, vmax=vmax0)
    ax2.text(0.5, 0.95, 'New APLC', fontsize=14, 
            horizontalalignment='center',
            verticalalignment='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.05, 't={0:03d}s'.format(iPSD), fontsize=14, 
            horizontalalignment='center',
            verticalalignment='center', transform=ax2.transAxes)
    
    pl.tight_layout()
    pl.savefig(str(fpath_image_plane_img_png), tight=True)

#%%
pl.show()
