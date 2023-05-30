#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:02:28 2023

@author: asimonnin
"""
#%%
"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import pdb

import os

from mpl_toolkits.axes_grid1 import AxesGrid

import time

plt.rcParams.update({'font.size': 13})

#%%
"""
### Parameters
"""
# Pupil size
nPup = 400

# Sampling of the coronagraph focal plane mask
nFPM = 100

# Image size
nImg = 400

# OPD map number in the files
nOPD = 2000

# wavelength in m
lam = 1600e-9

# Pupil diameter in m 
D = 38.54

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# conversion lam/D to mas
lamD2mas = (lam/D)*mas2rad

# plate scale in mas per pixel
pscale = 0.3

# FPM size in lam/D in the focal plane B
mB = 4

# FoV in lam/D in the final image plane D
mD = 58.393*(nImg/1600)


#%%
def uniform_disk(n, radius, CtrBtwnPix=False):
    """
    Generates a uniform disk in a 2D array.
    
    Parameters
    ----------
    n : integer
        size of the array
    
    radius : float
        radius of the disk
    
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ----------
    res : array_like
        (ys x xs) array with a uniform disk of radius "radius".
        
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2 
    xx,yy  = np.meshgrid(np.arange(n)-n/2+val, np.arange(n)-n/2+val)
    mydist = np.hypot(yy,xx)
    res    = np.zeros_like(mydist)
    # res[mydist <= radius] = 1.0
    res[mydist < radius] = 1.0
    return res

#%%
def profile(img, ptype='mean', step=1, mask=None, center=None, rmax=0, clip=True, exact=False):
    '''
    Azimuthal statistics of an image

    Parameters
    ----------
    img : array
        Image on which the profiles
        
    ptype : str, optional
        Type of profile. Allowed values are mean, std, var, median, min, max. Default is mean.
    
    mask : array, optional
        Mask for invalid values (must have the same size as image)
        
    center : array_like, optional
        Center of the image

    rmax : float
        Maximum radius for calculating the profile, in pixel. Default is 0 (no limit)
    
    clip : bool, optional
        Clip profile to area of image where there is a full set of data
        
    exact : bool, optional
        Performs an exact estimation of the profile. This can be very long for 
        large arrays. Default is False, which rounds the radial distance to the 
        closest 1 pixel.
    
    Returns
    -------
    prof : array
        1D profile vector
        
    rad : array
        Separation vector, in pixel
    '''
    
    # make sure we work on a copy
    img = img.copy()
    
    # array dimensions
    dimx = img.shape[1]
    dimy = img.shape[0]

    # center
    if center is None:
        center = (dimx // 2, dimy // 2)

    # masking
    if mask is not None:
        # check size
        if mask.shape != img.shape:
            raise ValueError('Image and mask don''t have the same size. Returning.')

        img[mask == 0] = np.nan
        
    # intermediate cartesian arrays
    x = np.arange(dimx, dtype=np.int64) - center[0]
    y = np.arange(dimy, dtype=np.int64) - center[1]
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    
    # rounds for faster calculation
    if not exact:
        rr = np.round(rr, decimals=0)
    
    # find unique radial values
    uniq = np.unique(rr, return_inverse=True, return_counts=True)
    r_uniq_val = uniq[0]
    r_uniq_inv = uniq[1]
    r_uniq_cnt = uniq[2]

    # number of elements
    if clip:
        extr  = np.abs(np.array((x[0], x[-1], y[0], y[-1])))
        r_max = extr.min()
        i_max = int(r_uniq_val[r_uniq_val <= r_max].size)
    else:
        r_max = r_uniq_val.max()
        i_max = r_uniq_val.size

    # limit extension of profile
    if (rmax > 0):
        r_max = rmax
        i_max = int(r_uniq_val[r_uniq_val <= r_max].size)
        
    t_max = r_uniq_cnt[0:i_max].max()

    # intermediate polar array
    polar = np.empty((i_max, t_max), dtype=img.dtype)
    polar.fill(np.nan)
    
    img_flat = img.ravel()
    for r in range(i_max):
        cnt = r_uniq_cnt[r]
        val = img_flat[r_uniq_inv == r]
        polar[r, 0:cnt] = val
            
    # calculate profile
    rad  = r_uniq_val[0:i_max]

    ptype = ptype.lower()
    if step == 1:
        # fast statistics if step=1
        if ptype == 'mean':
            prof = np.nanmean(polar, axis=1)
        elif ptype == 'std':
            prof = np.nanstd(polar, axis=1, ddof=1)
        elif ptype == 'var':
            prof = np.nanvar(polar, axis=1)
        elif ptype == 'median':
            prof = np.nanmedian(polar, axis=1)
        elif ptype == 'min':
            prof = np.nanmin(polar, axis=1)
        elif ptype == 'max':
            prof = np.nanmax(polar, axis=1)
        else:
            raise ValueError('Unknown statistics ptype = {0}. Allowed values are mean, std, var, median, min and max'.format(ptype))
    else:
        # slower if we need step > 1
        prof = np.zeros(i_max, dtype=img.dtype)
        for r in range(i_max):
            idx = ((rad[r]-step/2) <= rad) & (rad <= (rad[r]+step/2))
            val = polar[idx, :]
            
            if ptype == 'mean':
                prof[r] = np.nanmean(val)
            elif ptype == 'std':
                prof[r] = np.nanstd(val)
            elif ptype == 'var':
                prof[r] = np.nanvar(val)
            elif ptype == 'median':
                prof[r] = np.nanmedian(val)
            elif ptype == 'min':
                prof[r] = np.nanmin(val)
            elif ptype == 'max':
                prof[r] = np.nanmax(val)
            else:
                raise ValueError('Unknown statistics ptype = {0}. Allowed values are mean, std, var, median, min and max'.format(ptype))

    return prof, rad


#%%
user = 'Adrien'
if user == 'Adrien':
    # File directory
    fdir_dat = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/data').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/').resolve()
    
    # Directory for the pupils
    fdir_pupil = fdir_dat / 'Pupil'

    # Directory for the PSF generated by Anne-Laure Cheffaut
    fdir_psf = fdir_dat / 'PSF' 

    # Filename and path for the ELT pupil
    fname_elt = 'Tel-Pupil.fits'
    fpath_elt = fdir_pupil / fname_elt

    # Filename for the PSF generated by Anne-Laure Cheffaut 
    flist_psf = os.listdir(fdir_psf)
    fpath_psf = [fdir_psf / flist_psf[i] for i in range(len(flist_psf))]
    
coro_config = "lyot" #'aplc2' #

#%%


diff_conf =  ['1','2','3']

mB_conf = [3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0]
contrast=np.zeros((5,len(diff_conf),len(mB_conf)))
size_fpm=mB_conf
contrast0 = np.zeros((5,len(diff_conf),len(mB_conf)))

count=0
for lyot_config in diff_conf :
    fdir_res   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/').resolve()
    
    count2=0
    disque_fpm = uniform_disk(nPup,mB*1600/(2*38))
    """without apodizer psf"""
    fname_Int_DD02 = 'wonoise_psf_lyot_'+str(lyot_config)+'.fits'
    fpath_Int_DD02 = fdir_res / fname_Int_DD02

    Int_DD02= fits.getdata(fpath_Int_DD02)

    Int_DD0_prf_avg2, rad_DD0_prf_avg2 = profile(Int_DD02, ptype='mean')
    Int_DD0_prf_std2, rad_DD0_prf_std2 = profile(Int_DD02, ptype='std')
            
    fname_Int_D02 = 'seed=' + str(12345) + '_psf_lyot_'+str(lyot_config)+'.fits'
    fpath_Int_D02 = fdir_res / fname_Int_D02

    Int_D02 = fits.getdata(fpath_Int_D02)

    Int_D0_prf_avg2, rad_D0_prf_avg2 = profile(Int_D02, ptype='mean')
    Int_D0_prf_std2, rad_D0_prf_std2 = profile(Int_D02, ptype='std')
                    
    rad_D0_prf_avg_lamD2 = rad_D0_prf_avg2 * mD/nImg 
    rad_DD0_prf_avg_lamD2 = rad_DD0_prf_avg2 * mD/nImg 

    rad_D0_prf_avg_mas2 = rad_D0_prf_avg_lamD2 / 38.54*1.6e-6/rad2mas
    rad_DD0_prf_avg_mas2 = rad_DD0_prf_avg_lamD2 / 38.54*1.6e-6/rad2mas
            
    if lyot_config == '1':
        conf1 = Int_D02
    elif lyot_config == '2':
        conf2 = Int_D02
    elif lyot_config == '3':
        conf3 = Int_D02
            

    for mB in mB_conf :#np.linspace(3,4.8,16): #range(4,5):  #
        fdir_plt = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/').resolve()
        fdir_res   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/').resolve()
        if lyot_config == '2':
            fdir_plt = fdir_plt / 'lyot_conf2_mB={0}'.format(mB)
            fdir_res = fdir_res / 'lyot_conf2_mB={0}'.format(mB)
        elif lyot_config == '3':
            fdir_plt = fdir_plt / 'lyot_conf3_mB={0}'.format(mB)
            fdir_res = fdir_res / 'lyot_conf3_mB={0}'.format(mB)
        elif lyot_config == '4':
            fdir_plt = fdir_plt / 'lyot_conf4_mB={0}'.format(mB)
            fdir_res = fdir_res / 'lyot_conf4_mB={0}'.format(mB)
        else :
            fdir_plt = fdir_plt / 'lyot_base_mB={0}'.format(mB)
            fdir_res = fdir_res / 'lyot_base_mB={0}'.format(mB)
        if not os.path.exists(fdir_plt):
            os.makedirs(fdir_plt)
        if not os.path.exists(fdir_res):
            os.makedirs(fdir_res)
            
        """with Apodizer"""
        
        # fname_Int_DD0 = 'wonoise_'+coro_config+'.fits'
        # fpath_Int_DD0 = fdir_res / fname_Int_DD0

        # Int_DD0= fits.getdata(fpath_Int_DD0)

        # Int_DD0_prf_avg, rad_DD0_prf_avg = profile(Int_DD0, ptype='mean')
        # Int_DD0_prf_std, rad_DD0_prf_std = profile(Int_DD0, ptype='std')
                
        # fname_Int_D0 = 'seed=' + str(12345) + '_psf'+coro_config+'.fits'
        # fpath_Int_D0 = fdir_res / fname_Int_D0

        # Int_D0 = fits.getdata(fpath_Int_D0)

        # Int_D0_prf_avg, rad_D0_prf_avg = profile(Int_D0, ptype='mean')
        # Int_D0_prf_std, rad_D0_prf_std = profile(Int_D0, ptype='std')
                        
        # rad_D0_prf_avg_lamD = rad_D0_prf_avg * mD/nImg 
        # rad_DD0_prf_avg_lamD = rad_DD0_prf_avg * mD/nImg 

        # rad_D0_prf_avg_mas = rad_D0_prf_avg_lamD / 38.54*1.6e-6/rad2mas
        # rad_DD0_prf_avg_mas = rad_DD0_prf_avg_lamD / 38.54*1.6e-6/rad2mas

        """with Apodizer"""
        # fname_Int_DD = 'wonoise_cor_'+coro_config+'.fits'

        """without Apodizer"""
        fname_Int_DD2 = 'wonoise_cor.fits'

        # filepath for the direct and coronagraphic images
       
        # fpath_Int_DD  = fdir_res / fname_Int_DD

        fpath_Int_DD2  = fdir_res / fname_Int_DD2

        # save the direct and coronagraphic images
        
        # Int_DD = fits.getdata(fpath_Int_DD)

        Int_DD2 = fits.getdata(fpath_Int_DD2)
        
      
        
        """
        ### Compute the radial intensity profiles of the images without turbulence
        # """
        # # computation of the averaged intensity profiles of the images   
        # Int_DD_prf_avg, rad_DD_prf_avg = profile(Int_DD, ptype='mean')
        Int_DD_prf_avg2, rad_DD_prf_avg2 = profile(Int_DD2, ptype='mean')
       

        # computation of the standard deviation intensity profiles of the images
        # Int_DD_prf_std, rad_DD_prf_std = profile(Int_DD, ptype='std')
        Int_DD_prf_std2, rad_DD_prf_std2 = profile(Int_DD2, ptype='std')

            
        """
        ### Cropping of the PSF generated by Anne-Laure Cheffaut
        """
        
        for iPSF in range(0,1):
            # Read the PSF generated by Anne-Laure Cheffaut
            PSF_alc = fits.getdata(fpath_psf[iPSF],)
        
            # header data unit
            hdu = fits.open(fpath_psf[iPSF])
        
            # read header
            hdr = hdu[0].header
        
            # get the seed valie
            seed = hdr['RNGSEED']
            # Size of the original image
            nImg_alc = np.size(PSF_alc, 0)
        
            # dimensions to crop the images to nImg
            nIni = (nImg_alc-nImg)//2
            nEnd = (nImg_alc+nImg)//2
        
            # crop the images to nImg
            PSF_alc1 = PSF_alc[nIni:nEnd,nIni:nEnd]
        
            # flip image upd-down and left-right
            PSF_alc1 = np.flipud(np.fliplr(PSF_alc1))
        
            # normalize image
            PSF_alc1 /= np.max(PSF_alc1)
            
            
            # filename for the direct and coronagraphic images
            """with apodizer"""
            # fname_Int_D = 'seed=' + str(seed) + '_cor_'+coro_config+'.fits'

            """without apodizer"""
            fname_Int_D2 = 'seed=' + str(seed) + '_cor.fits'
          
            # filepath for the direct and coronagraphic images
            
            # fpath_Int_D  = fdir_res / fname_Int_D

            fpath_Int_D2  = fdir_res / fname_Int_D2
          
            # save the direct and coronagraphic images
            
            # Int_D = fits.getdata(fpath_Int_D)

            Int_D2 = fits.getdata(fpath_Int_D2)

            
            # Int_D_prf_avg, rad_D_prf_avg = profile(Int_D, ptype='mean')

            Int_D_prf_avg2, rad_D_prf_avg2 = profile(Int_D2, ptype='mean')

            PSF_alc1_prf_avg, rad_D_prf_avg = profile(PSF_alc1, ptype='mean')


    
            # computation of the standard deviation intensity profiles of the images
            # Int_D_prf_std, rad_D_prf_std = profile(Int_D, ptype='std')

            Int_D_prf_std2, rad_D_prf_std2 = profile(Int_D2, ptype='std')

            PSF_alc1_prf_std, rad_D0_prf_std = profile(PSF_alc1, ptype='std')

            """
            ### Display images
            """
            # filename of the plot
            # fname_images = 'seed=' + str(seed) + '_images_alc_'+coro_config+'_back_up.pdf'

            fname_images2 = 'seed=' + str(seed) + '_images_alc_withoutapodizer.pdf'

            taille_FPM = lamD2mas*mB/2
            FPM = uniform_disk(nPup, taille_FPM/pscale)
    
            # filepath for the direct and coronagraphic images
            # fpath_images = fdir_plt / fname_images

            fpath_images2 = fdir_plt / fname_images2
        
            # boundaries for the images in log scale
            vmin0 = -5
            vmax0 = 0
        
            fig = plt.figure(1, figsize=(12,6))
            plt.clf()
        
            grid = AxesGrid(fig, 111,
                        nrows_ncols=(2, 2),
                        axes_pad=0.3,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.2
                        )
        
            # Perfect PSF
            im = grid[0].imshow(np.log10(Int_DD02), vmin=vmin0, vmax=vmax0, cmap='inferno')
            
            grid[0].set_title('perfect PSF')
    
            # AO corrected PSF (MND)
            im = grid[1].imshow(np.log10(Int_D02), vmin=vmin0, vmax=vmax0, cmap='inferno')
            
            grid[1].set_title(f'AO corrected PSF ')
        
            # # AO corrected PSF (ALC)
            # im = grid[2].imshow(np.log10(PSF_alc1), vmin=vmin0, vmax=vmax0, cmap='inferno')
            # grid[2].set_title(f'AO corrected PSF (ALC)')
        
            # Perfect coronagraphic image
            im = grid[2].imshow(np.log10(Int_DD2), vmin=vmin0, vmax=vmax0, cmap='inferno')
            grid[2].contour(FPM, colors='white')
            grid[2].set_title('Perfect coro. image')
        
            # AO corrected coronagraphic image
            im = grid[3].imshow(np.log10(Int_D2), vmin=vmin0, vmax=vmax0, cmap='inferno')
            grid[3].contour(FPM, colors='white')
            grid[3].set_title('AO corrected coro. image')
        
            # colorbar
            cbar = grid[0].cax.colorbar(im)
            cbar = grid.cbar_axes[0].colorbar(im)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel('Intensity in log scale', rotation=270)
        
            plt.tight_layout()
            plt.savefig(fpath_images2)
            plt.close()
            """
            ### plot the radial profiles of the image intensity
            """
            # filename of the plot
            fname_prf = 'seed=' + str(seed) + '_profiles_in_mas_2plots_'+coro_config+'_lambda_1200.pdf'
        
            # filepath for the direct and coronagraphic images
            fpath_prf = fdir_plt / fname_prf
        
        
            # convert pixel scale into lam/D scale for the x-axis
            
            # rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg

            rad_D_prf_avg_lamD2 = rad_D_prf_avg2 * mD/nImg
            
            
            # rad_DD_prf_avg_lamD = rad_DD_prf_avg * mD/nImg

            rad_DD_prf_avg_lamD2 = rad_DD_prf_avg2 * mD/nImg


            #  convert lam/D into mas for the x-axis

            # rad_D_prf_avg_mas = rad_D_prf_avg_lamD / 38.54*1.6e-6/rad2mas
            
            rad_D_prf_avg_mas2 = rad_D_prf_avg_lamD2 / 38.54*1.6e-6/rad2mas
            
            # rad_DD_prf_avg_mas = rad_DD_prf_avg_lamD / 38.54*1.6e-6/rad2mas

            rad_DD_prf_avg_mas2 = rad_DD_prf_avg_lamD2 / 38.54*1.6e-6/rad2mas

            
            
            # value of contrast for some angular separation in mas

            angular_sep = [np.where(rad_D_prf_avg_mas2 >= 14.0)[0][0],np.where(rad_D_prf_avg_mas2 >= 17.0)[0][0],np.where(rad_D_prf_avg_mas >= 25.0)[0][0],np.where(rad_D_prf_avg_mas2 >= 33.0)[0][0],np.where(rad_D_prf_avg_mas2 >= 40.0)[0][0]]


            
            for size_1 in range(len(angular_sep)):
                contrast[size_1,count,count2] = Int_D_prf_avg2[angular_sep[size_1]]
                contrast0[size_1,count,count2] = Int_D0_prf_avg2[angular_sep[size_1]]
            
            
            # plot of the radial profiles
            plt.figure(2, (8, 4.5))
            plt.clf()
        
            # AO corrected PSF (MND)
            
            plt.plot(rad_D0_prf_avg_mas2, Int_D0_prf_avg2, label='PSF ',color='blue')
            #plt.plot(rad_D0_prf_avg_mas2, Int_D0_prf_avg2, label='PSF without apodizer ',color='blue',ls='--',alpha=0.2)
            # plt.plot(rad_DD0_prf_avg_mas, Int_DD0_prf_avg, label='PSF without turbulence',color='blue',linestyle='dotted',alpha=0.5)
            # AO corrected PSF (ALC)
            #if lyot_config =='1' :
             #   plt.plot(rad_D0_prf_avg_mas, PSF_alc1_prf_avg, label='PSF (ALC)', ls='--',color='orange')
        
            # AO corrected coronagraphic image
         
            plt.plot(rad_D_prf_avg_mas2, Int_D_prf_avg2, label='corono',color='forestgreen')
            # plt.plot(rad_DD_prf_avg_mas, Int_DD_prf_avg, label='corono without Turubulence',color='forestgreen',linestyle='dotted',alpha=0.5)
            #plt.plot(rad_D_prf_avg_mas2, Int_D_prf_avg2, label='corono without Apodizer',color='forestgreen',ls='--',alpha=0.2)
            # Focal plane mask boundary
            x = np.arange(0.0, mB/2/ 38.54*1.2e-6/rad2mas, 0.01)
            plt.axvline(x=mB/2/ 38.54*1.2e-6/rad2mas, color='k', ls='--')
        
            # Focal plane mask grey area
            
            plt.fill_between(x, 0, mB/2/ 38.54*1.2e-6/rad2mas, color='gray', alpha=0.3)
            #plt.xlim(-0.05, np.max(rad_D_prf_avg_lamD)+0.05)
            plt.xlim(-0.05,55)
            plt.ylim(2e-7, 2e0)
            #else : 
             #   plt.ylim(2e-5, 2e0)
            plt.xlabel('Angular separation [mas]')
            plt.ylabel('Normalized intensity in log scale')
            plt.yscale('log')        
            lam=1.2e-6
            plt.title(f'Radial averaged intensity profile at $\lambda$={lam*1e6:.3f}$\mu$m')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(fpath_prf)
            plt.close()
            #plt.show()

            plt.figure(3, (8, 4.5))
            plt.clf()
        
            # AO corrected PSF (MND)
            # perfect = True
            # if perfect == True : 
            #     plt.plot(rad_DD0_prf_avg_mas, Int_DD0_prf_avg, label='PSF ',color='blue')
            # else : 
            #     plt.plot(rad_D0_prf_avg_mas, Int_D0_prf_avg, label='PSF ',color='blue')
        
            # # AO corrected PSF (ALC)
            # if lyot_config =='1' :
            #     plt.plot(rad_D0_prf_avg_lamD, PSF_alc1_prf_avg, label='PSF (ALC)', ls='--',color='orange')
        
            # # AO corrected coronagraphic image
             
            # if perfect == True : 
            #     plt.plot(rad_DD_prf_avg_mas2, Int_DD_prf_avg, label='corono',color='tab:green')
            #     fname_prf = 'seed=' + str(seed) + '_profiles_without_turbulence_in_mas_'+coro_config+'.pdf'
            #     fpath_prf = fdir_plt / fname_prf
            # else :
            #     plt.plot(rad_D_prf_avg_mas, Int_D_prf_avg, label='corono',color='forestgreen')
        
            # # Focal plane mask boundary
            # x = np.arange(0.0, mB/2/ 38.54*1.6e-6/rad2mas, 0.01)
            # plt.axvline(x=mB/2/ 38.54*1.6e-6/rad2mas, color='k', ls='--')
        
            # # Focal plane mask grey area
            # plt.fill_between(x, 0, mB/2/ 38.54*1.6e-6/rad2mas, color='gray', alpha=0.3)
        
            # #plt.xlim(-0.05, np.max(rad_D_prf_avg_lamD)+0.05)
            # plt.xlim(-0.05,55)
            # if perfect == True : 
            #     plt.ylim(2e-7, 2e0)
            # #else : 
            #  #   plt.ylim(2e-5, 2e0)
            # plt.xlabel('Angular separation [mas]')
            # plt.ylabel('Normalized intensity in log scale')
            # plt.yscale('log')
            # if perfect == True : 
            #     plt.title(f'Radial averaged intensity profile at $\lambda$={lam*1e6:.3f}$\mu$m without turbulence')
            # else : 
            #     plt.title(f'Radial averaged intensity profile at $\lambda$={lam*1e6:.3f}$\mu$m')
            # plt.grid(True)
            # plt.tight_layout()
            # plt.legend()
        
            # plt.savefig(fpath_prf)
            # plt.close()
            
            count2 +=1
    count +=1

#%%
fig = plt.figure(1, figsize=(10,10))
plt.clf()
        
grid2 = AxesGrid(fig, 111,
                        nrows_ncols=(2, 1),
                        axes_pad=0.3,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.2
)
im = grid2[0].imshow(np.log10(Int_DD),vmin=-5,vmax=0)
grid2[0].set_title(f'Corono without AO correction')
im = grid2[1].imshow(np.log10(Int_D),vmin=-5,vmax=0)
grid2[1].set_title(f'Corono with AO correction')
cbar = grid2.cbar_axes[0].colorbar(im)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Intensity in log scale', rotation=270)
plt.savefig('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/lyot_conf3_mB=4.0/' + 'seed=' + str(seed) + '_corono_with_and_without_AO_correction.pdf')
plt.close()
#%%


for a in range(len(angular_sep)):
    size_fpm=np.array(size_fpm)
    limit = np.where(size_fpm/2/ 38.54*1.6e-6/rad2mas >= rad_D_prf_avg_mas[angular_sep[a]])[0]
    plt.figure(a,(8, 4.5))  
    plt.clf()
    plt.plot(size_fpm,contrast[a,0],label='Lyot Stop 1')
    plt.plot(size_fpm,contrast[a,1],label = 'Lyot Stop 2')
    plt.plot(size_fpm,contrast[a,2],label='Lyot Stop 3')
    plt.plot(size_fpm,contrast0[a,0],label='No FPM', ls='--',color='tab:blue',alpha=0.3)
    plt.plot(size_fpm,contrast0[a,1],label='No FPM', ls='--',color='tab:orange',alpha=0.3)
    plt.plot(size_fpm,contrast0[a,2],label='No FPM', ls='--',color='tab:green',alpha=0.3)
    if limit != [] :
        x = np.arange(size_fpm[limit[0]],6.0, 0.01)
        plt.fill_between(x,0,size_fpm[limit[0]], color='gray', alpha=0.3)
        plt.axvline(x=size_fpm[limit[0]], color='k', ls='--')
    plt.ylim(2e-5, 1e-1)
    plt.xlabel(r'Size of the FPM [$\lambda$/D]')
    plt.ylabel('Contrast')
    plt.yscale('log')
    plt.title('Contrast vs size of the FPM at '+str(int(rad_D_prf_avg_mas[angular_sep[a]]))+' mas')
    plt.legend()
    plt.grid(True)
    #plt.tight_layout()
    plt.savefig('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/contrast_vs_size_fpm_spearation_'+str(int(rad_D_prf_avg_mas[angular_sep[a]]))+'_test2.pdf')
    plt.close()        
    #plt.show()    
        
        
#%% 

mB = 4.0 


        
       
        
        
# %%
