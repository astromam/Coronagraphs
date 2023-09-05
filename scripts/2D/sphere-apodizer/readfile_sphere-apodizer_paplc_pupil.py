#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 14:07:21 2023

@author: mndiaye
"""

"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from pathlib import Path

from astropy.io import fits

#%%
"""
### Parameters
"""
nPupBis = 1030
kPad = 4
nPadBis = kPad*nPupBis

dSub_nPts= 1500

# save file in fits file
do_fits = 1

# save file in png file
do_png = 1

#%%
"""
### FLoyd-Steinberg algorithm function
"""
def floyd_steinberg(array):
    
    n0 = np.shape(array)[0]
    n1 = np.shape(array)[1]
    
    new_array = array*1.
    
    for j in range(n1):
        for i in range(n0):
            
            old_pixel = new_array[i, j]
            new_pixel = 0.
            if old_pixel >0.5:
                new_pixel = 1.
            new_array[i, j] = new_pixel
            quant_error = old_pixel - new_pixel
            
            if i != n0-1: 
                new_array[i + 1, j    ] += quant_error * 7. / 16.
            if i != 0 and j != n1-1:
                new_array[i - 1, j + 1] += quant_error * 3. / 16.
            if j != n1-1:    
                new_array[i    , j + 1] += quant_error * 5. / 16.
            if i != n0-1 and j != n1-1: 
                new_array[i + 1, j + 1] += quant_error * 1. / 16.
    
    return new_array


#%%
"""
### Directory
"""
fdir_pupils = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/Coronagraphs/data/2D/pupils/').resolve()

fdir_sphere = fdir_pupils / 'SPHERE'
fdir_newapod = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/Coronagraphs/SPHERE_PAPLC_surfptv/paplc_sphere_12fpm_12iwa_010tt_robust_50bandwidth_1e8_ttconstraint_10surfptvconstraint/').resolve()

#%%
"""
### Apodizer file
"""
fname_newapod = 'pupil.fits'
fpath_newapod = fdir_newapod / fname_newapod



#%%
"""
### filepaths to save results
"""
fdir_apod = Path('/Users/mndiaye/Nextcloud/SPHERE_upgrade/WP_corono/specifications/1.3.0/').resolve()
# filename of the 2D apodizations (initial and binary) 
fname_apod2d_ini = 'paplc_pup_initial'
fname_apod2d_bin = 'paplc_pup_binary'

# filename in fits format
fname_apod2d_ini_fits = fname_apod2d_ini + f'_{dSub_nPts:04d}' + '.fits'
fname_apod2d_bin_fits = fname_apod2d_bin + f'_{dSub_nPts:04d}' + '.fits'

# filename in png format
fname_apod2d_ini_png = fname_apod2d_ini + f'_{dSub_nPts:04d}' +'.png'
fname_apod2d_bin_png = fname_apod2d_bin + f'_{dSub_nPts:04d}' +'.png'

# filepath for fits format
fpath_apod2d_ini_fits = fdir_apod / fname_apod2d_ini_fits
fpath_apod2d_bin_fits = fdir_apod / fname_apod2d_bin_fits

# filepath for png format
fpath_apod2d_ini_png = fdir_apod / fname_apod2d_ini_png
fpath_apod2d_bin_png = fdir_apod / fname_apod2d_bin_png


#%%
"""
### Read file
"""
apod = fits.getdata(fpath_newapod)
nPup = np.shape(apod)[0]

apod_area = np.sum(apod)

#%%
"""
### Zero padding of the apddizer
"""
nPad = kPad*nPup
padapod = np.pad(apod, (nPad-nPup)//2)

#%%
"""
### FFT of the apodizer
"""
fftapod = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(padapod), norm="ortho"))

#%%
"""
### Zero padding
"""
padfftapod = np.pad(fftapod, (nPadBis-nPad)//2)

#%%
"""
### Rescaled apodizer
"""
scaledpadapod = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(padfftapod), norm="ortho"))
scaledpadapod = scaledpadapod.real

#%%
"""
### Cropped scaled apodizer
"""
ini = (nPadBis-nPupBis)//2
end = (nPadBis+nPupBis)//2
scaledapod = scaledpadapod[ini:end, ini:end]

normscaledapod = scaledapod* (nPupBis/nPup)
normscaledapod[normscaledapod < 0 ] = 0

apod2d_ini = np.pad(normscaledapod, (dSub_nPts-nPupBis)//2)

#%%
"""
### Floyd Steinberg algorithm  
"""
apod2d_bin_tmp = np.round(normscaledapod) #floyd_steinberg(normscaledapod)

#%%
"""
### Pad binary apodizer to substrate dimensions
"""
apod2d_bin = np.pad(apod2d_bin_tmp, (dSub_nPts-nPupBis)//2)

#%%
"""
### Save file
"""
if do_fits:
    fits.writeto(fpath_apod2d_ini_fits, apod2d_ini, overwrite=True)
    fits.writeto(fpath_apod2d_bin_fits, apod2d_bin, overwrite=True)
 


#%%
"""
### Display initial apodizer
"""
plt.figure(0)
plt.clf()
plt.subplot(141)
plt.imshow(apod, cmap='inferno', vmin=0, vmax=1)
plt.title('apodizer')
plt.subplot(142)
plt.imshow(padapod, cmap='inferno', vmin=0, vmax=1)
plt.title('padded apodizer')
plt.subplot(143)
plt.imshow(normscaledapod, cmap='inferno', vmin=0, vmax=1)
plt.title('scaled real')
plt.subplot(144)
plt.imshow(apod2d_bin, cmap='inferno', vmin=0, vmax=1)
plt.title('binary apod')

#%%
"""
### Final apodizer for manufacturing
"""
plt.figure(1)
plt.clf()
plt.imshow(apod2d_bin, cmap='inferno')

#%%
"""
### Display 2d profile before binarisation and save it in png format
"""
plt.figure(11)
plt.clf()
plt.imshow(apod2d_ini, cmap='Greys_r')
if do_png:
    plt.savefig(fpath_apod2d_ini_png)

#%%
"""
### Display 2d profile after binarisation and save it in png format
"""
plt.figure(12)
plt.clf()
plt.imshow(apod2d_bin, cmap='Greys_r')
if do_png:
    plt.savefig(fpath_apod2d_bin_png)

