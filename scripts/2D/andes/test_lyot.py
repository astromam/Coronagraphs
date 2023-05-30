#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:29 2023

@author: mndiaye
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
import glob

import os

from mpl_toolkits.axes_grid1 import AxesGrid

import time


plt.rcParams.update({'font.size': 20})

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
bande = 'J'
if bande =='H':
    lam = 1600e-9

elif bande =='J':
    lam = 1200e-9

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
"""
### Functions
"""
def sft(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square. 

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    
    sign = -1.0
    if inv:
        sign = 1.0

    U = np.zeros((1,NB))
    X = np.zeros((1,NA))
    
    X[0,:] = (1./NA)*(np.arange(NA)-NA/2.+val)
    U[0,:] =  (m/NB)*(np.arange(NB)-NB/2.+val)
       
    XU = 2.*np.pi* X.T.dot(U)
    A3 = sign*1j*np.sin(XU)  +np.cos(XU)
    A1 = A3.T
    
    B  = A1.dot(A2.dot(A3))

    return coeff*B

#%%
def isft(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].

    See Also
    --------
    sft() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

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
"""
### Working directory
"""
user = 'Adrien'
if user == 'Adrien':
    # File directory
    fdir_dat = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/data').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/').resolve()

elif user == 'Mamadou':
    # File directory
    fdir_dat = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/data/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/results/').resolve()
    # Directory for the OPD with the corresponding seed value
    fdir_plt   = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/andes/plots/').resolve()


# Directory for the pupils
fdir_pupil = fdir_dat / 'Pupil'
fdir_elt = '/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/data/elt/'

# Directory for the PSF generated by Anne-Laure Cheffaut
fdir_psf = fdir_dat / 'PSF' 

# Filename and path for the ELT pupil
fname_elt = 'Tel-Pupil.fits'
fpath_elt = fdir_pupil / fname_elt

# Filename for the PSF generated by Anne-Laure Cheffaut 
flist_psf = os.listdir(fdir_psf)
fpath_psf = [fdir_psf / flist_psf[i] for i in range(len(flist_psf))]

apodizer_files = np.sort(glob.glob(fdir_elt + 'pupilelt_nPup400_A**'))

coro_config = "lyot" 

###  Test differents parameters
# diametre_lyot = [0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00]#0.9,0.92
# obstruction = [0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,]
# mB_conf = [3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0]

### Optimal parameters

diametre_lyot = [0.9]
obstruction = [0.4]
mB_conf = [3.8]

result = np.zeros((len(diametre_lyot),len(obstruction),len(mB_conf)))


### To have all seeds

Int_D_all =[]
Int_D0_all=[]
Rad_D_all=[]
Rad_D0_all=[]

fdir_res   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/').resolve()
i=0
# frac_lum = np.load(fdir_res / 'fraction_lum.npy')
#%%
   
for diam in diametre_lyot :
    j=0 
    for obst in obstruction :
        k=0
        for mB in mB_conf :
            """without apodizer psf"""

            # fdir_plt = fdir_plt / 'lyot_diam_{diam}_obst_{obst}_mB={mB}'.format(diam=diam,obst = obst,mB=mB)
            fdir_res   = Path('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/').resolve()
            fdir_res = fdir_res / 'lyot_diam_{diam}_obst_{obst}_mB={mB}_bande_J'.format(diam=diam,obst = obst,mB=mB)
            
            fname_Int_DD = 'wonoise_cor_'+coro_config+'.fits'
            fname_Int_DD0 = 'wonoise_lyot.fits'

            fpath_Int_DD0 = fdir_res / fname_Int_DD0
            fpath_Int_DD = fdir_res / fname_Int_DD

            Int_DD= fits.getdata(fpath_Int_DD)
            Int_DD0= fits.getdata(fpath_Int_DD0)

            Int_DD0_prf_avg, rad_DD0_prf_avg = profile(Int_DD0, ptype='mean')
            Int_DD0_prf_std, rad_DD0_prf_std = profile(Int_DD0, ptype='std')
            

            Int_DD_prf_avg, rad_DD_prf_avg = profile(Int_DD, ptype='mean')
            Int_DD_prf_std, rad_DD_prf_std = profile(Int_DD, ptype='std')

            rad_DD_prf_avg_lamD = rad_DD_prf_avg * mD/nImg 
            rad_DD0_prf_avg_lamD = rad_DD0_prf_avg * mD/nImg 

            rad_DD_prf_avg_mas = rad_DD_prf_avg_lamD / 38.54*lam/rad2mas
            rad_DD0_prf_avg_mas = rad_DD0_prf_avg_lamD / 38.54*lam/rad2mas

            # plt.imshow(np.log10(Int_DD), cmap='inferno')
            # plt.colorbar()
            # plt.show()
            # print(Int_DD_prf_avg[np.where(rad_DD_prf_avg_mas >= 20)[0][0]])
            # print(print(result[i,j,k]))
            
            ### When search best parameters
            
            # basse_sep = np.where(rad_DD_prf_avg_mas >= 20)[0][0]
            # haut_sep = np.where(rad_DD_prf_avg_mas >= 50)[0][0]
            # mean = np.mean(Int_DD_prf_avg[basse_sep:haut_sep])
            # result[i][j][k] = mean

            ### When we have it

            # If only one seed

            # fname_Int_D = 'seed=12345_cor_'+coro_config+'.fits'
            # fname_Int_D0 = 'seed=12345_psf_'+coro_config+'.fits'
            # fname_alc = 'seed=12345_alc_'+coro_config+'.fits'

            # fpath_Int_D0 = fdir_res / fname_Int_D0
            # fpath_Int_D = fdir_res / fname_Int_D
            # fpath_alc = fdir_res / fname_alc


            ### for all Seed 
             
            Int_D_all_file = glob.glob('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/lyot_diam_'+str(diam)+'_obst_'+str(obst)+'_mB='+str(mB)+'_bande_J/seed=*_cor_'+coro_config+'.fits')
            Int_D0_all_file = glob.glob('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/results/lyot_diam_'+str(diam)+'_obst_'+str(obst)+'_mB='+str(mB)+'_bande_J/seed=*_psf_'+coro_config+'.fits')

            # print(Int_D_all_file)
            for opd in range(len(Int_D_all_file)) :
                Int_D= fits.getdata(Int_D_all_file[opd])
                Int_D0= fits.getdata(Int_D0_all_file[opd])

                
                # alc = fits.getdata(fpath_alc)

                Int_D0_prf_avg, rad_D0_prf_avg = profile(Int_D0, ptype='mean')
                Int_D0_prf_std, rad_D0_prf_std = profile(Int_D0, ptype='std')
            
                Int_D_prf_avg, rad_D_prf_avg = profile(Int_D, ptype='mean')
                Int_D_prf_std, rad_D_prf_std = profile(Int_D, ptype='std')

                # alc_avg, rad_alc_avg = profile(alc, ptype='mean')
                # alc_std, rad_alc_std = profile(alc, ptype='std')

                rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg 
                rad_D0_prf_avg_lamD = rad_D0_prf_avg * mD/nImg 
                # alc_avg_lamD = rad_alc_avg * mD/nImg

                rad_D_prf_avg_mas = rad_D_prf_avg_lamD / 38.54*lam/rad2mas
                rad_D0_prf_avg_mas = rad_D0_prf_avg_lamD / 38.54*lam/rad2mas
                # rad_alc_avg_mas = alc_avg_lamD / 38.54*lam/rad2mas

                Int_D_all.append(Int_D_prf_avg)
                Int_D0_all.append(Int_D0_prf_avg)

                Rad_D_all.append(rad_D_prf_avg_mas)
                Rad_D0_all.append(rad_D0_prf_avg_mas)


            # plt.plot(rad_DD0_prf_avg_mas, Int_DD0_prf_avg, label='without coro ',color='blue')
            # plt.plot(rad_DD_prf_avg_mas, Int_DD_prf_avg, label='with coro ',color='green')
            # plt.yscale('log')
            # plt.xlabel('Angular separation [mas]')
            # plt.ylabel('Normalized intensity in log scale')
            # plt.legend()
            # plt.show()
            # pdb.set_trace()

            k+=1
        j+=1
    i+=1
#%%
mean_Int_D = np.mean(Int_D_all,axis=0)
mean_Int_D0 = np.mean(Int_D0_all,axis=0)

for i in range(len(Int_D_all)) :
    plt.plot(Rad_D_all[i], Int_D_all[i],color='lime',alpha=0.3)
    plt.plot(Rad_D0_all[i], Int_D0_all[i],color='cyan',alpha=0.3)
plt.plot(rad_D0_prf_avg_mas, mean_Int_D0, label='without coro ',color='blue')
plt.plot(rad_D_prf_avg_mas, mean_Int_D, label='with coro ',color='green')
plt.yscale('log')
plt.xlabel('Angular separation [mas]')
plt.ylabel('Normalized intensity in log scale')

x = np.arange(0.0, mB/2/ 38.54*lam/rad2mas, 0.01)
plt.axvline(x=mB/2/ 38.54*lam/rad2mas, color='k', ls='--')
        
# Focal plane mask grey area
plt.fill_between(x, 0, mB/2/ 38.54*lam/rad2mas, color='gray', alpha=0.3)

plt.xlim(-0.05, np.max(rad_D_prf_avg_mas)+0.05)
# plt.xlim(-0.05,55)
plt.ylim(2e-7, 2e0)
plt.title(f'Radial averaged intensity profile at $\lambda$={lam*1e6:.3f}$\mu$m')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% 

plt.plot(rad_D0_prf_avg_mas, Int_D0_prf_avg, label='without coro ',color='blue')
plt.plot(rad_D_prf_avg_mas, Int_D_prf_avg, label='with coro ',color='green')
plt.plot(rad_DD0_prf_avg_mas, Int_DD0_prf_avg, label='without coro without turbulence ',color='blue',ls= '--')
plt.plot(rad_DD_prf_avg_mas, Int_DD_prf_avg, label='with coro without turbulence',color='green',ls='--')

# plt.plot(rad_alc_avg_mas, alc_avg, label='ALC ',color='red')
plt.yscale('log')
plt.xlabel('Angular separation [mas]')
plt.ylabel('Normalized intensity in log scale')

x = np.arange(0.0, mB/2/ 38.54*lam/rad2mas, 0.01)
plt.axvline(x=mB/2/ 38.54*lam/rad2mas, color='k', ls='--')
        
# Focal plane mask grey area
plt.fill_between(x, 0, mB/2/ 38.54*lam/rad2mas, color='gray', alpha=0.3)

plt.xlim(-0.05, np.max(rad_D_prf_avg_mas)+0.05)
# plt.xlim(-0.05,55)
plt.ylim(2e-7, 2e0)
plt.title(f'Radial averaged intensity profile at $\lambda$={lam*1e6:.3f}$\mu$m')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
            # pdb.set_trace()
### Save file for test lyot conf 
#%%

# ns_np2 = result/frac_lum 
# obstruction = np.array(obstruction)
# b = np.sort((0,2,4,6,8,10,12,14,16,18,20))
# a = obstruction[b]



# fig,ax = plt.subplots(1)
# im = ax.imshow(frac_lum[:,:,0],origin='lower')
# plt.xlabel('ID in Dtel')
# plt.ylabel('OD in Dtel')
# ax.set_yticks(np.arange(len(diametre_lyot)))
# ax.set_yticklabels(diametre_lyot)
# ax.set_xticks(np.arange(len(obstruction),step=2))
# ax.set_xticklabels(a)
# plt.colorbar(im)
# plt.title('Fraction of energy through the Lyot stop')
# plt.show()
# np.save('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/test_lyot_file/result_OD_ID_mB.npy',result)
# np.save('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/test_lyot_file/ns_np2.npy',ns_np2)
# x_axis = np.array(diametre_lyot)
# y_axis = np.array(obstruction)


#%%
### parameter OD,ID,mB

# imshow[OD,ID] -> y = OD, x = ID 
# imshow[OD,mB] -> y = OD, x = mB
# imshow[ID,mB] -> y = ID, x = mB

### CHOOSE PARAMETERS
# min_frac_lum = np.where(frac_lum> 0.8)

# od_min = np.array(min_frac_lum[0])
# id_min = np.array(min_frac_lum[1])

# valeur_choisi = np.min(ns_np2[od_min,id_min,:])
# # mb_min = np.array(min_frac_lum[2])
# # min_frac_lum = np.array(min_frac_lum)

# valeurs_choisies = np.where(ns_np2 == valeur_choisi)

###FIXE FPM DIAMETER

# val_choi = np.min(ns_np2[:,:,3])
# val_choi2 = np.where(ns_np2[:,:,3]==val_choi)

#%%
# c = np.where(ns_np2 == np.min(ns_np2))
# ### Plot different imshow to find best config

# fig,ax = plt.subplots(1)
# # fig.figsize=(15,15)
# # im = ax.imshow(np.log10(result[:,c[1][0],:]), cmap='inferno',vmin=-4,vmax=-3,origin='lower')
# im = ax.imshow(np.log10(ns_np2[:,valeurs_choisies[1][0],:]), cmap='inferno',vmin=-4,vmax=-3,origin='lower',extent=[0,20,0,20])
# plt.ylabel('Lyot Stop Outer diameter')
# plt.xlabel('FPM Diameter in $\lambda$/D')
# ax.set_yticks(np.arange(len(diametre_lyot)))
# ax.set_yticklabels(diametre_lyot)
# ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
# ax.set_xticklabels(mB_conf)
# plt.axvline(x=valeurs_choisies[2]*2,linestyle='--',color='white')
# plt.axhline(y=valeurs_choisies[0],linestyle='--',color='white')
# # plt.xlim(diametre_lyot[0],diametre_lyot[-1])
# # plt.ylim(obstruction[0],obstruction[-1])
# plt.colorbar(im,label="Ns/Np² in log scale")#'Contrast in log scale')
# plt.title('Lyot Stop Inner diameter : '+str(obstruction[valeurs_choisies[1][0]]))
# # plt.contour([0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10],np.log10(result[:,10,:]),levels=[-3.9,-3.8,-3.7,-3.6,-3.5,-3.4],colors='white')
# # plt.contour([0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10],np.log10(ns_np2[:,10,:]),levels=[-3.9,-3.8,-3.7,-3.6,-3.5,-3.4],colors='white')
# # plt.savefig('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/choix_Lyot/lyot_diam_vs_obst_mB={mB}.png'.format(mB=mB_conf[5]),dpi=300)
# # plt.close()
# plt.show()

#%%

# fig,ax = plt.subplots(1)
# # fig.figsize=(15,15)
# # im = ax.imshow(np.log10(result[0,:,:]), cmap='inferno',vmin=-4,vmax=-3,origin='lower')
# im = ax.imshow(np.log10(ns_np2[valeurs_choisies[0][0],:,:]), cmap='inferno',vmin=-4,vmax=-3,origin='lower',extent=[0,20,0,20])
# plt.ylabel('Lyot Stop Inner diameter')
# plt.xlabel('FPM Diameter in $\lambda$/D')
# ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
# ax.set_xticklabels(mB_conf)
# ax.set_yticks(np.arange(len(obstruction)))
# ax.set_yticklabels(obstruction)
# # plt.xlim(diametre_lyot[0],diametre_lyot[-1])
# # plt.ylim(obstruction[0],obstruction[-1])
# plt.axhline(y=valeurs_choisies[1],linestyle='--',color='white')
# plt.axvline(x=valeurs_choisies[2]*2,linestyle='--',color='white')
# plt.colorbar(im,label="Ns/Np² in log scale")#'Contrast in log scale')
# plt.title('Lyot Stop Outer diameter : '+str(diametre_lyot[valeurs_choisies[0][0]]))
# # plt.contour([0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10],np.log10(result[0,:,:]),levels=[-3.9,-3.8,-3.7,-3.6,-3.5,-3.4],colors='white')
# # plt.contour([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[0,1,2,3,4,5,6,7,8,9,10],np.log10(ns_np2[0,:,:]),levels=[-3.9,-3.8,-3.7,-3.6,-3.5,-3.4],colors='white')
# # plt.savefig('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/choix_Lyot/lyot_diam_vs_mB_obst={obst}.png'.format(obst=obstruction[5]),dpi=300)
# # plt.close()
# plt.show()

#%%
# fig,ax = plt.subplots(1)
# # im = ax.imshow(np.log10(result[:,:,4]), cmap='inferno',vmin=-4,vmax=-3,origin='lower')
# im = ax.imshow(np.log10(ns_np2[:,:,valeurs_choisies[2][0]]), cmap='inferno',vmin=-4,vmax=-3,origin='lower')
# plt.xlabel('Lyot Stop Inner diameter')
# plt.ylabel('Lyot Stop Outer diameter')
# ax.set_xticks(np.arange(len(obstruction),step=2))
# ax.set_xticklabels(obstruction[::2])
# ax.set_yticks(np.arange(len(diametre_lyot)))
# ax.set_yticklabels(diametre_lyot)
# plt.axvline(x=valeurs_choisies[1],linestyle='--',color='white')
# plt.axhline(y=valeurs_choisies[0],linestyle='--',color='white')
# # plt.xlim(diametre_lyot[0],diametre_lyot[-1])
# # plt.ylim(obstruction[0],obstruction[-1])
# plt.colorbar(im,label="Ns/Np² in log scale")#'Contrast in log scale')
# plt.title('FPM diameter : '+str(mB_conf[valeurs_choisies[2][0]])+r' $\lambda$/D')
# # plt.contour([0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10],np.log10(result[:,:,4]),levels=[-3.9,-3.8,-3.7,-3.6,-3.5,-3.4],colors='white')
# # plt.contour([0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10],np.log10(ns_np2[:,:,4]),levels=[-3.9,-3.8,-3.7,-3.6,-3.5,-3.4],colors='white')
# # plt.savefig('/home/asimonnin/Bureau/ThesisAdrien/Andes/Data_corono/plots/choix_Lyot/lyot_obst_vs_mB_diam={diam}.png'.format(diam=diametre_lyot[5]),dpi=300)
# # plt.close()
# plt.show()


# %%
# a= np.where(result == np.min(result))

# mB_calc = np.asarray(mB_conf)

# sep_lim  = mB_calc/2/ 38.54*1.6e-6/rad2mas
# %%
