#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:43:27 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyzelda.utils import imutils, zernike
from vigan.optics import aperture

ftsz = 16 
plt.rcParams.update({'font.size': ftsz})

import os
from matplotlib import cm
from astropy.io import fits
import corono as coro

#from scipy.misc import imresize
import cv2

#%% parameters
"""
Parameters
"""
plt.close('all')
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'vlt_btw2' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxContrastL1' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
    solver       = 'gurobipy' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    Binarity    = False
    FirstDerGlobalLim = 1.
    BinarityReg       = 0.1
    LSRobustness = True
    
    #nPup = corono0.params['nPup']
    nArr = 520
    nPup = 520
    nPup0= 500
    nFPM = 50
    Fmax2d = 22.5
    nImg2d = 45
    
    wv        = 1.593e-6
    dAper     = 8
    mas2rad   = np.pi/(180.*3600)
    rMask_m   = 287e-6/2.
    Fratio    = 40
    rMask0  = rMask_m/(wv*Fratio)
    rMask_mas = 1000.*rMask0 * (wv/dAper)/mas2rad
    print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))
    
    # use real data    
    use_real_data = False
    if use_real_data:
        nArr = 384+20 #520
        nPup = 384+20 #520
        nPup0= 384 #500
        
    
    # apodizer type
    ApodBinary = False
    
    # mask radius in lam0/D unit
    rMask = rMask0*(nPup/nPup0)
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  0.0
    rho1 = 20.0
    
    # contrast in the dark region
    cDarkHole = 6.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.756
    
    # CtrBtwnPix2
    CtrBtwnPix  = True
    CtrBtwnPix2 = False
    Pupil2dSym  = False # set it True only for optimization
    ImPart      = True
    
    if use_real_data:
        CtrBtwnPix  = False
        CtrBtwnPix2 = False
    
    #nlam
    bw   = 0.2
    nlam = 5
   
    do_fits = True

nlambis = 11 

nImg2dbis = 134 #256
Fmax2dbis0 = nImg2dbis/(2*(wv/950e-9))
  
Fmax2dbis = Fmax2dbis0*(nPup/nPup0)

do_plot = False    

# separation for contrast estimates
sepbis=3.0
septer=5.0

# maximum pixel shift along a given axis for apodizer 
pix_max = 10
nLinShift = 5


#%%
"""
### File reading for Pupil and Lyot stop
"""
if True:
#    fdir = Path('../../data/2D/pupils/').resolve()
    fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').resolve()
    
    fdir_apod = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/SPHERE/upgrade/').resolve()  

    fname_apod = f'apod_initial_{nArr:04d}.fits'
    if ApodBinary is True:
        fname_apod = f'apod_binary_{nArr:04d}.fits'
        
    
    if pupil_name == 'lvr':
        fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
        fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
    elif pupil_name == 'vlt':
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
        fname_lys = 'SPHERE/sphere_stop_ST_ALC2.fits'
    elif pupil_name == 'vlt_btw2':
        fname_pup = f'pupil=vlt_btw_nPup={nPup0}_nArr={nArr}.fits'
        fname_lys = f'sphere_stop_ST_ALC2_nPup={nPup0}_nArr={nArr}.fits'
    else:
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
        fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)

    if use_real_data is True:  
        fdir_apod = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/SPHERE/').resolve()  
        fname_apod = 'SPHERE_APO1_field_transmission_map.fits'
        fname_lys = 'sphere_stop_ST_ALC2.fits'
        fpath_lys = fdir_apod / fname_lys    
        Pupil2d = aperture.vlt_pupil(nArr, nPup0, dead_actuator_diameter=0, cpix=True)
        LyotStop2dtmp = fits.getdata(fpath_lys)
        
    else:
        fpath_pup = fdir / fname_pup
        fpath_lys = fdir / fname_lys
        Pupil2d   = aperture.vlt_pupil(nArr, nPup0, dead_actuator_diameter=0, cpix=False) #fits.getdata(fpath_pup)
        LyotStop2d = fits.getdata(fpath_lys)

#%%
"""
### Read the apodizer file
"""

fpath_apod = fdir_apod / fname_apod
print(fpath_apod)


if use_real_data:
    Apod2dtmp = fits.getdata(fpath_apod,)
    if nArr == nPup0:
        Apod2d = Apod2dtmp
        LyotStop2d = LyotStop2dtmp
    else:
        Apod2d = np.zeros((nArr, nArr))
        LyotStop2d = np.zeros((nArr, nArr))
        x_ini = (nArr-nPup0)//2
        x_end = (nArr+nPup0)//2
        Apod2d[x_ini:x_end, x_ini:x_end] = Apod2dtmp
        LyotStop2d[x_ini:x_end, x_ini:x_end] = LyotStop2dtmp
else:
    Apod2d = fits.getdata(fpath_apod,)

#%%    
    
    if solver != 'gurobipy' and solver != 'stdgrb':
        solver = 'scipy'

params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg,
                 ImPart = ImPart, LSRobustness = LSRobustness)

#%%
"""
### Working directories
"""
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth').resolve() / pupil_name

fdir_pdf = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/plots/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

#%%
"""
### Filename for the images
"""
fname_direct_poly_img_f     = 'direct_poly_img_f.fits'
fname_corono_poly_img_f     = 'corono_poly_img_f.fits'
fpath_direct_poly_img_f     = fdir_pdf / fname_direct_poly_img_f
fpath_corono_poly_img_f     = fdir_pdf / fname_corono_poly_img_f

fname_direct_poly_prf_avg_f = 'direct_poly_prf_avg_f.fits'
fname_corono_poly_prf_avg_f = 'corono_poly_prf_avg_f.fits'
fname_direct_poly_prf_std_f = 'direct_poly_prf_std_f.fits'
fname_corono_poly_prf_std_f = 'corono_poly_prf_std_f.fits'
fpath_direct_poly_prf_avg_f = fdir_pdf / fname_direct_poly_prf_avg_f
fpath_corono_poly_prf_avg_f = fdir_pdf / fname_corono_poly_prf_avg_f
fpath_direct_poly_prf_std_f = fdir_pdf / fname_direct_poly_prf_std_f
fpath_corono_poly_prf_std_f = fdir_pdf / fname_corono_poly_prf_std_f

fname_direct_mono_img_t     = 'direct_mono_img_t.fits'
fname_corono_mono_img_t     = 'corono_mono_img_t.fits'
fpath_direct_mono_img_t     = fdir_pdf / fname_direct_mono_img_t
fpath_corono_mono_img_t     = fdir_pdf / fname_corono_mono_img_t

fname_direct_mono_prf_avg_t = 'direct_mono_prf_avg_t.fits'
fname_corono_mono_prf_avg_t = 'corono_mono_prf_avg_t.fits'
fname_direct_mono_prf_std_t = 'direct_mono_prf_std_t.fits'
fname_corono_mono_prf_std_t = 'corono_mono_prf_std_t.fits'
fpath_direct_mono_prf_avg_t = fdir_pdf / fname_direct_mono_prf_avg_t
fpath_corono_mono_prf_avg_t = fdir_pdf / fname_corono_mono_prf_avg_t
fpath_direct_mono_prf_std_t = fdir_pdf / fname_direct_mono_prf_std_t
fpath_corono_mono_prf_std_t = fdir_pdf / fname_corono_mono_prf_std_t


#%%  
""" 
### Coronagraph defintion
"""
if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))


#%% Display of the apodizer
"""
### Display of the apodizers
"""
plt.figure(4)
plt.clf()
plt.imshow(corono0.Pupil2d, cmap = 'inferno')
plt.title('Pupil transmission')


plt.figure(2)
plt.clf()
plt.imshow(Apod2d*corono0.Pupil2d, cmap = 'inferno')
plt.title('Apodized entrance pupil')
plt.tight_layout()
# if do_plot is True:
    # plt.savefig(str(fpath), transparent=True)


#%% Signal in intensity
"""
### Computation of the direct and coronagraphic images for the nominal case
"""
# fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis) 

if corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params2)
else:
    raise NameError(f'{corono_name}: Not an existing coronagraph!')

if corono_name == 'APLC':
    direct_poly_img_f = corono0.compute_direct_intensity_2d(Apod2d)
    direct_mono_img_t = corono0.compute_direct_intensity_2d(Apod2d, poly=False)
else:
    raise NameError(f'{corono_name}: Not an existing coronagraph!')

corono_poly_img_f = corono0.compute_corono_intensity_2d(Apod2d)
corono_mono_img_t = corono0.compute_corono_intensity_2d(Apod2d, poly=False)


#%%
"""
### computation of the azimuthally averaged radial intensity profile in broadband light
"""
direct_poly_prf_avg_f, rad_direct = imutils.profile(direct_poly_img_f, type='mean')
corono_poly_prf_avg_f, rad_corono = imutils.profile(corono_poly_img_f, type='mean')
direct_poly_prf_std_f, rad_direct = imutils.profile(direct_poly_img_f, type='std')
corono_poly_prf_std_f, rad_corono = imutils.profile(corono_poly_img_f, type='std')

#%%
"""
### computation of the azimuthally averaged radial intensity profile in monochromatic light
"""
direct_mono_prf_avg_t = np.zeros((corono0.nlam, nImg2dbis//2))
corono_mono_prf_avg_t = np.zeros((corono0.nlam, nImg2dbis//2))
direct_mono_prf_std_t = np.zeros((corono0.nlam, nImg2dbis//2))
corono_mono_prf_std_t = np.zeros((corono0.nlam, nImg2dbis//2))

for ilam in range(corono0.nlam):
    direct_mono_prf_avg_t[ilam], rad_direct = imutils.profile(direct_mono_img_t[ilam], type='mean')
    corono_mono_prf_avg_t[ilam], rad_corono = imutils.profile(corono_mono_img_t[ilam], type='mean')
    direct_mono_prf_std_t[ilam], rad_direct = imutils.profile(direct_mono_img_t[ilam], type='std')
    corono_mono_prf_std_t[ilam], rad_corono = imutils.profile(corono_mono_img_t[ilam], type='std')    

#%% Intensity profiles of the direct and coronagraphic images
"""
### Display of the coronagraphic images in broadband light
"""

#val = 0
#if CtrBtwnPix2 is True:
#    val = 1/2

#val = 0
#if nImg2d%2 == 0:
#    val = 1/2
#
#xx,yy  = np.meshgrid(np.arange(nImg2d)-nImg2d//2+val, np.arange(nImg2d)-nImg2d//2+val)
#mydist = (Fmax2d/nImg2d)*np.hypot(yy,xx)
#xi2d = mydist[nImg2d//2,nImg2d//2:]

xi2d = corono0.xi2d
if nImg2dbis%2 == 0:
    xi2d = corono0.xi2d_ctr


#%%

fname_image_plane_f_disp = 'corono_poly_img_f_nPup={0}_disp.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_pdf / fname_image_plane_f_disp

f2 = plt.figure(23, figsize=(8,4.5))
plt.clf()
exec('ax{0} = f2.add_subplot(1,{1},{0})'.format(1,1))
exec('im = ax{0}.imshow(np.log10(corono_poly_img_f/direct_poly_img_f.max()), cmap = "inferno", vmin=-7, vmax=-3)'.format(1))
#exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "nmap={1:05d}", fontsize=16, horizontalalignment="center", color = "white")'.format(1,1))
exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(1,))
exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(1,))

f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
if do_plot is True:
    plt.savefig(str(fpath_image_plane_f_disp), transparent=True)
plt.tight_layout()
plt.show()



#%% plot displays at multiple wavelengths
"""
### Display of the intensity profiles of the coronagraphic images in monochromatic light
"""

fname_image_plane_mono_plot = 'corono_poly_prf_std_t_mono_nPup={0}_plot.pdf'.format(nPup)
fpath_image_plane_mono_plot = fdir_pdf / fname_image_plane_mono_plot


values = range(nlambis)
colors_wv = plt.cm.rainbow(np.linspace(0,1,nlambis))

if corono0.nlam > 1:
    idx_lam_peak = (corono0.nlam+1)//2
else:
    idx_lam_peak = 0

# Intensity profiles of the direct and coronagraphic images
plt.figure(12, figsize=(8,4.5))
plt.clf()
for ilam in range(corono0.nlam):
    plt.semilogy(rad_corono*Fmax2dbis/nImg2dbis, corono_mono_prf_std_t[ilam]/direct_mono_img_t[idx_lam_peak].max(),
                label=r'{0:.3f}$\lambda_0$'.format(corono0.lam_t[ilam]), color = colors_wv[ilam])
    
plt.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
plt.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi2d.min(), xmax=corono0.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
plt.xlabel(r'Angular separation in $\lambda_0$/D')
plt.ylabel(r'1$\sigma$ normalized intensity in log scale')
plt.ylim(3e-8, 3e-4)
plt.legend()
plt.title('Intensity profile in monochromatic light')
plt.tight_layout()
if do_plot is True:
    plt.savefig(str(fpath_image_plane_mono_plot), transparent=True)

plt.show()


#%% Intensity profiles of the direct and coronagraphic images

fname_image_plane_plot = 'corono_poly_prf_std_t_nPup={0}_plot.pdf'.format(nPup)
fpath_image_plane_plot = fdir_pdf / fname_image_plane_plot

rad_corono = np.arange(nImg2dbis//2)
colors_cor = plt.cm.rainbow(np.linspace(0,1,1))

i0 = 0

plt.figure(11, figsize=(8,4.5))
plt.clf()
plt.semilogy(rad_corono*Fmax2dbis/nImg2dbis, corono_poly_prf_std_f/direct_poly_img_f.max(),
        label='current APLC', color = colors_cor[i0])

plt.axvline(x=rMask0, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
plt.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
plt.axhline(10**(-cDarkHole), xmin=corono0.xi2d.min(), xmax=corono0.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
plt.xlabel(f'Angular separation in $\lambda_0$/D ($\lambda_0={wv*1e6:.3f}\mu m$)')
plt.ylabel(r'1$\sigma$ normalized intensity in log scale')
plt.xlim(-1.0, 21.0)
plt.ylim(3e-8, 3e-4)
plt.legend()
plt.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
plt.tight_layout()
if do_plot is True:
    plt.savefig(str(fpath_image_plane_plot), transparent=True)


#%%
"""
### Robustness to apodizer misalignments
"""

if nLinShift >1:
    pix_t = (np.round(pix_max*(np.arange(nLinShift)-nLinShift//2)*(2/(nLinShift-1)))).astype(int) #np.arange(2*pix_max+1)-pix_max
else:
    pix_t = np.asarray([0])
npix = len(pix_t)

Apod2d_t = []
# for j in range(npix-1, -1, -1):
for j in range(npix):
    for i in range(npix):
        Apod2d_t.append(np.roll(np.roll(Apod2d, pix_t[i], axis=1), pix_t[j], axis=0))
        print('{0:.2f}, {1:.2f}'.format(pix_t[j], pix_t[i]))
        
#%%
# number of coronagraph configuration
nApod2d      = len(Apod2d_t)
print('# of coronagraph configurations: {0}'.format(nApod2d))

#%%
      
params2    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis) 

if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params2)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
val = 0
if nImg2dbis%2 == 0:
    val = 1/2


# array of angular distances in the final image plane
xx,yy  = np.meshgrid(np.arange(nImg2dbis)-nImg2dbis//2+val, np.arange(nImg2dbis)-nImg2dbis//2+val)
mydist = (Fmax2dbis/nImg2dbis)*np.hypot(yy,xx)        
resbis = (mydist <= sepbis +0.5)*(mydist >= sepbis -0.5)
rester = (mydist <= septer +0.5)*(mydist >= septer -0.5)

plt.figure(3)
plt.clf()
plt.imshow(resbis)
plt.title(f'Angular resolution at {sepbis}$\lambda$/D')

#%%
direct_poly_img_aberr_t = [] 
corono_poly_img_aberr_t = []

values = range(nlambis)
colors = plt.cm.rainbow(np.linspace(0,1,nlambis))

for k in range(nApod2d):
    if corono_name == 'APLC':
        direct_poly_img_aberr_t.append(corono0.compute_direct_intensity_2d(Apod2d_t[k]))
    else:
        direct_poly_img_aberr_t.append(corono0.compute_direct_intensity_2d(corono0.Pupil2d))
    corono_poly_img_aberr_t.append(corono0.compute_corono_intensity_2d(Apod2d_t[k]))

direct_poly_img_aberr_t = np.asarray(direct_poly_img_aberr_t)
corono_poly_img_aberr_t = np.asarray(corono_poly_img_aberr_t)

#%%
plt.figure(1)
plt.clf()
plt.imshow(rester)
plt.title(f'Angular resolution at {septer}$\lambda$/D')

#%%
corono_poly_avg_resbis_aberr_t = []
corono_poly_avg_rester_aberr_t = []

for i in range(nApod2d):
    corono_poly_avg_resbis_aberr_t.append(np.mean(corono_poly_img_aberr_t[i, resbis != 0])/direct_poly_img_f.max())
    corono_poly_avg_rester_aberr_t.append(np.mean(corono_poly_img_aberr_t[i, rester != 0])/direct_poly_img_f.max())


#%%
"""
### Create the apodizer mosaic
"""
if nApod2d <= 25:
    apo_mosaic = np.zeros((nLinShift*nPup,nLinShift*nPup))
    for  i in range(nLinShift):
        for j in range(nLinShift):
            apo_mosaic[i*nPup:(i+1)*nPup,j*nPup:(j+1)*nPup] = np.abs(Apod2d_t[i*nLinShift+j]*Pupil2d)**2

#%%
"""
### Create the coronagraphic image maosaic
"""
if nApod2d <= 25:
    img_mosaic = np.zeros((nLinShift*nImg2dbis,nLinShift*nImg2dbis))
    for  i in range(nLinShift):
        for j in range(nLinShift):
            img_mosaic[i*nImg2dbis:(i+1)*nImg2dbis,j*nImg2dbis:(j+1)*nImg2dbis] = corono_poly_img_aberr_t[i*nLinShift+j]
            
    img_mosaic /= direct_poly_img_f.max()

#%%
"""
### Create the contrast maosaic
"""
if nApod2d <= 25:
    corono_poly_avg_resbis_aberr_mosaic = np.zeros((nLinShift,nLinShift))
    corono_poly_avg_rester_aberr_mosaic = np.zeros((nLinShift,nLinShift))
    for  i in range(nLinShift):
        for j in range(nLinShift):
            corono_poly_avg_resbis_aberr_mosaic[i,j] = np.log10(corono_poly_avg_resbis_aberr_t[i*nLinShift+j])
            corono_poly_avg_rester_aberr_mosaic[i,j] = np.log10(corono_poly_avg_rester_aberr_t[i*nLinShift+j])

#corono_poly_avg_resbis_aberr_mosaic = 100*np.abs((corono_poly_avg_resbis_aberr_mosaic - corono_poly_avg_resbis_aberr_mosaic[nLinShift//2, nLinShift//2]))/np.abs(corono_poly_avg_resbis_aberr_mosaic[nLinShift//2, nLinShift//2])
#corono_poly_avg_rester_aberr_mosaic = 100*np.abs((corono_poly_avg_rester_aberr_mosaic - corono_poly_avg_rester_aberr_mosaic[nLinShift//2, nLinShift//2]))/np.abs(corono_poly_avg_rester_aberr_mosaic[nLinShift//2, nLinShift//2])

corono_poly_avg_resbis_aberr_mosaic = corono_poly_avg_resbis_aberr_mosaic - corono_poly_avg_resbis_aberr_mosaic[nLinShift//2, nLinShift//2]
corono_poly_avg_rester_aberr_mosaic = corono_poly_avg_rester_aberr_mosaic - corono_poly_avg_rester_aberr_mosaic[nLinShift//2, nLinShift//2]


 
#%%
"""
### Plot display of the impact of apodizer misalignment on contrast 
"""
colors_shifts = plt.cm.rainbow(np.linspace(0,1,2))
ls_shifts = ["-", "--"]

fname_lowfe_plot = 'corono_poly_apodizer_sensitivity_plot.pdf'
fpath_lowfe_plot = fdir_pdf / fname_lowfe_plot

plot_lines = []

idx = list((npix-1)//2+npix*np.arange(npix))

plt.figure(31, figsize=(8,4.5))
plt.clf()
l1, = plt.semilogy(100*pix_t/nPup0, corono_poly_avg_resbis_aberr_t[npix*(npix-1)//2:npix*((npix-1)//2+1)],
            color = colors_shifts[0], marker='x', ls ='-')
l2, = plt.semilogy(100*pix_t/nPup0, corono_poly_avg_rester_aberr_t[npix*(npix-1)//2:npix*((npix-1)//2+1)],
            color = colors_shifts[1], marker='x', ls ='-')
l3, = plt.semilogy(100*pix_t/nPup0, np.asarray(corono_poly_avg_resbis_aberr_t)[idx],
            color = colors_shifts[0], marker='x', ls='--')
l4, = plt.semilogy(100*pix_t/nPup0, np.asarray(corono_poly_avg_rester_aberr_t)[idx],
            color = colors_shifts[1], marker='x', ls='--')

l5, = plt.semilogy([], [], color = "k", ls='-')
l6, = plt.semilogy([], [], color = "k", ls='--')

plt.xlabel(r'Apodizer shift in pupil diameter [%]')
plt.ylabel(r'Averaged normalized intensity'.format(sepbis))
plt.axhline(10**(-cDarkHole+2), xmin=pix_t.min(), xmax=pix_t.max(), 
           linewidth=1, color='k', linestyle='--')    
plt.axhline(10**(-cDarkHole), xmin=pix_t.min(), xmax=pix_t.max(), 
           linewidth=1, color='k', linestyle='--')    
plt.xlim(-2.5, 2.5)
plt.ylim(3e-7, 3e-4)  
plt.title(r'Averaged intensity in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
plt.grid(True,which="both",ls="--")

legend1 = plt.legend([l5,l6], ["x-axis", "y-axis"], loc=3)
plt.gca().add_artist(legend1)
plt.legend([l1,l2], [r'{0:.1f} $\lambda_0/D$'.format(sepbis), r'{0:.1f} $\lambda_0/D$'.format(septer)], loc=4)

plt.tight_layout()
if do_plot is True:
    plt.savefig(str(fpath_lowfe_plot), transparent=True)

#%%
"""
### Display of the apodizer mosaic
"""
    
if nApod2d <= 25:
    
    fname_mosaic_plot = 'corono_poly_apodizer_sensitivity_mosaic_apo_plot.pdf'
    fpath_mosaic_plot = fdir_pdf / fname_mosaic_plot
    
    ticks0 = [nPup//2 + i*nPup for i in range(nLinShift)]
    xticklabels0 = [str(100*pix_t[i]/nPup0) for i in range(nLinShift)]
    
    #xticklabels0 = ['-1.0', '-0.5', '0.0', '0.5', '1.0']
    xcoords = [i*nPup for i in range(nLinShift)]
    
    
    f3 = plt.figure(33, figsize=(13,10))
    plt.clf()
    ax1 = f3.add_subplot(111)
    im = ax1.imshow(apo_mosaic, cmap ="inferno", vmin=0, vmax=1)
    plt.xlabel('Apodizer shift in pupil diameter [%](x-axis)')
    plt.ylabel('Apodizer shift in pupil diameter [%](y-axis)')
    
    ax1.set_xticks(ticks0)
    ax1.set_xticklabels(xticklabels0) #, fontdict=font_dict)

    ax1.set_yticks(ticks0)
    ax1.set_yticklabels(xticklabels0) #, fontdict=font_dict)
    
    for xc in xcoords:
        plt.axvline(x=xc, color='k')
        plt.axhline(y=xc, color='k')        

    f3.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)
    
    f3.subplots_adjust(right=0.85)
    cbar_ax = f3.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar    = f3.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Normalized intensity', rotation=270, labelpad = 20)
    plt.tight_layout()
    plt.show()     

    if do_plot is True:
        plt.savefig(str(fpath_mosaic_plot), transparent=True)

       
#%%
"""
### Display of the coronagraphic image mosaic
"""
if nApod2d <= 25:
    fname_mosaic_img_plot = 'corono_poly_apodizer_sensitivity_mosaic_img_plot.pdf'
    fpath_mosaic_img_plot = fdir_pdf / fname_mosaic_img_plot
    
    ticks0 = [nImg2dbis//2 + i*nImg2dbis for i in range(nLinShift)]
    xticklabels0 = [str(100*pix_t[i]/nPup0) for i in range(nLinShift)]
    
    #xticklabels0 = ['-1.0', '-0.5', '0.0', '0.5', '1.0']
    xcoords = [i*nImg2dbis for i in range(nLinShift)]
    
    if nApod2d <= 25:
        f3 = plt.figure(32, figsize=(13,10))
        plt.clf()
        ax1 = f3.add_subplot(111)
        im = ax1.imshow(np.log10(img_mosaic), cmap ="inferno", vmin=-7, vmax=-3)
        plt.xlabel('Apodizer shift in pupil diameter [%](x-axis)')
        plt.ylabel('Apodizer shift in pupil diameter [%](y-axis)')
        
        ax1.set_xticks(ticks0)
        ax1.set_xticklabels(xticklabels0) #, fontdict=font_dict)
    
        ax1.set_yticks(ticks0)
        ax1.set_yticklabels(xticklabels0) #, fontdict=font_dict)
        
        for xc in xcoords:
            plt.axvline(x=xc, color='k')
            plt.axhline(y=xc, color='k')        
    
        f3.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
        
        f3.subplots_adjust(right=0.85)
        cbar_ax = f3.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar    = f3.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel('Image intensity in log scale', rotation=270, labelpad = 20)
        plt.tight_layout()
        plt.show()     
    
        if do_plot is True:
            plt.savefig(str(fpath_mosaic_img_plot), transparent=True)


#%%
"""
### Display of the contrast mosaic
"""
    
if nApod2d <= 25:
    
    fname_mosaic_cst1_plot = 'corono_poly_apodizer_sensitivity_mosaic_cst1_plot.pdf'
    fpath_mosaic_cst1_plot = fdir_pdf / fname_mosaic_cst1_plot
    
    ticks0 = [i for i in range(nLinShift)]
    xticklabels0 = [str(100*pix_t[i]/nPup0) for i in range(nLinShift)]
    
    #xticklabels0 = ['-1.0', '-0.5', '0.0', '0.5', '1.0']
    xcoords = [0.5+ i*1 for i in range(nLinShift)]
    
    
    f3 = plt.figure(34, figsize=(12,5))
    plt.clf()
    ax1 = f3.add_subplot(121)
    im = ax1.imshow(corono_poly_avg_resbis_aberr_mosaic, cmap ="inferno", vmin=0, vmax=0.1)
    plt.xlabel('Apodizer shift in pupil diameter [%]')
    plt.ylabel('Apodizer shift in pupil diameter [%]')
    
    ax1.set_xticks(ticks0)
    ax1.set_xticklabels(xticklabels0) #, fontdict=font_dict)

    ax1.set_yticks(ticks0)
    ax1.set_yticklabels(xticklabels0) #, fontdict=font_dict)
    
    ax1.set_title(f'at {sepbis:.1f}$\lambda$/D')
    
    for xc in xcoords:
        ax1.axvline(x=xc, color='k')
        ax1.axhline(y=xc, color='k')        

    ax2 = f3.add_subplot(122)
    im = ax2.imshow(corono_poly_avg_rester_aberr_mosaic, cmap ="inferno", vmin=0, vmax=0.1)
    ax2.set_xlabel('Apodizer shift in pupil diameter [%]')
    #ax2.set_ylabel('Apodizer shift in pupil diameter [%]')
    
    ax2.set_xticks(ticks0)
    ax2.set_xticklabels(xticklabels0) #, fontdict=font_dict)

    ax2.set_yticks(ticks0)
    ax2.set_yticklabels(xticklabels0) #, fontdict=font_dict)
    
    ax2.set_title(f'at {septer:.1f}$\lambda$/D')
    
    for xc in xcoords:
        ax2.axvline(x=xc, color='k')
        ax2.axhline(y=xc, color='k')        


    f3.subplots_adjust(bottom=0.1, top=0.9, left=0., right=0.7,
                    wspace=0.02, hspace=0.02)
    
    f3.subplots_adjust(right=1.)
    cbar_ax = f3.add_axes([0.9, 0.2, 0.025, 0.7])
    cbar    = f3.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Contrast loss in dex', rotation=270, labelpad = 20)
    plt.tight_layout()
    plt.show()     

    if do_plot is True:
        plt.savefig(str(fpath_mosaic_cst1_plot), transparent=True)


