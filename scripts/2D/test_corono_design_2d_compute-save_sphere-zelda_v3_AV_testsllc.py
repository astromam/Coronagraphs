#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 17:18:23 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%% Initialization
import numpy as np
import os
import time
from pyzelda.utils import aperture, imutils
from pathlib import Path
from astropy.io import fits
import corono as coro

#%% APLC2d tests
"""
tests on APLC 2d class
"""
corono_name   = 'APLC' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = True
CtrBtwnPix2 = False
Pupil2dSym  = False
cDarkHole   = 6

wv        = 1.593e-6
width     = 52e-9

dAper     = 8

mas2rad   = np.pi/(180.*3600)

rMask_m   = 287e-6/2.
Fratio    = 40

rMask  = rMask_m/(wv*Fratio)
rMask_mas = 1000.*rMask * (wv/dAper)/mas2rad
print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))

rho0   = 5.
rho1   = 20.
nPup   = 384
nImg2d = 600
Fmax2d = 60
nlam   = 5
bw     = width/wv 
nFPM   = 200
   
kw_aberr     = True
kw_2nddate   = False    
kw_skyobs    = False
kw_aftercorr = False
kw_saxo      = False
saxomap_i    = 0
saxomap_f    = 9

if saxomap_i <= saxomap_f:
    nsaxomap     = saxomap_f - saxomap_i + 1
else:
    raise NameError('initial saxo map (saxomap_i={0}) must be smaller than final saxo map (saxomap_f={1})!'.format(saxomap_i, saxomap_f))


#%% 
if kw_aberr is False:
    str_aberr = 'wo_aberr'
    str_date  = ''
    if kw_skyobs is True:
        str_obs   = 'sky'
    else:
        str_obs   = 'internal'
    str_corr  = ''
    str_saxo  = ''
    str_saxoset= ''
    nmap      = 1 
else:
    str_aberr = 'with_aberr'    
    str_date  = '2018-04-01'
    str_obs   = 'internal'
    str_corr  = 'before_correction'
    imap0     = 0
    nmap      = 1 
    str_saxo_tmp  = 'wo_saxo'
    if kw_2nddate is True:
        str_date = '2018-04-03'
    if kw_skyobs is True:
        str_obs  = 'sky'
    if kw_aftercorr is True:
        str_corr = 'after_correction'
        imap0    = 3
    if kw_saxo is True and kw_2nddate is True:
        str_saxo = 'with_saxo'
        nmap     = nsaxomap*1
    else:
        str_saxo = ''

fdir = Path('../../').resolve()

fdir_pupils_sllc  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE-SLLC'
fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 

if kw_aberr is True:
    fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_date / str_obs / str_saxo / str_corr  
else:
    fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_obs / str_saxo / str_corr  

fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
fdir_saxo    = fdir / 'data' / '2D' / 'ZELDA' / '2018-04-03'

#%%
if not os.path.exists(fdir_results):
    os.makedirs(fdir_results)

fname_Apod2d     = 'SPHERE_APO1_field_transmission_map.fits'
#fname_Apod2d = 'sphere_pupil_APO1_BH.fits'
fname_Ampmap2d   = 'sphere_pupil_clear_BH_amp.fits'
fname_LyotStop2d = 'sphere_stop_ST_ALC2.fits'

if kw_aberr is True:
    if kw_skyobs is True:
        fname_ZELDAmapnm3d = '2018-04-01_night_ncpa_loop_700modes_5_ncpa_loop_opd.fits'
        if kw_2nddate is True:
            fname_ZELDAmapnm3d = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
    else:
        fname_ZELDAmapnm3d = '20150805_0004_opd_map_modified.fits'        
        if kw_2nddate is True:        
            fname_ZELDAmapnm3d = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'
    
    if kw_saxo is True and kw_2nddate is True:    
        fname_SAXOmapnm3d = '2018-04-04T03_06_15-saxo_residual_turbulence.fits'
        if kw_aftercorr is True:
            fname_SAXOmapnm3d = '2018-04-04T03_12_50-saxo_residual_turbulence.fits'

        
fpath_Apod2d     = fdir_pupils / fname_Apod2d
fpath_Ampmap2d   = fdir_pupils_sllc / fname_Ampmap2d

if kw_aberr is True:
    fpath_ZELDAmapnm3d = fdir_pupils_sllc  / fname_ZELDAmapnm3d   
    if kw_saxo is True and kw_2nddate is True:
        fpath_SAXOmapnm3d = fdir_saxo / fname_SAXOmapnm3d
    
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d

#%%

fname_direct_poly_img_t     = 'direct_poly_img_nmap={0:05d}_t_test.fits'.format(nmap)
fname_corono_poly_img_t     = 'corono_poly_img_nmap={0:05d}_t_test.fits'.format(nmap)
fpath_direct_poly_img_t     = fdir_results / fname_direct_poly_img_t
fpath_corono_poly_img_t     = fdir_results / fname_corono_poly_img_t

fname_direct_poly_img_f     = 'direct_poly_img_nmap={0:05d}_f_test.fits'.format(nmap)
fname_corono_poly_img_f     = 'corono_poly_img_nmap={0:05d}_f_test.fits'.format(nmap)
fpath_direct_poly_img_f     = fdir_results / fname_direct_poly_img_f
fpath_corono_poly_img_f     = fdir_results / fname_corono_poly_img_f

fname_direct_poly_prf_avg_f = 'direct_poly_prf_nmap={0:05d}_avg_f_test.fits'.format(nmap)
fname_corono_poly_prf_avg_f = 'corono_poly_prf_nmap={0:05d}_avg_f_test.fits'.format(nmap)
fname_direct_poly_prf_std_f = 'direct_poly_prf_nmap={0:05d}_std_f_test.fits'.format(nmap)
fname_corono_poly_prf_std_f = 'corono_poly_prf_nmap={0:05d}_std_f_test.fits'.format(nmap)
fpath_direct_poly_prf_avg_f = fdir_results / fname_direct_poly_prf_avg_f
fpath_corono_poly_prf_avg_f = fdir_results / fname_corono_poly_prf_avg_f
fpath_direct_poly_prf_std_f = fdir_results / fname_direct_poly_prf_std_f
fpath_corono_poly_prf_std_f = fdir_results / fname_corono_poly_prf_std_f


#%% Entrance pupil
if kw_skyobs is True:
    Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)
else:
    Pupil2d = aperture.disc_obstructed(nPup, nPup/2, 0.14)
    
#%% Apodization
Apod2d = fits.getdata(fpath_Apod2d)

#%% Amplitude errors
if kw_aberr is True:
    Ampmap2d = fits.getdata(fpath_Ampmap2d)

#%% Phase errors
if kw_aberr is True:
    ZELDAmapnm3d = fits.getdata(fpath_ZELDAmapnm3d)

    if kw_saxo is True and kw_2nddate is True:
        SAXOmapnm3d_tmp = fits.getdata(fpath_SAXOmapnm3d)
        nsaxo_all = len(SAXOmapnm3d_tmp)        
        pupil_tmp = aperture.sphere_saxo_pupil()
        pupil = np.round(imutils.scale(pupil_tmp, 0, new_dim=(384,384), method='interp'))

        # rescale NCPA map
        SAXOmapnm3d = []
        for i in range(nmap):
            SAXOmapnm3d.append(imutils.scale(SAXOmapnm3d_tmp[i], 0, new_dim=(384,384), method='interp'))
            print('SAXO map before scaling: {0:.2f} nm RMS, after: {1:.2f} nm RMS'.format(np.std(SAXOmapnm3d_tmp[i, pupil_tmp != 0]), np.std(np.asarray(SAXOmapnm3d)[i, pupil != 0])))
        
        SAXOmapnm3d = np.asarray(SAXOmapnm3d)


#%% Lyot Stop
LyotStop2d = fits.getdata(fpath_LyotStop2d)

#%%
if corono_name != 'APLC':
    raise NameError('Check the name of the coronagraph!')

params = coro.to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                 rMask = rMask,
                 Pupil2dSym = Pupil2dSym, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                 CtrBtwnPix=CtrBtwnPix,
                 CtrBtwnPix2 = CtrBtwnPix2, 
                 nlam=nlam, bw = bw, wv =wv,
                 rho0   = rho0, rho1 = rho1, cDarkHole = cDarkHole,
                 OPDmap2d = None, Ampmap2d = None)

#%%
# define the array of images for each map
direct_poly_img_t = np.zeros((nmap, nImg2d, nImg2d))
corono_poly_img_t = np.zeros((nmap, nImg2d, nImg2d))

direct_poly_pup_t = np.zeros((nmap, nPup, nPup))
corono_poly_pup_t = np.zeros((nmap, nPup, nPup))

# define the averaged image
direct_poly_img_f = np.zeros((nImg2d, nImg2d))
corono_poly_img_f = np.zeros((nImg2d, nImg2d))

# define the averaged and standard deviation profiles of the images
direct_poly_prf_avg_f = np.zeros((nImg2d//2))
corono_poly_prf_avg_f = np.zeros((nImg2d//2))
direct_poly_prf_std_f = np.zeros((nImg2d//2))
corono_poly_prf_std_f = np.zeros((nImg2d//2))

#%%
# definition of the coronagraph class
for imap in range(nmap):
    t0 = time.time()
    if kw_aberr is True:
        OPDmap2d = ZELDAmapnm3d*1e-9
        if kw_saxo is True and kw_2nddate is True:
            OPDmap2d += SAXOmapnm3d[saxomap_i+imap]*1e-9
        
        params   = coro.update_params(params, OPDmap2d = OPDmap2d, Ampmap2d = Ampmap2d, LyotStop2d = LyotStop2d)
        corono0  = coro.design.APLC2d(**params)
    else:
        params  = coro.update_params(params, OPDmap2d = None, Ampmap2d = None, LyotStop2d = LyotStop2d)
        corono0 = coro.design.APLC2d(**params)    
                
    direct_poly_img_t[imap] = corono0.compute_direct_intensity_2d(Apod2d)
    corono_poly_img_t[imap] = corono0.compute_corono_intensity_2d(Apod2d)
    direct_poly_pup_t[imap] = (np.abs(corono0.compute_direct_lyot_field_2d(Apod2d))**2)[2]
    corono_poly_pup_t[imap] = (np.abs(corono0.compute_corono_lyot_field_2d(Apod2d))**2)[2]
    

    t1 = time.time()
    print('map {1}/{2}, computation time: {0:.2f}s'.format(t1-t0, imap+1, nmap))

# computation of the averaged images
if nmap > 1:
    direct_poly_img_f = np.mean(direct_poly_img_t, axis=0)
    corono_poly_img_f = np.mean(corono_poly_img_t, axis=0)
else:
    direct_poly_img_f = direct_poly_img_t[0]
    corono_poly_img_f = corono_poly_img_t[0]

# image normalization
direct_peak_val = direct_poly_img_f.max()
direct_poly_img_f /= direct_peak_val
corono_poly_img_f /= direct_peak_val
     
# computation of the averaged and standard deviation profiles of the images   
direct_poly_prf_avg_f, rad_direct = imutils.profile(direct_poly_img_f, type='mean')
corono_poly_prf_avg_f, rad_corono = imutils.profile(corono_poly_img_f, type='mean')
direct_poly_prf_std_f, rad_direct = imutils.profile(direct_poly_img_f, type='std')
corono_poly_prf_std_f, rad_corono = imutils.profile(corono_poly_img_f, type='std')

#%% saving of the images
fits.writeto(fpath_direct_poly_img_t, direct_poly_img_t, overwrite=True)
fits.writeto(fpath_corono_poly_img_t, corono_poly_img_t, overwrite=True)

fits.writeto(fpath_direct_poly_img_f, direct_poly_img_f, overwrite=True)
fits.writeto(fpath_corono_poly_img_f, corono_poly_img_f, overwrite=True)

fits.writeto(fpath_direct_poly_prf_avg_f, direct_poly_prf_avg_f, overwrite=True)
fits.writeto(fpath_corono_poly_prf_avg_f, corono_poly_prf_avg_f, overwrite=True)
fits.writeto(fpath_direct_poly_prf_std_f, direct_poly_prf_std_f, overwrite=True)
fits.writeto(fpath_corono_poly_prf_std_f, corono_poly_prf_std_f, overwrite=True)


#%%
import pylab as pl

pl.figure(0)
pl.clf()
pl.imshow(direct_poly_pup_t[0]*LyotStop2d, cmap='inferno')
pl.show()

pl.figure(1)
pl.clf()
pl.imshow(corono_poly_pup_t[0]**0.25*LyotStop2d, cmap='inferno')
pl.show()

#%%
pl.figure(2)
pl.clf()
pl.imshow(direct_poly_img_t[0]**0.25, cmap='inferno')
pl.title('direct image')
pl.show()

pl.figure(3)
pl.clf()
pl.imshow(corono_poly_img_t[0]**0.25, cmap='inferno')
pl.title('coronagraphic image')
pl.show()
    