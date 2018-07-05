#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:55:49 2018

@author: mndiaye
"""

#%% Initialization
import numpy as np
import os

import pylab as pl
from corono import corono_design as cd
from corono.utils import to_dict, uniform_disk, radius_disk

from pyzelda.utils import aperture, imutils


from pathlib import Path

from astropy.io import fits

#%% APLC2d tests
"""
tests on APLC 2d class
"""
corono_name   = 'APLC' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = True
CtrBtwnPix2 = False
SymPupil2d  = False
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

#%%
fdir = Path('.').resolve()

fdir_pupils = fdir / 'pupils' / '2D' / 'SPHERE' 
fdir_data   = fdir / 'results' / '2D' / 'SPHERE' / 'data'

if not os.path.exists(fdir_data):
    os.makedirs(fdir_data)

fname_Apod2d     = 'SPHERE_APO1_field_transmission_map.fits'
#fname_Apod2d = 'sphere_pupil_APO1_BH.fits'
fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
fname_OPDmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'
fname_LyotStop2d = 'sphere_stop_ST_ALC2.fits'

fpath_Apod2d     = fdir_pupils / fname_Apod2d
fpath_Ampmap2d   = fdir_pupils / fname_Ampmap2d
fpath_OPDmapnm3d = fdir_pupils / fname_OPDmapnm3d
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d

fname_direct_poly_img_t     = 'direct_poly_img_t.fits'
fname_corono_poly_img_t     = 'corono_poly_img_t.fits'
fname_direct_poly_prf_avg_t = 'direct_poly_prf_avg_t.fits'
fname_corono_poly_prf_avg_t = 'corono_poly_prf_avg_t.fits'
fname_direct_poly_prf_std_t = 'direct_poly_prf_std_t.fits'
fname_corono_poly_prf_std_t = 'corono_poly_prf_std_t.fits'

fpath_direct_poly_img_t     = fdir_data / fname_direct_poly_img_t
fpath_corono_poly_img_t     = fdir_data / fname_corono_poly_img_t
fpath_direct_poly_prf_avg_t = fdir_data / fname_direct_poly_prf_avg_t
fpath_corono_poly_prf_avg_t = fdir_data / fname_corono_poly_prf_avg_t
fpath_direct_poly_prf_std_t = fdir_data / fname_direct_poly_prf_std_t
fpath_corono_poly_prf_std_t = fdir_data / fname_corono_poly_prf_std_t

fname_direct_mono_img_t     = 'direct_mono_img_t.fits'
fname_corono_mono_img_t     = 'corono_mono_img_t.fits'
fname_direct_mono_prf_avg_t = 'direct_mono_prf_avg_t.fits'
fname_corono_mono_prf_avg_t = 'corono_mono_prf_avg_t.fits'
fname_direct_mono_prf_std_t = 'direct_mono_prf_std_t.fits'
fname_corono_mono_prf_std_t = 'corono_mono_prf_std_t.fits'

fpath_direct_mono_img_t     = fdir_data / fname_direct_mono_img_t
fpath_corono_mono_img_t     = fdir_data / fname_corono_mono_img_t
fpath_direct_mono_prf_avg_t = fdir_data / fname_direct_mono_prf_avg_t
fpath_corono_mono_prf_avg_t = fdir_data / fname_corono_mono_prf_avg_t
fpath_direct_mono_prf_std_t = fdir_data / fname_direct_mono_prf_std_t
fpath_corono_mono_prf_std_t = fdir_data / fname_corono_mono_prf_std_t

fname_direct_mono_lyot_re_t = 'direct_mono_lyot_t_re.fits'
fname_direct_mono_lyot_im_t = 'direct_mono_lyot_t_im.fits'
fname_corono_mono_lyot_re_t = 'corono_mono_lyot_t_re.fits'
fname_corono_mono_lyot_im_t = 'corono_mono_lyot_t_im.fits'

fpath_direct_mono_lyot_re_t = fdir_data / fname_direct_mono_lyot_re_t
fpath_direct_mono_lyot_im_t = fdir_data / fname_direct_mono_lyot_im_t
fpath_corono_mono_lyot_re_t = fdir_data / fname_corono_mono_lyot_re_t
fpath_corono_mono_lyot_im_t = fdir_data / fname_corono_mono_lyot_im_t

#%% Entrance pupil
#Pupil2d = fits.getdata(fpath_Pupil2d)
#Pupil2d = uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)
#Pupil2d = aperture.vlt_pupil(nPup, nPup, spiders_thickness=4*0.008)
#Pupil2d = aperture.sphere_irdis_pupil()

#%% Apodization
Apod2d = fits.getdata(fpath_Apod2d)

#%% Phase errors
OPDmapnm3d = fits.getdata(fpath_OPDmapnm3d)
nmap = len(OPDmapnm3d)

#%% Amplitude errors
Ampmap2d = fits.getdata(fpath_Ampmap2d)

#%% Lyot Stop
LyotStop2d = fits.getdata(fpath_LyotStop2d)

#%%
if corono_name != 'APLC':
    raise NameError('Check the name of the coronagraph!')

label_lst = ['all errors', 'phase errors only', 'amplitude errors only', 'no errors']
ncase = len(label_lst)

direct_poly_img_t = np.zeros((ncase, nmap, nImg2d, nImg2d))
corono_poly_img_t = np.zeros((ncase, nmap, nImg2d, nImg2d))
direct_mono_img_t = np.zeros((ncase, nmap, nlam, nImg2d, nImg2d))
corono_mono_img_t = np.zeros((ncase, nmap, nlam, nImg2d, nImg2d))

direct_poly_prf_avg_t = np.zeros((ncase, nmap, nImg2d//2))
corono_poly_prf_avg_t = np.zeros((ncase, nmap, nImg2d//2))
direct_mono_prf_avg_t = np.zeros((ncase, nmap, nlam, nImg2d//2))
corono_mono_prf_avg_t = np.zeros((ncase, nmap, nlam, nImg2d//2))

direct_poly_prf_std_t = np.zeros((ncase, nmap, nImg2d//2))
corono_poly_prf_std_t = np.zeros((ncase, nmap, nImg2d//2))
direct_mono_prf_std_t = np.zeros((ncase, nmap, nlam, nImg2d//2))
corono_mono_prf_std_t = np.zeros((ncase, nmap, nlam, nImg2d//2))

direct_mono_lyot_t = np.zeros((ncase, nmap, nlam, nPup, nPup), dtype='complex128')
corono_mono_lyot_t = np.zeros((ncase, nmap, nlam, nPup, nPup), dtype='complex128')

#%%      

for imap in range(nmap):
    print('computation for map {0}'.format(imap))
    OPDmapnm2d = OPDmapnm3d[imap]
    OPDmap2d   = OPDmapnm2d*1e-9    

    params = to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                     rMask = rMask,
                     SymPupil2d = SymPupil2d, 
                     Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                     CtrBtwnPix=CtrBtwnPix,
                     CtrBtwnPix2 = CtrBtwnPix2, 
                     nlam=nlam, bw = bw, wv =wv,
                     rho0   = rho0, rho1 = rho1, cDarkHole = cDarkHole,
                     OPDmap2d = None, Ampmap2d = None)
    corono00 = cd.APLC2d(**params)
     
    params = to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                     rMask = rMask,
                     SymPupil2d = SymPupil2d, 
                     Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                     CtrBtwnPix=CtrBtwnPix,
                     CtrBtwnPix2 = CtrBtwnPix2, 
                     nlam=nlam, bw = bw, wv =wv,
                     rho0   = rho0, rho1 = rho1, cDarkHole = cDarkHole,
                     OPDmap2d = OPDmap2d, Ampmap2d = None)
    corono01 = cd.APLC2d(**params)
    
    params = to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                     rMask = rMask,
                     SymPupil2d = SymPupil2d, 
                     Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                     CtrBtwnPix=CtrBtwnPix,
                     CtrBtwnPix2 = CtrBtwnPix2, 
                     nlam=nlam, bw = bw, wv =wv,
                     rho0   = rho0, rho1 = rho1, cDarkHole = cDarkHole,
                     OPDmap2d = None, Ampmap2d = Ampmap2d)
    corono10 = cd.APLC2d(**params)
    
    params = to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                     rMask = rMask,
                     SymPupil2d = SymPupil2d, 
                     Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                     CtrBtwnPix=CtrBtwnPix,
                     CtrBtwnPix2 = CtrBtwnPix2, 
                     nlam=nlam, bw = bw, wv =wv,
                     rho0   = rho0, rho1 = rho1, cDarkHole = cDarkHole,
                     OPDmap2d = OPDmap2d, Ampmap2d = Ampmap2d)
    corono11 = cd.APLC2d(**params)

    coro_lst  = [corono11, corono01, corono10, corono00]
    
    for icase in range(ncase):
        direct_poly_img_t[icase, imap] = coro_lst[icase].compute_direct_intensity_2d(Apod2d)
        corono_poly_img_t[icase, imap] = coro_lst[icase].compute_corono_intensity_2d(Apod2d)
        direct_mono_img_t[icase, imap] = coro_lst[icase].compute_direct_intensity_2d(Apod2d, poly=False)
        corono_mono_img_t[icase, imap] = coro_lst[icase].compute_corono_intensity_2d(Apod2d, poly=False)
    
    for icase in range(ncase):
        direct_poly_prf_avg_t[icase, imap], rad_direct = imutils.profile(direct_poly_img_t[icase, imap], type='mean')
        corono_poly_prf_avg_t[icase, imap], rad_corono = imutils.profile(corono_poly_img_t[icase, imap], type='mean')
        direct_poly_prf_std_t[icase, imap], rad_direct = imutils.profile(direct_poly_img_t[icase, imap], type='std')
        corono_poly_prf_std_t[icase, imap], rad_corono = imutils.profile(corono_poly_img_t[icase, imap], type='std')
        for i in range(corono00.nlam):
            direct_mono_prf_avg_t[icase, imap, i], rad_direct = imutils.profile(direct_mono_img_t[icase, imap, i], type='mean')
            corono_mono_prf_avg_t[icase, imap, i], rad_corono = imutils.profile(corono_mono_img_t[icase, imap, i], type='mean')
            direct_mono_prf_std_t[icase, imap, i], rad_direct = imutils.profile(direct_mono_img_t[icase, imap, i], type='std')
            corono_mono_prf_std_t[icase, imap, i], rad_corono = imutils.profile(corono_mono_img_t[icase, imap, i], type='std')

    for icase in range(ncase):
        direct_mono_lyot_t[icase, imap] = coro_lst[icase].compute_direct_lyot_field_2d(Apod2d)
        corono_mono_lyot_t[icase, imap] = coro_lst[icase].compute_corono_lyot_field_2d(Apod2d)

#%%
fits.writeto(fpath_direct_poly_img_t, direct_poly_img_t, overwrite=True)
fits.writeto(fpath_corono_poly_img_t, corono_poly_img_t, overwrite=True)
fits.writeto(fpath_direct_poly_prf_avg_t, direct_poly_prf_avg_t, overwrite=True)
fits.writeto(fpath_corono_poly_prf_avg_t, corono_poly_prf_avg_t, overwrite=True)
fits.writeto(fpath_direct_poly_prf_std_t, direct_poly_prf_std_t, overwrite=True)
fits.writeto(fpath_corono_poly_prf_std_t, corono_poly_prf_std_t, overwrite=True)

fits.writeto(fpath_direct_mono_img_t, direct_mono_img_t, overwrite=True)
fits.writeto(fpath_corono_mono_img_t, corono_mono_img_t, overwrite=True)
fits.writeto(fpath_direct_mono_prf_avg_t, direct_mono_prf_avg_t, overwrite=True)
fits.writeto(fpath_corono_mono_prf_avg_t, corono_mono_prf_avg_t, overwrite=True)
fits.writeto(fpath_direct_mono_prf_std_t, direct_mono_prf_std_t, overwrite=True)
fits.writeto(fpath_corono_mono_prf_std_t, corono_mono_prf_std_t, overwrite=True)
 
#%%    
fits.writeto(fpath_direct_mono_lyot_re_t, direct_mono_lyot_t.real, overwrite=True)
fits.writeto(fpath_direct_mono_lyot_im_t, direct_mono_lyot_t.imag, overwrite=True)
fits.writeto(fpath_corono_mono_lyot_re_t, corono_mono_lyot_t.real, overwrite=True)
fits.writeto(fpath_corono_mono_lyot_im_t, corono_mono_lyot_t.imag, overwrite=True)    
    
#%%
print('ok')
    