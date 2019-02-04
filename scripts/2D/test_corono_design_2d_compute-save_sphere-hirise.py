#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:20:52 2018

Author: 
Mamadou N'Diaye <mamadou.ndiaye@oca.eu>
Arthur Vigan <arthur.vigan@lam.fr>

License: MIT license

"""

#%% Initialization
"""
### Initialization
"""
import numpy as np
import os
import time
#from pyzelda.utils import aperture, imutils, zernike
from vigan.utils import imutils
from vigan.optics import aperture, zernike
from pathlib import Path
from astropy.io import fits
import corono as coro

#%% APLC2d tests
"""
### Parameters
"""
# Coronagraph type
corono_name   = 'DummyLC' # APLC or DummyLC
CtrBtwnPix  = True
CtrBtwnPix2 = False
Pupil2dSym  = False

# Spectral bandwidth
wv        = 1.593e-6
width     = 1e-9

# Telescope characteristics
dAper     = 8
Fratio    = 40

# Focal plane mask 
mas2rad   = np.pi/(180.*3600) # Conversion factor from mas to rads
rMask_m   = 287e-6/2.         # mask size in m
rMask  = rMask_m/(wv*Fratio)  # mask size in lam0/D
rMask_mas = 1000.*rMask * (wv/dAper)/mas2rad
print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))

# sampling
pixel  = 12.25 # IRDIS pixel sampling [mas/pix]
nPup   = 384   # pupil
nFPM   = 200   # focal plane mask
nImg2d = 400   # final image plane 

# compute spatial frequencies in the final image plane
loD    = wv/dAper*180/np.pi*3600*1000/pixel
Fmax2d = nImg2d/loD    # spatial frequencies in the final image plane

# wavelength sampling
nlam   = 5
bw     = width/wv 

# simulation configuration   
kw_aberr     = True       # phase and amplitude aberrations
kw_2nddate   = False      # use NCPA measurements of the 1st or second date
kw_skyobs    = True       # telescope pupil or internal pupil
kw_aftercorr = False      # before or after NCPA correction
kw_caos      = True       # CAOS screens
caosmap_i    = 0          # caos first screen
caosmap_f    = 99         # caos last screen

# test on the order of the min and max number of caos phase screen
if caosmap_i <= caosmap_f:
    ncaosmap = caosmap_f - caosmap_i + 1
else:
    raise NameError('initial caos map (caosmap_i={0}) must be smaller than final caos map (caosmap_f={1})!'.format(caosmap_i, caosmap_f))

ndefo = 21
defo_ampl_arr = [0.]#-100 + 10.*np.arange(ndefo)
tipp_ampl = 0           
tilt_ampl = 0 

    
#%%
"""
### Directories
"""    
if kw_aberr is False:
    str_aberr = 'wo_aberr'
    str_date  = ''
    if kw_skyobs is True:
        str_obs   = 'sky'
    else:
        str_obs   = 'internal'
    str_corr  = ''
    str_caos  = ''
    str_caosset= ''
    nmap      = 1 
else:
    str_aberr = 'with_aberr'    
    str_date  = '2018-04-01'
    str_obs   = 'internal'
    str_corr  = 'before_correction'
    str_caos  = ''
    imap0     = 0
    nmap      = 1
    beta_wfs  = 1/0.90
    str_caos_tmp  = 'wo_caos'
    if kw_2nddate is True:
        str_date = '2018-04-03'
    if kw_skyobs is True:
        str_obs  = 'sky'
        beta_wfs = 1/0.6
    if kw_aftercorr is True:
        str_corr = 'after_correction'
        imap0    = 3
        beta_wfs = 1./0.95
    if kw_caos is True:
        str_caos = 'with_caos'
        nmap     = ncaosmap*1

#%%
fdir = Path('./').resolve()
fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
fdir_caos    = fdir / 'data' / '2D' / 'ZELDA'

if kw_aberr is True:
    fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_date / str_obs / str_caos / str_corr  
else:
    fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_obs / str_caos / str_corr  

if not os.path.exists(fdir_results):
    os.makedirs(fdir_results)

#%%
"""
### Filenames for the sources
"""
fname_Apod2d     = 'SPHERE_APO1_field_transmission_map.fits'
fname_Apod2d_OPDmapnm = 'apo_substrate_D1.fits'
fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
fname_LyotStop2d = 'sphere_stop_ST_ALC2.fits'

if kw_aberr is True:
    if kw_skyobs is True:
        fname_ZELDAmapnm3d = '2018-04-01_night_ncpa_loop_700modes_5_ncpa_loop_opd.fits'
        if kw_2nddate is True:
            fname_ZELDAmapnm3d = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
    else:
        fname_ZELDAmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'        
        if kw_2nddate is True:        
            fname_ZELDAmapnm3d = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'
    
    if kw_caos is True:
        fname_CAOSmapnm3d = 'SPHERE_SAXO_CAOS_phase_screens_seeing=0.9as.fits'

#%% Filepaths for the file sources
fpath_Apod2d          = fdir_pupils / fname_Apod2d
fpath_Apod2d_OPDmapnm = fdir_pupils / fname_Apod2d_OPDmapnm
fpath_Ampmap2d        = fdir_pupils / fname_Ampmap2d

if kw_aberr is True:
    fpath_ZELDAmapnm3d = fdir_zelda  / fname_ZELDAmapnm3d   
    if kw_caos is True:
        fpath_CAOSmapnm3d = fdir_caos / fname_CAOSmapnm3d
    
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d
    
    
#%% 
"""
### File reading
"""
# Pupil
if kw_skyobs is True:
    Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)
else:
    Pupil2d = aperture.disc(nPup, nPup/2)
    
#%% Apodization
if corono_name == 'APLC':
    Apod2d = fits.getdata(fpath_Apod2d)
elif corono_name == 'DummyLC':
    Apod2d = np.ones_like(Pupil2d)
else:
    raise NameError('Unknown coronagraph {}'.format(corono_name))

#%% APodization OPD map
Apod2d_OPDmapnm = fits.getdata(fpath_Apod2d_OPDmapnm)
Apod2d_OPDmapnm[np.isnan(Apod2d_OPDmapnm)] = 0

#%% Amplitude errors
if kw_aberr is True:
    Ampmap2d = fits.getdata(fpath_Ampmap2d)

#%% Phase errors
if kw_aberr is True:
    ZELDAmapnm3d = fits.getdata(fpath_ZELDAmapnm3d)

    if kw_caos is True:
        CAOSmapnm3d = fits.getdata(fpath_CAOSmapnm3d)

#%% Lyot Stop
LyotStop2d = fits.getdata(fpath_LyotStop2d)

#%%
Defo_mapnm2d = zernike.zernike1(4, npix=nPup, outside=0.)
Tipp_mapnm2d = zernike.zernike1(2, npix=nPup, outside=0.)
Tilt_mapnm2d = zernike.zernike1(3, npix=nPup, outside=0.)

#%%
"""
### Image generation
"""
#%% array initialization
# define the averaged image
direct_poly_img_f = np.zeros((nImg2d, nImg2d))
corono_poly_img_f = np.zeros((nImg2d, nImg2d))

# define the averaged and standard deviation profiles of the images
direct_poly_prf_avg_f = np.zeros((nImg2d//2))
corono_poly_prf_avg_f = np.zeros((nImg2d//2))
direct_poly_prf_std_f = np.zeros((nImg2d//2))
corono_poly_prf_std_f = np.zeros((nImg2d//2))

#%% definition of the coronagraph class parameters
if corono_name != 'APLC' and corono_name != 'DummyLC':
    raise NameError('Check the name of the coronagraph!')

params = coro.to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                 rMask = rMask,
                 Pupil2dSym = Pupil2dSym, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                 CtrBtwnPix=CtrBtwnPix,
                 CtrBtwnPix2 = CtrBtwnPix2, 
                 nlam=nlam, bw = bw, wv =wv,
                 OPDmap2d = None, Ampmap2d = None,
                 OPDmap2d_post = None)

#%%
if kw_aberr is True:
    params   = coro.update_params(params, OPDmap2d = None,
                                  Ampmap2d = Ampmap2d, LyotStop2d = LyotStop2d)
    corono0  = coro.design.APLC2d(**params)
else:
    params  = coro.update_params(params, OPDmap2d = None, 
                                 Ampmap2d = None, LyotStop2d = LyotStop2d)
    corono0 = coro.design.APLC2d(**params)    

"""
### Filepaths for the file results
"""        
for i, defo_ampl in enumerate(defo_ampl_arr):
    print('defo={0}nm rms'.format(defo_ampl))    
    fname_direct_poly_img_f     = 'direct_poly_img_nmap={0:05d}_defo={1:.1f}_tip={2:.1f}_tilt={3:.1f}_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
    fname_corono_poly_img_f     = 'corono_poly_img_nmap={0:05d}_defo={1:.1f}_tip={2:.1f}_tilt={3:.1f}_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
    fpath_direct_poly_img_f     = fdir_results / fname_direct_poly_img_f
    fpath_corono_poly_img_f     = fdir_results / fname_corono_poly_img_f
    
    #%%
    fname_direct_poly_prf_avg_f = 'direct_poly_prf_nmap={0:05d}_defo={1:.1f}_tip={2:.1f}_tilt={3:.1f}_avg_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
    fname_corono_poly_prf_avg_f = 'corono_poly_prf_nmap={0:05d}_defo={1:.1f}_tip={2:.1f}_tilt={3:.1f}_avg_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
    fname_direct_poly_prf_std_f = 'direct_poly_prf_nmap={0:05d}_defo={1:.1f}_tip={2:.1f}_tilt={3:.1f}_std_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
    fname_corono_poly_prf_std_f = 'corono_poly_prf_nmap={0:05d}_defo={1:.1f}_tip={2:.1f}_tilt={3:.1f}_std_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
    fpath_direct_poly_prf_avg_f = fdir_results / fname_direct_poly_prf_avg_f
    fpath_corono_poly_prf_avg_f = fdir_results / fname_corono_poly_prf_avg_f
    fpath_direct_poly_prf_std_f = fdir_results / fname_direct_poly_prf_std_f
    fpath_corono_poly_prf_std_f = fdir_results / fname_corono_poly_prf_std_f
    

    #%%
    # definition of the coronagraph class
    if kw_aberr is True:
        OPDmap2d0 = (beta_wfs*ZELDAmapnm3d[imap0]+Apod2d_OPDmapnm\
                    +defo_ampl*Defo_mapnm2d\
                    +tipp_ampl*Tipp_mapnm2d\
                    +tilt_ampl*Tilt_mapnm2d)*1e-9

    for imap in range(nmap):
        t0 = time.time()
        if kw_aberr is True:
            if kw_caos is True:
                OPDmap2d = OPDmap2d0 + CAOSmapnm3d[caosmap_i+imap]
                direct_poly_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d)
                corono_poly_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d)
            else:
                direct_poly_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d0)
                corono_poly_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d0)                
        else:                    
            direct_poly_img_f += corono0.compute_direct_intensity_2d(Apod2d)
            corono_poly_img_f += corono0.compute_corono_intensity_2d(Apod2d)    
    
        t1 = time.time()
        if (imap+1) % 10 == 0: 
            print('map {1}/{2}, computation time: {0:.2f}s'.format(t1-t0, imap+1, nmap))
    
    # computation of the averaged images
    direct_poly_img_f /= nmap
    corono_poly_img_f /= nmap
    
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
    """
    ### File saving
    """
    fits.writeto(fpath_direct_poly_img_f, direct_poly_img_f, overwrite=True)
    fits.writeto(fpath_corono_poly_img_f, corono_poly_img_f, overwrite=True)
    
    fits.writeto(fpath_direct_poly_prf_avg_f, direct_poly_prf_avg_f, overwrite=True)
    fits.writeto(fpath_corono_poly_prf_avg_f, corono_poly_prf_avg_f, overwrite=True)
    fits.writeto(fpath_direct_poly_prf_std_f, direct_poly_prf_std_f, overwrite=True)
    fits.writeto(fpath_corono_poly_prf_std_f, corono_poly_prf_std_f, overwrite=True)
   
