#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:20:52 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%% Initialization
"""
### Initialization
"""
import numpy as np
import os
import time
from pyzelda.utils import aperture, imutils, zernike
from pathlib import Path
from astropy.io import fits
import corono as coro

#%% APLC2d tests
"""
### Parameters
"""
# Coronagraph type
corono_name   = 'APLC' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = True
CtrBtwnPix2 = False
Pupil2dSym  = False

# Spectral bandwidth
wv        = 1.593e-6
width     = 52e-9

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
nPup   = 384   # pupil
nFPM   = 200   # focal plane mask
nImg2d = 600   # final image plane 
Fmax2d = 60    # spatial frequencies in the final image plane

# wavelength sampling
nlam   = 5
bw     = width/wv 

# simulation configuration   
kw_aberr     = True
kw_2nddate   = True    
kw_skyobs    = True
kw_aftercorr = False
kw_saxo      = True
saxomap_i    = 0    # saxo first screen
saxomap_f    = 10    # saxo last screen

# test on the order of the min and max number of saxo phase screen
if saxomap_i <= saxomap_f:
    nsaxomap     = saxomap_f - saxomap_i + 1
else:
    raise NameError('initial saxo map (saxomap_i={0}) must be smaller than final saxo map (saxomap_f={1})!'.format(saxomap_i, saxomap_f))

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
    str_saxo  = ''
    str_saxoset= ''
    nmap      = 1 
else:
    str_aberr = 'with_aberr'    
    str_date  = '2018-04-01'
    str_obs   = 'internal'
    str_corr  = 'before_correction'
    str_saxo  = ''
    imap0     = 0
    nmap      = 1
    beta_wfs  = 1./0.90
    str_saxo_tmp  = 'wo_saxo'
    if kw_2nddate is True:
        str_date = '2018-04-03'
    if kw_skyobs is True:
        str_obs  = 'sky'
    if kw_aftercorr is True:
        str_corr = 'after_correction'
        imap0    = 3
        beta_wfs = 1./0.95
    if kw_saxo is True and kw_2nddate is True:
        str_saxo = 'with_saxo'
        nmap     = nsaxomap*1
        beta_wfs = 1./0.6

#%%
fdir = Path('../../').resolve()
fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
fdir_saxo    = fdir / 'data' / '2D' / 'ZELDA' / '2018-04-03'

if kw_aberr is True:
    fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_date / str_obs / str_saxo / str_corr  
else:
    fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_obs / str_saxo / str_corr  

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
    
    if kw_saxo is True and kw_2nddate is True:    
        fname_SAXOmapnm3d = '2018-04-04T03_06_15-saxo_residual_turbulence.fits'
        if kw_aftercorr is True:
            fname_SAXOmapnm3d = '2018-04-04T03_12_50-saxo_residual_turbulence.fits'

#%% Filepaths for the file sources
fpath_Apod2d          = fdir_pupils / fname_Apod2d
fpath_Apod2d_OPDmapnm = fdir_pupils / fname_Apod2d_OPDmapnm
fpath_Ampmap2d        = fdir_pupils / fname_Ampmap2d

if kw_aberr is True:
    fpath_ZELDAmapnm3d = fdir_zelda  / fname_ZELDAmapnm3d   
    if kw_saxo is True and kw_2nddate is True:
        fpath_SAXOmapnm3d = fdir_saxo / fname_SAXOmapnm3d
    
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
Apod2d = fits.getdata(fpath_Apod2d)

#%% APodization OPD map
Apod2d_OPDmapnm = fits.getdata(fpath_Apod2d_OPDmapnm)
Apod2d_OPDmapnm[np.isnan(Apod2d_OPDmapnm)] = 0

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
        pupil = np.round(imutils.scale(pupil_tmp, 0, new_dim=(nPup,nPup), method='interp'))

        # rescale NCPA map
        SAXOmapnm3d = np.empty((nsaxo_all, nPup, nPup))
        for i in range(nmap):
            SAXOmapnm3d[i] = imutils.scale(SAXOmapnm3d_tmp[i+saxomap_i], 0, new_dim=(nPup,nPup), method='interp')
            print('{0:05}/{1:05}: SAXO map before scaling: {2:.2f} nm RMS, after: {3:.2f} nm RMS'.format(i+1, nmap, np.std(SAXOmapnm3d_tmp[i, pupil_tmp != 0]), np.std(np.asarray(SAXOmapnm3d)[i, pupil != 0])))


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
if corono_name != 'APLC':
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
            if kw_saxo is True and kw_2nddate is True:
                OPDmap2d = OPDmap2d0 + SAXOmapnm3d[saxomap_i+imap]*1e-9
                direct_poly_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d)
                corono_poly_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d)                
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
   