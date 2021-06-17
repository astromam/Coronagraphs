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

import matplotlib.pyplot as plt

#%% APLC2d tests
"""
### Parameters
"""
# Coronagraph type
corono_name   = 'APLC' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = True
CtrBtwnPix2 = False
Pupil2dSym  = False

# Telescope characteristics
dAper     = 8
Fratio    = 40

# Focal plane mask 
mas2rad   = np.pi/(180.*3600*1000) # Conversion factor from mas to rads
rad2mas   = 1/mas2rad
rMask_m   = 287e-6/2.         # mask size in m

# spatial sampling
nPup   = 384   # pupil
nFPM   = 200   # focal plane mask
nImg2d = 200   # final image plane 

# simulation configuration   
kw_aberr     = True
kw_2nddate   = True    
kw_skyobs    = True
kw_aftercorr = False
kw_saxo      = True
saxofudge    = 1. #80/120.
saxomap_i    = 0    # saxo first screen
saxomap_f    = 100 # saxo last screen

# seeing for on-sky observations
seeing = 0.7

# test on the order of the min and max number of saxo phase screen
if saxomap_i <= saxomap_f:
    nsaxomap     = saxomap_f - saxomap_i + 1
else:
    raise NameError('initial saxo map (saxomap_i={0}) must be smaller than final saxo map (saxomap_f={1})!'.format(saxomap_i, saxomap_f))

ndefo = 21
defo_ampl = 0.#-100 + 10.*np.arange(ndefo)
tipp_ampl = 0
tilt_ampl = 0 

# save multi-spectral images
do_sav = True 

# case with planet for plots
kwd_pla = True

# Planet position properties
sep_mas_p   = 12.25*16  #5*pscale      # planet separation in mas
theta_deg_p = 0  # planet position angle in degrees


    
#%%
"""
### Spectral parameters
"""
band = 'H2'
if band == 'H2':
    nlam = 11
    wv0   = 1.593e-6
    width = 52e-9
elif band == 'BB_H':
    nlam  = 11
    wv0   = 1625e-9 #1.593e-6
    width = 290e-9  #52e-9        
else:
    raise ValueError(f'Unknown {band} band')

# wavelength sampling
bw     = width/wv0 
lam0   = 1. 
dlam   = bw*lam0
lam_t  = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam)
wv_t   = wv0*lam_t
dwv_t  = np.zeros((1))
wv_R   = 0
if nlam > 1:
    dwv_t  = np.asarray([wv_t[1]-wv_t[0]]*nlam)
    # spectral resolution
    wv_R      = wv0/dwv_t[0]

# compute spatial frequencies in the final image plane
pixel  = 12.25 # IRDIS pixel sampling [mas/pix]
loD    = wv0/dAper*180/np.pi*3600*1000/pixel
nFre2d = nImg2d/loD    # spatial frequencies in the final image plane

# Focal plane mask 
rMask     = rMask_m/(wv0*Fratio)  # mask size in lam0/D
rMask_mas = rMask * (wv0/dAper)/mas2rad
print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv0*1e6))

# planet position properties
sep_loD_p   = sep_mas_p/(wv_t[nlam//2]/dAper*rad2mas) # planet separation in lam0/D
theta_rad_p = theta_deg_p*(np.pi/180)          # planet position angle in radians
print('planet sep: {:6.1f}mas, {:6.2f}lam0/D'.format(sep_mas_p,sep_loD_p))
print('planet ang: {:6.1f}deg, {:6.2f}rad'.format(theta_deg_p,theta_rad_p))
pla_dRA   = sep_mas_p*np.cos(theta_rad_p)      # delta in RA, in mas
pla_dDEC  = sep_mas_p*np.sin(theta_rad_p)      # delta in DEC, in mas


#%%
"""
### Directories
"""    
if kw_aberr is False:
    str_aberr = 'wo_aberr'
    str_date  = ''
    str_obs   = 'internal'
    if kw_skyobs:
        str_obs   = 'sky'
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
    if kw_2nddate:
        str_date = '2018-04-03'
    if kw_skyobs:
        str_obs  = 'sky'
    if kw_aftercorr:
        str_corr = 'after_correction'
        imap0    = 3
        beta_wfs = 1./0.95
    if kw_saxo and kw_2nddate:
        str_saxo = 'with_saxo'
        nmap     = nsaxomap*1
        beta_wfs = 1./0.6

#%%
#fdir = Path('../../').resolve()
fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs').resolve()
fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
fdir_saxo    = fdir / 'data' / '2D' / 'ZELDA' / '2018-04-03'
fdir_spectra = fdir / 'data' / '2D' / 'package_simu_spectra'

if kw_aberr:
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
        fname_Ampmap2d   = '2018-04-01_night_sphere_pupil_clear_sky_FeII_field.fits'
        fname_ZELDAmapnm3d = '2018-04-01_night_ncpa_loop_700modes_5_ncpa_loop_opd.fits'
        if kw_2nddate is True:
            fname_Ampmap2d   = '2018-04-03_night_sphere_pupil_clear_sky_FeII_field.fits'
            fname_ZELDAmapnm3d = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
    else:
        fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
        fname_ZELDAmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'        
        if kw_2nddate is True:
            fname_ZELDAmapnm3d = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'
    
    if kw_saxo and kw_2nddate:    
        fname_SAXOmapnm3d = '2018-04-04T00-41-34-saxo_residual_turbulence_time=30.0sec_seeing={:.1f}as_tiptilt=1_gains=0_fitting=1_alias=1.fits'.format(seeing)

#%%
"""
### Filepaths for the file sources
"""
fpath_Apod2d          = fdir_pupils / fname_Apod2d
fpath_Apod2d_OPDmapnm = fdir_pupils / fname_Apod2d_OPDmapnm
fpath_Ampmap2d        = fdir_zelda / fname_Ampmap2d

if kw_aberr:
    fpath_ZELDAmapnm3d = fdir_zelda  / fname_ZELDAmapnm3d   
    if kw_saxo and kw_2nddate:
        fpath_SAXOmapnm3d = fdir_saxo / fname_SAXOmapnm3d
    
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d    
  
#%%
"""
### Filename for the spectra
"""
# stellar parameters
star_SpT    = 'A0'
star_mass   = 2.2
star_age    = 20
star_dist   = 50 
#star_magH = 4.0
star_band   = 'H'
star_wv_res = 1000

plnt_mass   = 5
plnt_age    = 20
plnt_dist   = 50
plnt_wv_res = 1000


if star_band == 'J':
    star_wv_min = 0.9
    star_wv_max = 1.4
    plnt_wv_min = 0.9
    plnt_wv_max = 1.4
elif star_band == 'H':
    star_wv_min = 1.4
    star_wv_max = 1.8
    plnt_wv_min = 1.4
    plnt_wv_max = 1.8
else:
    raise ValueError()
    
fname_spectra_star = 'sphplus_medres_{}_{:.1f}MSun_{}Myr_{}pc_{:.1f}_{:.1f}_mic_R{}.fits'.format(star_SpT, star_mass, int(round(star_age)), int(round(star_dist)), star_wv_min, star_wv_max, int(round(star_wv_res)))
fname_spectra_plnt = 'sphplus_medres_{}MJup_{}Myr_{}pc_{:.1f}_{:.1f}_mic_R{}.fits'.format(int(round(plnt_mass)), int(round(plnt_age)), int(round(plnt_dist)), plnt_wv_min, plnt_wv_max,int(round(plnt_wv_res)))


#%%
"""
### Filepath for the spectra
"""
fpath_spectra_star = fdir_spectra / fname_spectra_star
fpath_spectra_plnt = fdir_spectra / fname_spectra_plnt
  
#%% 
"""
### File reading
"""
# Pupil
if kw_skyobs:
    Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)
else:
    Pupil2d = aperture.disc(nPup, nPup/2)
    
#%% Apodization
Apod2d = fits.getdata(fpath_Apod2d)

#%% APodization OPD map
Apod2d_OPDmapnm = fits.getdata(fpath_Apod2d_OPDmapnm)
Apod2d_OPDmapnm[np.isnan(Apod2d_OPDmapnm)] = 0

#%% Amplitude errors
Ampmap2d = None
if kw_aberr:
    Ampmap2d = fits.getdata(fpath_Ampmap2d)

#%% Phase errors
if kw_aberr:
    ZELDAmapnm3d = fits.getdata(fpath_ZELDAmapnm3d)

    if kw_saxo and kw_2nddate:
        SAXOmapnm3d_tmp = fits.getdata(fpath_SAXOmapnm3d)
        if saxofudge != 1.:
            SAXOmapnm3d_tmp *= saxofudge 
        nsaxo_all = len(SAXOmapnm3d_tmp)        
        pupil_tmp = aperture.sphere_saxo_pupil()
        pupil = np.round(imutils.scale(pupil_tmp, 0, new_dim=(nPup,nPup), method='interp'))

        # rescale NCPA map
        SAXOmapnm3d = np.empty((nmap, nPup, nPup))
        for i in range(nmap):
            SAXOmapnm3d[i] = imutils.scale(SAXOmapnm3d_tmp[i+saxomap_i], 0, new_dim=(nPup,nPup), method='interp')
            print('{0:05}/{1:05}: SAXO map before scaling: {2:6.2f} nm RMS, after: {3:6.2f} nm RMS'.format(i+1, nmap, np.std(SAXOmapnm3d_tmp[i, pupil_tmp != 0]), np.std(np.asarray(SAXOmapnm3d)[i, pupil != 0])))

        del SAXOmapnm3d_tmp
        

#%% Lyot Stop
LyotStop2d = fits.getdata(fpath_LyotStop2d)

#%%
Defo_mapnm2d = zernike.zernike1(4, npix=nPup, outside=0.)
Tipp_mapnm2d = zernike.zernike1(2, npix=nPup, outside=0.)
Tilt_mapnm2d = zernike.zernike1(3, npix=nPup, outside=0.)

#%%
"""
### Array initialization
"""
# define the averaged image
direct_mono_img_f = np.zeros((nlam, nImg2d, nImg2d))
corono_mono_img_f = np.zeros((nlam, nImg2d, nImg2d))
direct_peak_val   = np.zeros((nlam))

# define the averaged and standard deviation profiles of the images
direct_mono_prf_avg_f = np.zeros((nlam, nImg2d//2))
corono_mono_prf_avg_f = np.zeros((nlam, nImg2d//2))
direct_mono_prf_std_f = np.zeros((nlam, nImg2d//2))
corono_mono_prf_std_f = np.zeros((nlam, nImg2d//2))

# define the averaged image
direct_mono_img_fp = np.zeros((nlam, nImg2d, nImg2d))
corono_mono_img_fp = np.zeros((nlam, nImg2d, nImg2d))

#%% 
"""
### Definition of the coronagraph class parameters
"""
if corono_name != 'APLC':
    raise NameError('Check the name of the coronagraph!')

params = coro.to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = nFre2d, nFPM = nFPM,
                 rMask = rMask,
                 Pupil2dSym = Pupil2dSym, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                 CtrBtwnPix=CtrBtwnPix,
                 CtrBtwnPix2 = CtrBtwnPix2, 
                 nlam=nlam, bw = bw, wv =wv0,
                 OPDmap2d = None, Ampmap2d = Ampmap2d,
                 OPDmap2d_post = None)

corono0  = coro.design.APLC2d(**params)

#%%
"""
### Filepaths for the results
"""          
str_common = '_mono_nmap{:05d}_nlam{:04d}_saxomap_i{:05d}_f{:05d}_band{}'.format(nmap, nlam, saxomap_i, saxomap_f,band)
str_offaxis = '_sep{:04d}mas'.format(int(round(sep_mas_p)))

# filepaths for the images
fname_direct_mono_img_f     = 'direct' + str_common + '_img_f.fits'
fname_corono_mono_img_f     = 'corono' + str_common + '_img_f.fits'
fpath_direct_mono_img_f     = fdir_results / fname_direct_mono_img_f
fpath_corono_mono_img_f     = fdir_results / fname_corono_mono_img_f

# filepaths for the profiles
fname_direct_mono_prf_avg_f = 'direct' + str_common + '_prf_avg_f.fits'
fname_corono_mono_prf_avg_f = 'corono' + str_common + '_prf_avg_f.fits'
fname_direct_mono_prf_std_f = 'direct' + str_common + '_prf_std_f.fits'
fname_corono_mono_prf_std_f = 'corono' + str_common + '_prf_std_f.fits'
fpath_direct_mono_prf_avg_f = fdir_results / fname_direct_mono_prf_avg_f
fpath_corono_mono_prf_avg_f = fdir_results / fname_corono_mono_prf_avg_f
fpath_direct_mono_prf_std_f = fdir_results / fname_direct_mono_prf_std_f
fpath_corono_mono_prf_std_f = fdir_results / fname_corono_mono_prf_std_f

# filepaths for the images for the off-axis planet
fname_direct_mono_img_fp = 'direct' + str_common + '_img_f' + str_offaxis + '.fits'
fname_corono_mono_img_fp = 'corono' + str_common + '_img_f' + str_offaxis + '.fits'
fpath_direct_mono_img_fp = fdir_results / fname_direct_mono_img_fp
fpath_corono_mono_img_fp = fdir_results / fname_corono_mono_img_fp


#%%
"""
### Image generation
"""
# definition of the coronagraph class
if kw_aberr:
    OPDmap2d0 = (beta_wfs*ZELDAmapnm3d[imap0]+Apod2d_OPDmapnm\
                +defo_ampl*Defo_mapnm2d\
                +tipp_ampl*Tipp_mapnm2d\
                +tilt_ampl*Tilt_mapnm2d)*1e-9


for imap in range(nmap):
    t0 = time.time()
    OPDmap2d = None
    if kw_aberr:           
        if kw_saxo and kw_2nddate:
            OPDmap2d = OPDmap2d0 + SAXOmapnm3d[saxomap_i+imap]*1e-9
        else:
            OPDmap2d = OPDmap2d0*1.
    direct_mono_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d, poly=False)
    corono_mono_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d, poly=False)                

    t1 = time.time()
    if (imap+1) % 10 == 0: 
        print('map {1}/{2}, computation time: {0:.2f}s'.format(t1-t0, imap+1, nmap))

# computation of the averaged images
direct_mono_img_f /= nmap
corono_mono_img_f /= nmap

for ilam in range(nlam):
    # image normalization
    direct_peak_val[ilam] = direct_mono_img_f[ilam].max()
    direct_mono_img_f[ilam] /= direct_peak_val[ilam]
    corono_mono_img_f[ilam] /= direct_peak_val[ilam]
     
    # computation of the averaged and standard deviation profiles of the images   
    direct_mono_prf_avg_f[ilam], rad_direct = imutils.profile(direct_mono_img_f[ilam], type='mean')
    corono_mono_prf_avg_f[ilam], rad_corono = imutils.profile(corono_mono_img_f[ilam], type='mean')
    direct_mono_prf_std_f[ilam], rad_direct = imutils.profile(direct_mono_img_f[ilam], type='std')
    corono_mono_prf_std_f[ilam], rad_corono = imutils.profile(corono_mono_img_f[ilam], type='std')

#%%
"""
### Image generation for the off-axis companion
"""
# Generation of a tip and tilt mode
opd_p0 = wv0*(sep_loD_p/4)*(np.sin(theta_rad_p)*Tipp_mapnm2d + np.cos(theta_rad_p)*Tilt_mapnm2d)

for imap in range(nmap):
    t0 = time.time()
    OPDmap2d = None
    if kw_aberr:           
        if kw_saxo and kw_2nddate:
            OPDmap2d = OPDmap2d0 + SAXOmapnm3d[saxomap_i+imap]*1e-9 + opd_p0
        else:
            OPDmap2d = OPDmap2d0*1. + opd_p0
    direct_mono_img_fp += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d, poly=False)
    corono_mono_img_fp += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d, poly=False)                

    t1 = time.time()
    if (imap+1) % 10 == 0: 
        print('map {1}/{2}, computation time: {0:.2f}s'.format(t1-t0, imap+1, nmap))

# computation of the averaged images
direct_mono_img_fp /= nmap
corono_mono_img_fp /= nmap

for ilam in range(nlam):
    # image normalization
    direct_mono_img_fp[ilam] /= direct_peak_val[ilam]
    corono_mono_img_fp[ilam] /= direct_peak_val[ilam]

#%%
"""
### Photometry and spectra for the star 
"""
star_spec = fits.getdata(fpath_spectra_star)
plnt_spec = fits.getdata(fpath_spectra_plnt)

star_wave = star_spec[0]
star_flux = star_spec[1]

plnt_wave = plnt_spec[0]
plnt_flux = plnt_spec[1]



#%%
"""
### Photometry and spectra for the planet 
"""

plt.figure(0)
plt.clf()
plt.plot(star_wave, np.log10(star_flux))
plt.plot(plnt_wave, np.log10(plnt_flux))
plt.show()



#%% saving of the images
"""
### File saving
"""
if do_sav:
    data_list = [direct_mono_img_f,corono_mono_img_f,direct_mono_prf_avg_f,
                 corono_mono_prf_avg_f,direct_mono_prf_std_f,corono_mono_prf_std_f,
                 direct_mono_img_fp, corono_mono_img_fp]
    fpath_list = [fpath_direct_mono_img_f,fpath_corono_mono_img_f,fpath_direct_mono_prf_avg_f,
                  fpath_corono_mono_prf_avg_f,fpath_direct_mono_prf_std_f,fpath_corono_mono_prf_std_f,
                  fpath_direct_mono_img_fp, fpath_corono_mono_img_fp]
    nlist = len(data_list)
    
    for ilist in range(nlist):
    # save in FITS format
        hdu_prim = fits.PrimaryHDU()
        hdu_img  = fits.ImageHDU(data_list[ilist])
        hdu_wave = fits.BinTableHDU.from_columns([
            fits.Column(name='wave', unit='nm', array=wv_t, format='D'),
            fits.Column(name='dwave', unit='nm', array=dwv_t, format='D')
        ])
    
        # set some keywords in primary header
        hdu_prim.header['BAND']     = (band, 'Filter')
        hdu_prim.header['WAVE_MIN'] = (wv_t[0], 'Minimum wavelength [nm]')
        hdu_prim.header['WAVE_CEN'] = (wv_t[nlam//2], 'Central wavelength [nm]')
        hdu_prim.header['WAVE_MAX'] = (wv_t[-1], 'Maximum wavelength [nm]')
        hdu_prim.header['RESOL']    = (wv_R, 'Spectral resolution')
        hdu_prim.header['PIXELSIM'] = (pixel, 'Input simulation pixel size [mas]')
    
        hdu = fits.HDUList([hdu_prim, hdu_img, hdu_wave])
    
        hdu.writeto(fpath_list[ilist], overwrite=True)    
    