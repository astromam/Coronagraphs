#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:51:22 2022

@author: mndiaye
"""
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
import logging

import scipy.interpolate as interpolate
import astropy.units as u

from vigan.astro import skycalc

import matplotlib.pyplot as plt

# logging
logging.basicConfig(format='[%(asctime)s - %(process)6s - %(levelname)-8s] %(message)s', level='INFO')
_log = logging.getLogger(__name__)

#%%
"""
### Spectral binning function
"""
def spectral_binning(wave, dwave, obj_wave, obj_phot):
    '''
    Spectral binning of a spectrum on an irregular wavelength grid

    The function expects arrays with units but ultimately works with
    unitless arrays to avoid some conversion issues and to speed 
    things up

    Parameters
    ----------
    wave : array
        Wavelength array, in micron

    dwave : array
        Wavelength bins array, in micron

    obj_wave : array
        Wavelength array that needs binning, in micron

    obj_phot : array
        Photon array that needs binning, in phot/s/m^2/micron

    Returns
    -------
    obj_phot_bin : array
        Binned photon array, in phot/s/m^2
    '''

    # work with unitless variables
    wave_ul     = wave.to('micron').value
    dwave_ul    = dwave.to('micron').value
    obj_wave_ul = obj_wave.to('micron').value
    obj_phot_ul = obj_phot.to('ph s**-1 m**-2 micron**-1').value

    # spectral bins bounds
    wave_min_ul = wave_ul
    wave_max_ul = np.append(wave_ul[1:], wave_ul[-1]+dwave_ul[-1])

    # combined wavelength vector
    obj_wave_new_ul  = np.unique(np.append(obj_wave_ul, [wave_min_ul, wave_max_ul]))
    obj_dwave_new_ul = np.diff(obj_wave_new_ul)
    obj_wave_new_ul  = obj_wave_new_ul[:-1]

    # interpolate Flambda values on new combined wavelength vector
    interp_phot  = interpolate.interp1d(obj_wave_ul, obj_phot_ul)
    obj_phot_new_ul = interp_phot(obj_wave_new_ul)        # ph/s/m2/micron

    # integrated in individual bins
    obj_phot_int_ul = obj_phot_new_ul * obj_dwave_new_ul  # ph/s/m2

    # sum in final bins
    obj_phot_bin_ul = np.zeros(len(wave))
    for idx, (wmin, wmax) in enumerate(zip(wave_min_ul, wave_max_ul)):
        ii = np.where((wmin <= obj_wave_new_ul) & (obj_wave_new_ul < wmax))
        obj_phot_bin_ul[idx] = obj_phot_int_ul[ii].sum()

    # reapply units
    return obj_phot_bin_ul * u.ph * u.s**-1 * u.m**-2



#%% APLC2d tests
"""
### Parameters
"""
# Coronagraph type
corono_name = 'APLC' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = True
CtrBtwnPix2 = False
Pupil2dSym  = False

# Telescope characteristics
dAper     = 8
Fratio    = 40

# telescope parameters (HSIM values)
COtel = 0.14      # central obscuration
area  = np.pi*(dAper/2)**2*(1 - COtel**2)    # [m^2]

# Focal plane mask 
mas2rad   = np.pi/(180.*3600*1000) # Conversion factor from mas to rads
rad2mas   = 1/mas2rad
rMask_m   = 287e-6/2.         # mask size in m

# spatial sampling
nPup   = 100   # pupil
nFPM   = 200   # focal plane mask
nImg2d = 64    # final image plane 

# simulation configuration   
saxofudge    = 1. #80/120.
saxomap_i    = 0  # saxo first screen
saxomap_f    = 41399  # saxo last screen

# seeing for on-sky observations
seeing = 0.7

# test on the order of the min and max number of saxo phase screen
if saxomap_i <= saxomap_f:
    nsaxomap     = saxomap_f - saxomap_i + 1
else:
    raise NameError('initial saxo map (saxomap_i={0}) must be smaller than final saxo map (saxomap_f={1})!'.format(saxomap_i, saxomap_f))

ndefo = 21

# save multi-spectral images
do_sav = True 

# case with planet for plots
kwd_pla = False

# Planet position properties
sep_mas_p   = 16*8  #5*pscale      # planet separation in mas
theta_deg_p = 0  # planet position angle in degrees

# observation parameters
exposure  = 0      # exposure number in the sequence
airmass   = 1.2    # airmass for exposure
DIT       = 60     # nsaxomap1380.      # sec

# telescope and instrument transmission]
tel_transmission = 1
inst_transmission = 1

# Noise
kwd_noi = False
std_ron = 1 # photo-electrons

# Photometry
kwd_sav_onlyphot = False

# planet flux fudge factor
plnt_flux_fudge_factor = 1000

# stellar parameters
star_SpT    = 'F4'
star_mass   = 1.5
star_age    = 20
star_dist   = 50 
#star_magH = 4.0
star_band   = 'J'
star_wv_res = 1000

plnt_mass   = 1
plnt_age    = 20
plnt_dist   = 50
plnt_wv_res = 1000

t0_sim = time.time()
    
#%%
"""
### Spectral parameters
"""
band = 'BB_H'
if band == 'H2':
    nlam = 11
    wv0   = 1.593e-6
    width = 52e-9
elif band == 'BB_H':
    nlam  = 357 #1785
    wv0   = 1625e-9 #1.593e-6
    width = 290e-9  #52e-9
elif band == 'BB_J':
    nlam  = 386 #1928
    wv0   = 1245e-9
    width = 240e-9           
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
else:
    dwv_t[0] = 1e-9
    wv_R      = wv0/dwv_t[0]

# compute spatial frequencies in the final image plane
pixel  = 16. # IRDIS pixel sampling [mas/pix]
loD    = wv0/dAper*180/np.pi*3600*1000/pixel
nFre2d = nImg2d/loD    # spatial frequencies in the final image plane

# Focal plane mask 
rMask     = rMask_m/(wv0*Fratio)  # mask size in lam0/D
print(f'Mask radius: {rMask:.3f}lam0/D at {wv0*1e6:.3f}um')
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
str_aberr = 'with_aberr'    
str_date  = '2018-04-01'
str_obs   = 'internal'
str_corr  = 'before_correction'
str_saxo  = ''
imap0     = 0
nmap      = 1
beta_wfs  = 1./0.90
str_saxo_tmp  = 'wo_saxo'
str_date = '2018-04-03'
str_obs  = 'sky'
str_saxo = 'with_saxo'
nmap     = nsaxomap*1
beta_wfs = 1./0.6

#%%
#fdir = Path('../../').resolve()
fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs').resolve()
fdir_dat = fdir / 'data' / '2D' / 'medres_sim' 
fdir_spectra = fdir / 'data' / '2D' / 'package_simu_spectra'
fdir_sky     = fdir / 'data' / '2D' / 'skytable'

fdir_res = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_date / str_obs / str_saxo / str_corr  
if not os.path.exists(fdir_res):
    os.makedirs(fdir_res)

#%%
"""
### Filenames for the sources
"""
fname_Apod2d          = f'SPHERE_APO1_field_transmission_map_nPup{nPup:04d}.fits'
fname_Apod2d_OPDmapnm = f'apo_substrate_D1_nPup{nPup:04d}.fits'
fname_Ampmap2d        = f'2018-04-03_night_sphere_pupil_clear_sky_FeII_field_nPup{nPup:04d}.fits'
fname_SAXOmapnm3d     = f'2018-04-04T00-41-34-saxo_residual_turbulence_time30.0sec_seeing{seeing:.1f}as_tiptilt1_gains0_fitting1_alias1_nPup{nPup:04d}.fits'
fname_ZELDAmapnm3d    = f'2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd_nPup{nPup:04d}.fits'
fname_LyotStop2d      = f'sphere_stop_ST_ALC2_nPup{nPup:04d}.fits'

#%%
"""
### Filepaths for the file sources
"""
fpath_Apod2d          = fdir_dat / fname_Apod2d
fpath_Apod2d_OPDmapnm = fdir_dat / fname_Apod2d_OPDmapnm
fpath_Ampmap2d        = fdir_dat / fname_Ampmap2d
fpath_SAXOmapnm3d     = fdir_dat / fname_SAXOmapnm3d
fpath_ZELDAmapnm3d    = fdir_dat / fname_ZELDAmapnm3d   
fpath_LyotStop2d      = fdir_dat / fname_LyotStop2d    
  
#%%
"""
### Filename for the spectra
"""
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

fname_spectra_tell = 'sphplus_medres_teltransm_{:.1f}_{:.1f}_mic_R{}.fits'.format(star_wv_min, star_wv_max, int(round(star_wv_res)))

#%%
"""
### Filepath for the spectra
"""
fpath_spectra_star = fdir_spectra / fname_spectra_star
fpath_spectra_plnt = fdir_spectra / fname_spectra_plnt
fpath_spectra_tell = fdir_spectra / fname_spectra_tell


  
#%% 
"""
### File reading
"""
### Pupil
Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)
    
### Apodization
Apod2d = fits.getdata(fpath_Apod2d)

### Apodization OPD map
Apod2d_OPDmapnm = fits.getdata(fpath_Apod2d_OPDmapnm)

### Amplitude errors
Ampmap2d = fits.getdata(fpath_Ampmap2d)

### Phase errors
ZELDAmapnm3d = fits.getdata(fpath_ZELDAmapnm3d)

### SAXO maps
SAXOmapnm3d = np.empty((nmap, nPup, nPup))
SAXOmapnm3d = fits.getdata(fpath_SAXOmapnm3d)[saxomap_i:saxomap_i+nmap,:,:]
        
### Lyot Stop
LyotStop2d = fits.getdata(fpath_LyotStop2d)

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

# define the averaged image for the planet
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
str_common = '_nmap{:05d}_i{:05d}_f{:05d}_band{}_nlam{:04d}'.format(nmap, saxomap_i, saxomap_f,band, nlam)
str_offaxis_fp = '_sep{:04d}mas'.format(int(round(sep_mas_p)))
str_sphplus = '_{}_{}dMSun_{}MJup_{}Myr'.format(star_SpT, int(round(star_mass*10)), int(round(plnt_mass)), int(round(star_age)))
str_offaxis = ''
if kwd_pla:
    str_offaxis = str_offaxis_fp

str_sphplus = '_{}_{}dMSun_{}Myr'.format(star_SpT, int(round(star_mass*10)), int(round(star_age)))   
if kwd_pla:
    str_sphplus += '_{}MJup'.format(int(round(plnt_mass)))
str_noi = ''
if kwd_noi:
    str_noi = '_noi'

str_ff = ''
if kwd_pla:
    str_ff = f'_{plnt_flux_fudge_factor:03d}'

# filepaths for the images
fname_direct_mono_img_f     = 'dir' + str_common + '_img_f.fits'
fname_corono_mono_img_f     = 'cor' + str_common + '_img_f.fits'
fpath_direct_mono_img_f     = fdir_res / fname_direct_mono_img_f
fpath_corono_mono_img_f     = fdir_res / fname_corono_mono_img_f

# filepaths for the profiles
fname_direct_mono_prf_avg_f = 'dir' + str_common + '_prf_avg_f.fits'
fname_corono_mono_prf_avg_f = 'cor' + str_common + '_prf_avg_f.fits'
fname_direct_mono_prf_std_f = 'dir' + str_common + '_prf_std_f.fits'
fname_corono_mono_prf_std_f = 'cor' + str_common + '_prf_std_f.fits'
fpath_direct_mono_prf_avg_f = fdir_res / fname_direct_mono_prf_avg_f
fpath_corono_mono_prf_avg_f = fdir_res / fname_corono_mono_prf_avg_f
fpath_direct_mono_prf_std_f = fdir_res / fname_direct_mono_prf_std_f
fpath_corono_mono_prf_std_f = fdir_res / fname_corono_mono_prf_std_f

# filepaths for the images for the off-axis planet
fname_direct_mono_img_fp = 'dir' + str_common + '_img_f' + str_offaxis + '.fits'
fname_corono_mono_img_fp = 'cor' + str_common + '_img_f' + str_offaxis + '.fits'
fpath_direct_mono_img_fp = fdir_res / fname_direct_mono_img_fp
fpath_corono_mono_img_fp = fdir_res / fname_corono_mono_img_fp

# filepath for the images with star and planet
fname_direct_cube = 'dir' + str_common + '_img_f' + str_offaxis + str_sphplus + str_noi + '.fits'
fname_corono_cube = 'cor' + str_common + '_img_f' + str_offaxis + str_sphplus + str_noi + '.fits'
fpath_direct_cube = fdir_res / fname_direct_cube
fpath_corono_cube = fdir_res / fname_corono_cube


#%%
"""
### Image generation
"""
# # definition of SR table
# SR1_t = np.zeros((nmap))
# SR2_t = np.zeros((nmap))

# # set of points inside the pupil
# ind_Pupil2d = Pupil2d != 0

# # definition of the coronagraph class
# OPDmap2d0 = (beta_wfs*ZELDAmapnm3d+Apod2d_OPDmapnm)*1e-9

# OPDmap2d = OPDmap2d0 + SAXOmapnm3d*1e-9

# for imap in range(nmap):
#     t0 = time.time()
#     SR1_t[imap] = np.exp(-(2*np.pi*1e-9*np.std(SAXOmapnm3d[imap, ind_Pupil2d])/wv0)**2)
#     SR2_t[imap] = np.exp(-(2*np.pi*np.std(OPDmap2d[imap, ind_Pupil2d])/wv0)**2)
#     t1 = time.time()
#     if (imap+1) % 100 == 0: 
#         print(f'map {imap+1}/{nmap}, computation time: {t1-t0:.3f}s')
        
        
# SR1_mean = np.mean(SR1_t)
# SR1_std = np.std(SR1_t)
# SR2_mean = np.mean(SR2_t)
# SR2_std = np.std(SR2_t)

# print(f'SR1 = {SR1_mean:.3f} +/- {SR1_std:.3f} at lambda={wv0*1e6:.3f}um')
# print(f'SR2 = {SR2_mean:.3f} +/- {SR2_std:.3f} at lambda={wv0*1e6:.3f}um')

#%%
"""
### Apodizer throughput
"""

EE_Apod = np.sum((Apod2d*Pupil2d)**2)/np.sum(Pupil2d**2)

EE_Coro = np.sum((Apod2d*Pupil2d*LyotStop2d)**2)/np.sum(Pupil2d**2)

print(f'Apodizer throughput = {EE_Apod*100:.2f}%')
print(f'Coronagraph throughput = {EE_Coro*100:.2f}%')

#%%
plt.figure(0)
plt.clf()
plt.imshow(Pupil2d)


#%%

fname_Apod2d_bis          = f'SPHERE_APO1_field_transmission_map.fits'
fname_LyotStop2d_bis      = f'sphere_stop_ST_ALC2.fits'

fpath_Apod2d_bis          = fdir_dat / fname_Apod2d_bis
fpath_LyotStop2d_bis      = fdir_dat / fname_LyotStop2d_bis    

Pupil2d_bis = aperture.vlt_pupil(384, 384, dead_actuator_diameter=0)
Apod2d_bis = fits.getdata(fpath_Apod2d_bis)
LyotStop2d_bis = fits.getdata(fpath_LyotStop2d_bis)

EE_Apod_bis = np.sum((Apod2d_bis*Pupil2d_bis)**2)/np.sum(Pupil2d_bis**2)

EE_Coro_bis = np.sum((Apod2d_bis*Pupil2d_bis*LyotStop2d_bis)**2)/np.sum(Pupil2d_bis**2)

print(f'Apodizer throughput 384 = {EE_Apod_bis*100:.2f}%')
print(f'Coronagraph throughput 384 = {EE_Coro_bis*100:.2f}%')