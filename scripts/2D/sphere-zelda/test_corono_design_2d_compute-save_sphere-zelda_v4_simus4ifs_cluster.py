#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:20:52 2018

Authors: Mamadou N'Diaye <mamadou.ndiaye@oca.eu>, Arthur Vigan <arthur.vigan@lam.fr>

License: MIT license

"""

#%% Initialization
"""
### Initialization
"""
import sys
#sys.path.append('/home/avigan/GitHub/Coronagraphs/')
#sys.path.append('/Users/avigan/Work/GitHub/Coronagraphs/')

import numpy as np
import os
import pwd
import time
from pyzelda.utils import aperture, imutils, zernike
from pathlib import Path
from astropy.io import fits
import corono as coro
import ctypes
import multiprocessing
multiprocessing.set_start_method('fork') # on py38, default is 'spawn' instead of 'fork'
import psutil
import itertools
#import tqdm

import logging
import scipy.interpolate as interpolate
import astropy.units as u
from vigan.astro import skycalc
#import matplotlib.pyplot as plt

# logging
logging.basicConfig(format='[%(asctime)s - %(process)6s - %(levelname)-8s] %(message)s', level='INFO')
_log = logging.getLogger(__name__)

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

#%%
"""
### Functions
"""
def array_to_numpy(shared_array, shape):
    '''
    Map a raw memory array to a numpy array
    '''
    if shared_array is None:
        return None

    numpy_array = np.frombuffer(shared_array, dtype=float)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array


def tpool_init(OPDmap2d0_i, SAXOmapnm3d_i, corono0_i,
               direct_mono_img_cube_i, direct_mono_img_cube_shape_i, corono_mono_img_cube_i, corono_mono_img_cube_shape_i):
    '''
    Thread pool initialization
    '''
    global OPDmap2d0, SAXOmapnm3d, corono0, direct_mono_img_cube, direct_mono_img_cube_shape, corono_mono_img_cube, corono_mono_img_cube_shape

    OPDmap2d0   = OPDmap2d0_i
    SAXOmapnm3d = SAXOmapnm3d_i
    corono0     = corono0_i
    direct_mono_img_cube       = direct_mono_img_cube_i
    direct_mono_img_cube_shape = direct_mono_img_cube_shape_i
    corono_mono_img_cube       = corono_mono_img_cube_i
    corono_mono_img_cube_shape = corono_mono_img_cube_shape_i
    
def compute_corono_image(img_index, saxo_i, saxo_f):
    '''
    Compute a series of coronagraphic images
    '''
    global OPDmap2d0, SAXOmapnm3d, corono0, direct_mono_img_cube, direct_mono_img_cube_shape, corono_mono_img_cube, corono_mono_img_cube_shape

    # shared arrays
    direct_mono_img_cube_np = array_to_numpy(direct_mono_img_cube, direct_mono_img_cube_shape)
    corono_mono_img_cube_np = array_to_numpy(corono_mono_img_cube, corono_mono_img_cube_shape)
    
    # create temporary images
    direct_mono_img_f = np.zeros((nlam, nImg2d, nImg2d))
    corono_mono_img_f = np.zeros((nlam, nImg2d, nImg2d))

    # loop on phase screens
    saxo_i = int(saxo_i)
    saxo_f = int(saxo_f)
    nmap = saxo_f - saxo_i + 1
    for imap in range(nmap):#tqdm.tqdm(range(nmap), desc="AO maps"):
        OPDmap2d = OPDmap2d0 + SAXOmapnm3d[saxo_i+imap]*1e-9
        direct_mono_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d, poly=False)
        corono_mono_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d, poly=False)

        t_mean = (time.time()-t0) / (imap+1)
        if (imap+1) % 10 == 0:
            print(f'map {imap+1}/{nmap}, average computation time: {t_mean:.2f}s')

    # save result in shared arrays
    direct_mono_img_cube_np[img_index] = direct_mono_img_f
    corono_mono_img_cube_np[img_index] = corono_mono_img_f

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

#%%

if __name__ == '__main__':
    
    t_ini = time.time()
#%%
    """
    ### Simulation case
    """
    # simulation case and directory
    if user == 'mndiaye':
        if syst == 'darwin':
            fdir = Path('~/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs').expanduser()
            sim_case = 'test' # 'test' or 'server'
        elif syst == 'linux':
            fdir = Path('/scratch/{0}/data/Coronagraphs/'.format(user)).resolve()
            sim_case = 'server' # 'test' or 'server'            
        else:
            raise ValueError('Unknown operating system {0}'.format(user))
    else:
        raise ValueError('Unknown user {0}'.format(user))

    # multi-processing (to adjust with respect to the available proc, 10 and 20 cores in fdr and x40)   
    if sim_case == 'test':
        nproc    = multiprocessing.cpu_count()//2  #15
        nmap_sub = 690                    # number of saxo maps for a single node 
    elif sim_case == 'server':
        nproc    = 15 #multiprocessing.cpu_count()//2  #15
        nmap_sub = 690                    # number of saxo maps for a single node
    else:
        raise ValueError(f'unknown {sim_case}')    

#%%
    """
    ### Science case
    """
    # spectral band
    band = 'BB_H' # 'BB_J', 'BB_H', 'H2' filters

    # science case
    sci_case    = 'mature' # 'young' or mature'
    star_SpT0   = 'A' # 'A', 'F', 'K'
    star_wv_res = 5000
    plnt_wv_res = star_wv_res

    sci_case_lst  = ['young', 'mature']
    star_SpT0_lst = ['A', 'F', 'K']
    new_lst = list(itertools.product(sci_case_lst, star_SpT0_lst))

    sci_case    = sci_case_lst[eval(sys.argv[1])]#new_lst[eval(sys.argv[1])][0] # 'young' or mature'
    star_SpT0   = star_SpT0_lst[eval(sys.argv[2])]#new_lst[eval(sys.argv[1])][1] # 'A', 'F', 'K'
    
    _log.info(f'{sci_case} system, {star_SpT0} star, {band} band')
    if sci_case == 'young':        
        if star_SpT0 == 'A':
            star_SpT = 'A0'
            star_mass   = 2.2
        elif star_SpT0 == 'F':
            star_SpT = 'F4'
            star_mass   = 1.5
        elif star_SpT0 == 'K':
            star_SpT = 'K5'
            star_mass   = 1.0
        else:
            raise ValueError(f'unknown {star_SpT0}')
        star_age    = 20
        star_dist   = 50         
        plnt_mass   = 1
    elif sci_case == 'mature':
        if star_SpT0 == 'A':
            star_SpT = 'A4'
            star_mass   = 2.2
        elif star_SpT0 == 'F':
            star_SpT = 'F3'
            star_mass   = 1.5
        elif star_SpT0 == 'K':
            star_SpT = 'K0'
            star_mass   = 1.0
        else:
            raise ValueError(f'unknown {star_SpT0}')
        star_age    = 500
        star_dist   = 20 
        plnt_mass   = 5        
    else:
        raise ValueError(f'unknown {sci_case}')
            
    if band == 'BB_H' or band == 'H2':       
        star_band = 'H'
    elif band == 'BB_J':
        star_band = 'J'
    else:
        raise ValueError(f'unknown {band} band')
        
    plnt_age    = star_age
    plnt_dist   = star_dist
    
    
#%%    
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

    # telescope parameters (HSIM values)
    COtel = 0.14      # central obscuration
    area  = np.pi*(dAper/2)**2*(1 - COtel**2)    # [m^2]

    # Focal plane mask 
    mas2rad   = np.pi/(180.*3600*1000) # Conversion factor from mas to rads
    rad2mas   = 1/mas2rad
    rMask_m   = 287e-6/2.         # mask size in m

    # sampling
    nPup   = 100   # pupil
    nFPM   = 200   # focal plane mask
    nImg2d = 64   # final image plane 

    # compute spatial frequencies in the final image plane
    pixel  = 16. # IRDIS pixel sampling [mas/pix]    

    # simulation configuration   
    kw_aberr     = True
    kw_2nddate   = True    
    kw_skyobs    = True
    kw_aftercorr = False
    kw_saxo      = True
    saxofudge    = 1 #60/120              # saxo amplitude errors fudge factor
    saxomap_i    = int(nmap_sub*(eval(sys.argv[3])))       # saxo first screen
    saxomap_f    = int(nmap_sub*(eval(sys.argv[3])+1)-1)     # saxo last screen
    # saxomap_i    = int(nmap_sub*(0))       # saxo first screen
    # saxomap_f    = int(nmap_sub*(0+1)-1)     # saxo last screen
    _log.info(f'saxomap_i: {saxomap_i}, saxomap_f {saxomap_f}')
    # seeing for on-sky observations [arcsec]
    seeing = 0.7
    
    # make sure we have a number of phase screens multiple of the number of CPUs
    nsaxomap  = saxomap_f - saxomap_i + 1
    nsaxomap  = nsaxomap - (nsaxomap % nproc)
    _log.info(f'number of saxomaps to work with: {nsaxomap:05d}')
    
    ndefo = 21
    defo_ampl = 0#-100 + 10.*np.arange(21)
    tipp_ampl = 0
    tilt_ampl = 0 

    # save multi-spectral images
    do_sav = True 
    
    # case with planet for plots
    kwd_pla = False
    
    # Planet position properties
    sep_mas_p   = 16.*16  #5*pscale      # planet separation in mas
    theta_deg_p = 0  # planet position angle in degrees
    
    # observation parameters
    exposure  = 0      # exposure number in the sequence
    airmass   = 1.2    # airmass for exposure
    DIT       = 60     #nsaxomap/1380      # sec
    
    # telescope and instrument transmission]
    tel_transmission = 1
    inst_transmission = 1

    # Noise
    kwd_noi = False
    std_ron = 1 # photo-electrons

    # Photometry
    kwd_sav_onlyphot = True
    
    # planet flux fudge factor
    plnt_flux_fudge_factor = 1000

    #%%
    """
    ### Spectral parameters
    """
    if band == 'H2':
        nlam  = 11
        wv0   = 1.593e-6
        width = 52e-9
    elif band == 'BB_H':
        nlam  = 1785
        wv0   = 1625e-9
        width = 290e-9
    elif band == 'BB_J':
        nlam  = 1928
        wv0   = 1245e-9
        width = 240e-9            
    else:
        raise ValueError(f'Unknown {band} band') 
    _log.info(f'number of wavelengths: {nlam}')
    
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
        wv_R   = wv0/dwv_t[0]
    
    # compute spatial frequencies in the final image plane
    pixel  = 12.25 # IRDIS pixel sampling [mas/pix]
    loD    = wv0/dAper*180/np.pi*3600*1000/pixel
    nFre2d = nImg2d/loD    # spatial frequencies in the final image plane
    
    # Focal plane mask 
    rMask     = rMask_m/(wv0*Fratio)  # mask size in lam0/D
    rMask_mas = rMask * (wv0/dAper)/mas2rad
    _log.info(f'Mask radius: {rMask_mas:.2f} mas at {wv0*1e6:.3f}um')
    
    # planet position properties
    sep_loD_p   = sep_mas_p/(wv_t[nlam//2]/dAper*rad2mas) # planet separation in lam0/D
    theta_rad_p = theta_deg_p*(np.pi/180)          # planet position angle in radians
    _log.info(f'planet sep: {sep_mas_p:6.1f}mas, {sep_loD_p:6.2f}lam0/D')
    _log.info(f'planet ang: {theta_deg_p:6.1f}deg, {theta_rad_p:6.2f}rad')
    pla_dRA   = sep_mas_p*np.cos(theta_rad_p)      # delta in RA, in mas
    pla_dDEC  = sep_mas_p*np.sin(theta_rad_p)      # delta in DEC, in mas

    
    
    #%%
    """
    ### Directories
    """    
    # if kw_aberr is False:
    #     str_aberr = 'wo_aberr'
    #     str_date  = ''
    #     if kw_skyobs:
    #         str_obs   = 'sky'
    #     else:
    #         str_obs   = 'internal'
    #     str_corr  = ''
    #     str_saxo  = ''
    #     str_saxoset= ''
    #     nmap      = 1 
    # else:
    #     str_aberr = 'with_aberr'    
    #     str_date  = '2018-04-01'
    #     str_obs   = 'internal'
    #     str_corr  = 'before_correction'
    #     str_saxo  = ''
    #     imap0     = 0
    #     nmap      = 1
    #     beta_wfs  = 1./0.90
    #     str_saxo_tmp  = 'wo_saxo'
    #     if kw_2nddate:
    #         str_date = '2018-04-03'
    #     if kw_skyobs:
    #         str_obs  = 'sky'
    #     if kw_aftercorr:
    #         str_corr = 'after_correction'
    #         imap0    = 3
    #         beta_wfs = 1/0.95
    #     if kw_saxo and kw_2nddate:
    #         str_saxo = 'with_saxo'
    #         nmap     = nsaxomap*1
    #         beta_wfs = 1/0.6
    
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
#    fdir = Path('~/data/ZELDA/CoroSimulations/').expanduser()
    

    

    # fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
    # fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
    # fdir_saxo    = fdir / 'data' / '2D' / 'ZELDA' / '2018-04-03'
    # fdir_spectra = fdir / 'data' / '2D' / 'package_simu_spectra'
    # fdir_sky     = fdir / 'data' / '2D' / 'skytable'

    # if kw_aberr:
    #     fdir_res = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_date / str_obs / str_saxo / str_corr  
    # else:
    #     fdir_res = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_obs / str_saxo / str_corr  

    # if not os.path.exists(fdir_res):
    #     os.makedirs(fdir_res)
    

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
    # fname_Apod2d     = 'SPHERE_APO1_field_transmission_map.fits'
    # fname_Apod2d_OPDmapnm = 'apo_substrate_D1.fits'
    # fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
    # fname_LyotStop2d = 'sphere_stop_ST_ALC2.fits'

    # if kw_aberr is True:
    #     if kw_skyobs is True:
    #         fname_Ampmap2d   = '2018-04-01_night_sphere_pupil_clear_sky_FeII_field.fits'
    #         fname_ZELDAmapnm3d = '2018-04-01_night_ncpa_loop_700modes_5_ncpa_loop_opd.fits'
    #         if kw_2nddate is True:
    #             fname_Ampmap2d   = '2018-04-03_night_sphere_pupil_clear_sky_FeII_field.fits'
    #             fname_ZELDAmapnm3d = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
    #     else:
    #         fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
    #         fname_ZELDAmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'        
    #         if kw_2nddate is True:
    #             fname_ZELDAmapnm3d = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'

    #     if kw_saxo and kw_2nddate:
    #         fname_SAXOmapnm3d = '2018-04-04T00-41-34-saxo_residual_turbulence_time=30.0sec_seeing={:.1f}as_tiptilt=1_gains=0_fitting=1_alias=1.fits'.format(seeing)

    fname_Apod2d          = f'SPHERE_APO1_field_transmission_map_nPup{nPup:04d}.fits'
    fname_Apod2d_OPDmapnm = f'apo_substrate_D1_nPup{nPup:04d}.fits'
    fname_Ampmap2d        = f'2018-04-03_night_sphere_pupil_clear_sky_FeII_field_nPup{nPup:04d}.fits'
    fname_SAXOmapnm3d     = f'2018-04-04T00-41-34-saxo_residual_turbulence_time30.0sec_seeing{seeing:.1f}as_tiptilt1_gains0_fitting1_alias1_nPup{nPup:04d}.fits'
    fname_ZELDAmapnm3d    = f'2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd_nPup{nPup:04d}.fits'
    fname_LyotStop2d      = f'sphere_stop_ST_ALC2_nPup{nPup:04d}.fits'

    #%% 
    """
    ###Filepaths for the file sources
    """
    # fpath_Apod2d          = fdir_pupils / fname_Apod2d
    # fpath_Apod2d_OPDmapnm = fdir_pupils / fname_Apod2d_OPDmapnm
    # fpath_Ampmap2d        = fdir_zelda / fname_Ampmap2d

    # if kw_aberr:
    #     fpath_ZELDAmapnm3d = fdir_zelda  / fname_ZELDAmapnm3d   
    #     if kw_saxo and kw_2nddate:
    #         fpath_SAXOmapnm3d = fdir_saxo / fname_SAXOmapnm3d

    # fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d   
    
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
    # Pupil
    Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)

    #%% Apodization
    Apod2d = fits.getdata(fpath_Apod2d)

    #%% APodization OPD map
    Apod2d_OPDmapnm = fits.getdata(fpath_Apod2d_OPDmapnm)
    # Apod2d_OPDmapnm[np.isnan(Apod2d_OPDmapnm)] = 0

    #%% Amplitude errors
    Ampmap2d = fits.getdata(fpath_Ampmap2d)

    #%% Phase errors
    ZELDAmapnm3d = fits.getdata(fpath_ZELDAmapnm3d)
    
    #%% SAXO maps
    SAXOmapnm3d = np.empty((nmap, nPup, nPup))
    SAXOmapnm3d = fits.getdata(fpath_SAXOmapnm3d)[saxomap_i:saxomap_i+nmap,:,:]
            
    #%% Lyot Stop
    LyotStop2d = fits.getdata(fpath_LyotStop2d)

    #%%
    """
    ### Array Initialization
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

    #%% definition of the coronagraph class parameters
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
    ### Filepaths for the file results
    """

    str_common = '_nmap{:05d}_i{:05d}_f{:05d}_band{}_nlam{:04d}'.format(nmap, saxomap_i, saxomap_f,band, nlam)
    str_offaxis_fp = '_sep{:04d}mas'.format(int(round(sep_mas_p)))
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
    fname_direct_mono_img_fp = 'dir' + str_common + '_img_f' + str_offaxis_fp + '.fits'
    fname_corono_mono_img_fp = 'cor' + str_common + '_img_f' + str_offaxis_fp + '.fits'
    fpath_direct_mono_img_fp = fdir_res / fname_direct_mono_img_fp
    fpath_corono_mono_img_fp = fdir_res / fname_corono_mono_img_fp
    
    # filepath for the images with star and planet
    fname_direct_cube = 'dir' + str_common + '_img_f' + str_offaxis + str_sphplus + str_noi + str_ff + '.fits'
    fname_corono_cube = 'cor' + str_common + '_img_f' + str_offaxis + str_sphplus + str_noi + str_ff + '.fits'
    fpath_direct_cube = fdir_res / fname_direct_cube
    fpath_corono_cube = fdir_res / fname_corono_cube

    #%%
    # definition of the coronagraph class
    OPDmap2d0 = (beta_wfs*ZELDAmapnm3d+Apod2d_OPDmapnm)*1e-9

    t0 = time.time()

    # create shared arrays
    direct_mono_img_cube_shape = (nproc, nlam, nImg2d, nImg2d)
    direct_mono_img_cube_data  = multiprocessing.RawArray(ctypes.c_double, int(np.prod(direct_mono_img_cube_shape)))
    direct_mono_img_cube_np    = array_to_numpy(direct_mono_img_cube_data, direct_mono_img_cube_shape)
    
    corono_mono_img_cube_shape = (nproc, nlam, nImg2d, nImg2d)
    corono_mono_img_cube_data  = multiprocessing.RawArray(ctypes.c_double, int(np.prod(corono_mono_img_cube_shape)))
    corono_mono_img_cube_np    = array_to_numpy(corono_mono_img_cube_data, corono_mono_img_cube_shape)

    # create thread pool
    tpool = multiprocessing.Pool(processes=nproc, initializer=tpool_init,
                                 initargs=(OPDmap2d0, SAXOmapnm3d, corono0, direct_mono_img_cube_data, direct_mono_img_cube_shape,
                                           corono_mono_img_cube_data, corono_mono_img_cube_shape))
    # tpool_init(OPDmap2d0, SAXOmapnm3d, corono0, direct_mono_img_cube_data, direct_mono_img_cube_shape,
    #            corono_mono_img_cube_data, corono_mono_img_cube_shape)

    # create tasks
    tasks = []
    for image_index in range(nproc):
        block = nmap / nproc
        idx_i = image_index*block
        idx_f = (image_index+1)*block-1
        tasks.append(tpool.apply_async(compute_corono_image, args=(image_index, idx_i, idx_f)))
        # compute_corono_image(image_index, idx_i, idx_f)
        # stop

    for idx, task in enumerate(tasks):
        task.wait()

    # close thread pool
    tpool.close()
    tpool.join()

    direct_mono_img_cube_np = array_to_numpy(direct_mono_img_cube_data, direct_mono_img_cube_shape)
    corono_mono_img_cube_np = array_to_numpy(corono_mono_img_cube_data, corono_mono_img_cube_shape)
    
    direct_mono_img_f += direct_mono_img_cube_np.sum(axis=0)
    corono_mono_img_f += corono_mono_img_cube_np.sum(axis=0)
   
    # computation of the averaged images
    direct_mono_img_f /= nmap
    corono_mono_img_f /= nmap

    pbar_lam = range(nlam)#tqdm.tqdm(range(nlam), desc="wavelength", position=0)
    for ilam in pbar_lam:
        # image normalization
        direct_peak_val[ilam] = direct_mono_img_f[ilam].max()
        #pbar_lam.set_description(f'intensity peak at lam {ilam}: {direct_peak_val[ilam]}')
        direct_mono_img_f[ilam] /= direct_peak_val[ilam]
        corono_mono_img_f[ilam] /= direct_peak_val[ilam]

        # computation of the averaged and standard deviation profiles of the images   
        direct_mono_prf_avg_f[ilam], rad_direct = imutils.profile(direct_mono_img_f[ilam], type='mean')
        corono_mono_prf_avg_f[ilam], rad_corono = imutils.profile(corono_mono_img_f[ilam], type='mean')
        direct_mono_prf_std_f[ilam], rad_direct = imutils.profile(direct_mono_img_f[ilam], type='std')
        corono_mono_prf_std_f[ilam], rad_corono = imutils.profile(corono_mono_img_f[ilam], type='std')

    #%%
    # # definition of the coronagraph class
    
    if kwd_pla:
        Tipp_mapnm2d = zernike.zernike1(2, npix=nPup, outside=0.)
        Tilt_mapnm2d = zernike.zernike1(3, npix=nPup, outside=0.) 
        opd_p0 = wv0*(sep_loD_p/4)*(np.sin(theta_rad_p)*Tipp_mapnm2d + np.cos(theta_rad_p)*Tilt_mapnm2d)
    
    
        t0 = time.time()

        # create shared arrays
        direct_mono_img_fp_cube_shape = (nproc, nlam, nImg2d, nImg2d)
        direct_mono_img_fp_cube_data  = multiprocessing.RawArray(ctypes.c_double, int(np.prod(direct_mono_img_cube_shape)))
        direct_mono_img_fp_cube_np    = array_to_numpy(direct_mono_img_fp_cube_data, direct_mono_img_fp_cube_shape)
        
        corono_mono_img_fp_cube_shape = (nproc, nlam, nImg2d, nImg2d)
        corono_mono_img_fp_cube_data  = multiprocessing.RawArray(ctypes.c_double, int(np.prod(corono_mono_img_cube_shape)))
        corono_mono_img_fp_cube_np    = array_to_numpy(corono_mono_img_fp_cube_data, corono_mono_img_fp_cube_shape)

        # create thread pool
        tpool = multiprocessing.Pool(processes=nproc, initializer=tpool_init,
                                      initargs=(OPDmap2d0+opd_p0, SAXOmapnm3d, corono0, direct_mono_img_fp_cube_data, direct_mono_img_fp_cube_shape,
                                                corono_mono_img_fp_cube_data, corono_mono_img_fp_cube_shape))
        # tpool_init(OPDmap2d0, SAXOmapnm3d, corono0, direct_mono_img_cube_data, direct_mono_img_cube_shape,
        #            corono_mono_img_cube_data, corono_mono_img_cube_shape)

        # create tasks
        tasks = []
        for image_index in range(nproc):
            block = nmap / nproc
            idx_i = image_index*block
            idx_f = (image_index+1)*block-1
            tasks.append(tpool.apply_async(compute_corono_image, args=(image_index, idx_i, idx_f)))
            # compute_corono_image(image_index, idx_i, idx_f)
            # stop

        for idx, task in enumerate(tasks):
            task.wait()

        # close thread pool
        tpool.close()
        tpool.join()

        direct_mono_img_fp_cube_np = array_to_numpy(direct_mono_img_fp_cube_data, direct_mono_img_fp_cube_shape)
        corono_mono_img_fp_cube_np = array_to_numpy(corono_mono_img_fp_cube_data, corono_mono_img_fp_cube_shape)
        
        direct_mono_img_fp += direct_mono_img_fp_cube_np.sum(axis=0)
        corono_mono_img_fp += corono_mono_img_fp_cube_np.sum(axis=0)
    
    
        # computation of the averaged images
        direct_mono_img_fp /= nmap
        corono_mono_img_fp /= nmap
    
        for ilam in range(nlam):#tqdm.tqdm(range(nlam), desc="wavelength"):
            # image normalization
            direct_mono_img_fp[ilam] /= direct_peak_val[ilam]
            corono_mono_img_fp[ilam] /= direct_peak_val[ilam]

    #%%
    """
    ### Unit conversion
    """
    area     *= u.m**2
    DIT      *= u.s
    
    #%%
    """
    ### Photometry and spectra for the star, the planet and the telluric lines 
    """
    star_spec = fits.getdata(fpath_spectra_star)
    if kwd_pla:
        plnt_spec = fits.getdata(fpath_spectra_plnt)
    tell_spec = fits.getdata(fpath_spectra_tell)
    
    star_wave = star_spec[0]
    star_flux = star_spec[1]
    
    if kwd_pla:
        plnt_wave = plnt_spec[0]
        plnt_flux = plnt_spec[1]
    
    tell_wave = tell_spec[0]
    tell_flux = tell_spec[1]
    
    #%%
    """
    ### Photometry and spectra for the planet 
    """    
    #%%
    
    star_wave *= u.um
    star_flux *= u.W / u.m**2 / u.um
    star_phot = star_flux.to('ph s**-1 m**-2 micron**-1', equivalencies=u.spectral_density(star_wave))
    
    #%%
    if kwd_pla:
        plnt_wave *= u.um
        plnt_flux *= u.W / u.m**2 / u.um
        plnt_phot = plnt_flux.to('ph s**-1 m**-2 micron**-1', equivalencies=u.spectral_density(plnt_wave))
    
    #%%
    tell_wave *= u.um
    #tell_flux *= u.dimensionless_unscaled
    
    
    #%%
    wv_um_t = wv_t * 1e6
    dwv_um_t = dwv_t * 1e6 
    
    #%%
    wv_um_t *= u.um
    dwv_um_t *= u.um
    
    #%%
    """
    ### bin over final wavelength grid
    """
    
    star_phot = spectral_binning(wv_um_t, dwv_um_t, star_wave, star_phot)
    if kwd_pla:
        plnt_phot = plnt_flux_fudge_factor*spectral_binning(wv_um_t, dwv_um_t, plnt_wave, plnt_phot)
    
    #tell_phot = transmission_spectral_binning(wv_um_t, dwv_um_t, tell_wave, tell_flux)
    
    #%%
    """
    ### apply stellar photometry
    """
    direct_mono_img_f_obs = direct_mono_img_f * u.dimensionless_unscaled 
    direct_mono_img_f_obs *= star_phot[:, None, None]
    
    corono_mono_img_f_obs = corono_mono_img_f * u.dimensionless_unscaled 
    corono_mono_img_f_obs *= star_phot[:, None, None]
    
    #%%
    """
    ### apply planetary photometry
    """
    if kwd_pla:
        direct_mono_img_fp_obs = direct_mono_img_fp * u.dimensionless_unscaled 
        direct_mono_img_fp_obs *= plnt_phot[:, None, None]
        
        corono_mono_img_fp_obs = corono_mono_img_fp * u.dimensionless_unscaled 
        corono_mono_img_fp_obs *= plnt_phot[:, None, None]
    
    #%%
    """
    ### add stellar and planetary signal
    """
    if kwd_pla:    
        direct_cube = direct_mono_img_f_obs + direct_mono_img_fp_obs
        corono_cube = corono_mono_img_f_obs + corono_mono_img_fp_obs
    else:
        direct_cube = direct_mono_img_f_obs
        corono_cube = corono_mono_img_f_obs
    
    #%%
    """
    ### atmospheric transmission and emission
    """
    _log.info('Add sky transmission and emission')
    fname_sky = fdir_sky / f'skytable_airmass={airmass:.2f}.fits'
    wave_min_nm = int(round(wv_t[0] *1e9))
    wave_max_nm = int(round(wv_t[-1]*1e9))
    wdelta_nm   = dwv_t[0]*1e9
    
    # if fname_sky.exists():
    #      sky = fits.getdata(fname_sky)
    # else:
    sky = skycalc.sky_model(observatory='2640', airmass=airmass,
                            pwv_mode='pwv', season=0, time=0, pwv=2.5, msolflux=130.0,
                            incl_moon='N', incl_starlight='Y', incl_zodiacal='N',
                            incl_loweratm='Y', incl_upperatm='Y', incl_airglow='Y',
                            vacair='vac', wmin=wave_min_nm, wmax=wave_max_nm, wdelta=wdelta_nm,
                            wgrid_mode='fixed_wavelength_step', wres=star_wv_res)
    fits.writeto(fname_sky, sky, overwrite=True)
    
    # transmission
    sky_wave = sky['lam'] * u.nm
    sky_trsm = sky['trans']
    
    direct_cube *= sky_trsm[:, None, None]
    corono_cube *= sky_trsm[:, None, None]
    
    #%%
    """
    ### apply telescope and instrumental noises
    """
    _log.info('Add instrumental noises')
    
    direct_cube = direct_cube * tel_transmission * inst_transmission
    corono_cube = corono_cube * tel_transmission * inst_transmission
    
    direct_cube = direct_cube * DIT * area   # phot/s/m2 ==> phot
    corono_cube = corono_cube * DIT * area   # phot/s/m2 ==> phot
    
    if kwd_noi:
        _log.info('Add photon and readout noise')
        # photon noise
        direct_Int_phn = np.random.poisson(lam =direct_cube.value)
        corono_Int_phn = np.random.poisson(lam =corono_cube.value)
        
        # readout noise
        direct_Int_ron = np.random.normal(0.0, std_ron, size=(nlam,nImg2d,nImg2d))
        corono_Int_ron = np.random.normal(0.0, std_ron, size=(nlam,nImg2d,nImg2d))
    
        # total noise
        direct_cube = direct_Int_phn + direct_Int_ron
        corono_cube = corono_Int_phn + corono_Int_ron
                
    else:
        _log.warning(' ==> no noise added!')

    # apply stellar photometry
    direct_cube *= u.dimensionless_unscaled
    corono_cube *= u.dimensionless_unscaled
    
    # rounding of the values
    direct_cube = np.rint(direct_cube.value).astype(int)
    corono_cube = np.rint(corono_cube.value).astype(int)
    
    # clip negative values to zero
    direct_cube = direct_cube.clip(min=0)
    corono_cube = corono_cube.clip(min=0)


    #%% saving of the images
    """
    ### File saving
    """
    _log.info('Save data cubes')
    if do_sav:
        if kwd_sav_onlyphot:
            data_list = [direct_cube, corono_cube]
            fpath_list = [fpath_direct_cube, fpath_corono_cube]
        else:
            data_list = [direct_mono_img_f,corono_mono_img_f,direct_mono_prf_avg_f,
                         corono_mono_prf_avg_f,direct_mono_prf_std_f,corono_mono_prf_std_f,
                         direct_mono_img_fp, corono_mono_img_fp,
                         direct_cube, corono_cube]
            fpath_list = [fpath_direct_mono_img_f,fpath_corono_mono_img_f,fpath_direct_mono_prf_avg_f,
                          fpath_corono_mono_prf_avg_f,fpath_direct_mono_prf_std_f,fpath_corono_mono_prf_std_f,
                          fpath_direct_mono_img_fp, fpath_corono_mono_img_fp,
                          fpath_direct_cube, fpath_corono_cube]
        nlist = len(data_list)
        
        for ilist in range(nlist):#tqdm.tqdm(range(nlist), "file saving"):
        # save in FITS format
            hdu_prim = fits.PrimaryHDU()
            hdu_img  = fits.ImageHDU(data_list[ilist])
            hdu_wave = fits.BinTableHDU.from_columns([
                fits.Column(name='wave', unit='nm', array=wv_t, format='D'),
                fits.Column(name='dwave', unit='nm', array=dwv_t, format='D')
            ])
        
            # set some keywords in primary header
            hdu_prim.header['BAND']     = (band, 'Filter')
            hdu_prim.header['WAVE_MIN'] = (wv_t[0], 'Minimum wavelength [m]')
            hdu_prim.header['WAVE_CEN'] = (wv_t[nlam//2], 'Central wavelength [m]')
            hdu_prim.header['WAVE_MAX'] = (wv_t[-1], 'Maximum wavelength [m]')
            hdu_prim.header['RESOL']    = (wv_R, 'Spectral resolution')
            hdu_prim.header['PIXELSIM'] = (pixel, 'Input simulation pixel size [mas]')
    
            hdu_prim.header['STR_SpT'] = (star_SpT, 'Star spectral type')
            hdu_prim.header['STR_MASS']= (star_mass, 'Star mass [MSun]')
            hdu_prim.header['STR_AGE'] = (star_age, 'Star age [Myr]')
            hdu_prim.header['STR_DIST']= (star_dist, 'Star distance [pc]')
    
            hdu_prim.header['PLT_MASS']= (star_mass, 'Planet mass [MJup]')
            hdu_prim.header['PLT_AGE'] = (star_age, 'Planet age [Myr]')
            hdu_prim.header['PLT_DIST']= (star_dist, 'Planet distance [pc]')
            
            if kwd_noi:
                hdu_prim.header['RON']     = (std_ron, 'Readout noise [e rms]')
                hdu_prim.header['PH_NOISE'] = ('YES', 'Photon noise')
        
            hdu = fits.HDUList([hdu_prim, hdu_img, hdu_wave])
        
            hdu.writeto(fpath_list[ilist], overwrite=True)   
    
    #%%
    t_end = time.time()
    _log.info(f'time usage:   {t_end-t_ini:.2f}s for nlam={nlam:03} and nmap={nmap:05d}')
    _log.info(f'memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f}Mb  for nlam={nlam:03} and nmap={nmap:05d}')