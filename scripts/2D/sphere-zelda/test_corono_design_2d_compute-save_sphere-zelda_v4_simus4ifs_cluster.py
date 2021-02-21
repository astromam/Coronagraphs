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
import pwd
import psutil

def array_to_numpy(shared_array, shape):
    '''
    Map a raw memory array to a numpy array
    '''
    if shared_array is None:
        return None

    numpy_array = np.frombuffer(shared_array, dtype=np.float)
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
    for imap in range(nmap):
        OPDmap2d = OPDmap2d0 + SAXOmapnm3d[saxo_i+imap]*1e-9
        direct_mono_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d, poly=False)
        corono_mono_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d, poly=False)

        t_mean = (time.time()-t0) / (imap+1)
        if (imap+1) % 10 == 0:
            print('map {1}/{2}, average computation time: {0:.2f}s'.format(t_mean, imap+1, nmap))

    # save result in shared arrays
    direct_mono_img_cube_np[img_index] = direct_mono_img_f
    corono_mono_img_cube_np[img_index] = corono_mono_img_f


if __name__ == '__main__':
    
    t_ini = time.time()
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
    wv0       = 1625e-9 #1.593e-6
    width     = 290e-9  #52e-9

    # Telescope characteristics
    dAper     = 8
    Fratio    = 40

    # Focal plane mask 
    mas2rad   = np.pi/(180.*3600) # Conversion factor from mas to rads
    rMask_m   = 287e-6/2.         # mask size in m
    rMask     = rMask_m/(wv0*Fratio)  # mask size in lam0/D
    rMask_mas = 1000.*rMask * (wv0/dAper)/mas2rad
    print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv0*1e6))

    # sampling
    nPup   = 384   # pupil
    nFPM   = 200   # focal plane mask
    nImg2d = 200   # final image plane 

    # compute spatial frequencies in the final image plane
    pixel  = 12.25 # IRDIS pixel sampling [mas/pix]
    loD    = wv0/dAper*180/np.pi*3600*1000/pixel
    nFre2d = nImg2d/loD    # spatial frequencies in the final image plane
    
    # wavelength sampling
    nlam   = 892
    bw     = width/wv0 

    lam0   = 1. 
    dlam   = bw*lam0
    lam_t  = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam)
    wv_t   = wv0*lam_t


    # simulation configuration   
    kw_aberr     = True
    kw_2nddate   = True    
    kw_skyobs    = True
    kw_aftercorr = False
    kw_saxo      = True
    saxofudge    = 1 #60/120              # saxo amplitude errors fudge factor
    saxomap_i    = int(450*(eval(sys.argv[1])))                    # saxo first screen
    saxomap_f    = int(450*(eval(sys.argv[1])+1))     # saxo last screen

    # multi-processing
    nproc = multiprocessing.cpu_count()//2 - 1
    
    # make sure we have a number of phase screens multiple of the number of CPUs
    nsaxomap  = saxomap_f - saxomap_i + 1
    nsaxomap  = nsaxomap - (nsaxomap % nproc)

    ndefo = 21
    defo_ampl = 0#-100 + 10.*np.arange(21)
    tipp_ampl = 0
    tilt_ampl = 0 

    #%%
    """
    ### Directories
    """    
    if kw_aberr is False:
        str_aberr = 'wo_aberr'
        str_date  = ''
        if kw_skyobs:
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
        if kw_2nddate:
            str_date = '2018-04-03'
        if kw_skyobs:
            str_obs  = 'sky'
        if kw_aftercorr:
            str_corr = 'after_correction'
            imap0    = 3
            beta_wfs = 1/0.95
        if kw_saxo and kw_2nddate:
            str_saxo = 'with_saxo'
            nmap     = nsaxomap*1
            beta_wfs = 1/0.6

    #%%
#    fdir = Path('~/data/ZELDA/CoroSimulations/').expanduser()
    user = pwd.getpwuid(os.getuid())[0]
    syst = sys.platform
    
    if user == 'mndiaye':
        if syst == 'darwin':
            fdir = Path('~/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs').expanduser()
        elif syst == 'linux':
            fdir = Path('/scratch/{0}/data/Coronagraphs/'.format(user)).resolve()
        else:
            raise ValueError('Unknown operating system {0}'.format(user))
    else:
        raise ValueError('Unknown user {0}'.format(user))
    

    fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
    fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
    fdir_saxo    = fdir / 'data' / '2D' / 'ZELDA' / '2018-04-03'

    if kw_aberr:
        fdir_res = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_date / str_obs / str_saxo / str_corr  
    else:
        fdir_res = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_obs / str_saxo / str_corr  

    if not os.path.exists(fdir_res):
        os.makedirs(fdir_res)

    #%%
    """
    ### Filenames for the sources
    """
    fname_Apod2d     = 'SPHERE_APO1_field_transmission_map.fits'
    fname_Apod2d_OPDmapnm = 'apo_substrate_D1.fits'
    fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
    fname_LyotStop2d = 'sphere_stop_ST_ALC2.fits'

    if kw_aberr:
        if kw_skyobs:
            fname_ZELDAmapnm3d = '2018-04-01_night_ncpa_loop_700modes_5_ncpa_loop_opd.fits'
            if kw_2nddate:
                fname_ZELDAmapnm3d = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
        else:
            fname_ZELDAmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'        
            if kw_2nddate:        
                fname_ZELDAmapnm3d = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'

        if kw_saxo and kw_2nddate:
            fname_SAXOmapnm3d = '2018-04-04T03_06_15-saxo_residual_turbulence.fits'

    #%% Filepaths for the file sources
    fpath_Apod2d          = fdir_pupils / fname_Apod2d
    fpath_Apod2d_OPDmapnm = fdir_pupils / fname_Apod2d_OPDmapnm
    fpath_Ampmap2d        = fdir_pupils / fname_Ampmap2d

    if kw_aberr:
        fpath_ZELDAmapnm3d = fdir_zelda  / fname_ZELDAmapnm3d   
        if kw_saxo and kw_2nddate:
            fpath_SAXOmapnm3d = fdir_saxo / fname_SAXOmapnm3d

    fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d    

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
            # SAXO pupils
            pupil_tmp = aperture.sphere_saxo_pupil()
            pupil = np.round(imutils.scale(pupil_tmp, 0, new_dim=(nPup,nPup), method='interp'))

            # read SAXO phase residuals
            SAXOmapnm3d_tmp = fits.getdata(fpath_SAXOmapnm3d)

            # select only phase screens that will be actually used
            SAXOmapnm3d_tmp = SAXOmapnm3d_tmp[saxomap_i:saxomap_f]

            # apply SAXO performance fudge factor
            if saxofudge != 1:
                SAXOmapnm3d_tmp *= saxofudge
            
            # rescale NCPA map
            SAXOmapnm3d = np.empty((nmap, nPup, nPup))
            for i in range(nmap):
                SAXOmapnm3d[i] = imutils.scale(SAXOmapnm3d_tmp[i], 0, new_dim=(nPup,nPup), method='interp')
                print('{0:05}/{1:05}: SAXO map before scaling: {2:.2f} nm RMS, after: {3:.2f} nm RMS'.format(i+1, nmap, np.std(SAXOmapnm3d_tmp[i, pupil_tmp != 0]), np.std(np.asarray(SAXOmapnm3d)[i, pupil != 0])))

            del SAXOmapnm3d_tmp
            
    #%% Lyot Stop
    LyotStop2d = fits.getdata(fpath_LyotStop2d)

    #%%
    Defo_mapnm2d = zernike.zernike1(4, npix=nPup, outside=0.)
    Tipp_mapnm2d = zernike.zernike1(2, npix=nPup, outside=0.)
    Tilt_mapnm2d = zernike.zernike1(3, npix=nPup, outside=0.)

    #%%
    """
    ### Array Initialization
    """
    # define the averaged image
    direct_mono_img_f = np.zeros((nlam, nImg2d, nImg2d))
    corono_mono_img_f = np.zeros((nlam, nImg2d, nImg2d))
    
    # define the averaged and standard deviation profiles of the images
    direct_mono_prf_avg_f = np.zeros((nlam, nImg2d//2))
    corono_mono_prf_avg_f = np.zeros((nlam, nImg2d//2))
    direct_mono_prf_std_f = np.zeros((nlam, nImg2d//2))
    corono_mono_prf_std_f = np.zeros((nlam, nImg2d//2))

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

    str_common = '_mono_nmap={:05d}_saxofudge={:.2f}_nlam={:04d}_saxomap_i={:05d}_f={:05d}'.format(nmap, saxofudge, nlam, saxomap_i, saxomap_f)
    
    # filepaths for the images
    fname_direct_mono_img_f     = 'direct' + str_common + '_img_f.fits'
    fname_corono_mono_img_f     = 'corono' + str_common + '_img_f.fits'
    fpath_direct_mono_img_f     = fdir_res / fname_direct_mono_img_f
    fpath_corono_mono_img_f     = fdir_res / fname_corono_mono_img_f

    # filepaths for the profiles
    fname_direct_mono_prf_avg_f = 'direct' + str_common + '_prf_avg_f.fits'
    fname_corono_mono_prf_avg_f = 'corono' + str_common + '_prf_avg_f.fits'
    fname_direct_mono_prf_std_f = 'direct' + str_common + '_prf_std_f.fits'
    fname_corono_mono_prf_std_f = 'corono' + str_common + '_prf_std_f.fits'
    fpath_direct_mono_prf_avg_f = fdir_res / fname_direct_mono_prf_avg_f
    fpath_corono_mono_prf_avg_f = fdir_res / fname_corono_mono_prf_avg_f
    fpath_direct_mono_prf_std_f = fdir_res / fname_direct_mono_prf_std_f
    fpath_corono_mono_prf_std_f = fdir_res / fname_corono_mono_prf_std_f

    #%%
    # definition of the coronagraph class
    if kw_aberr:
        OPDmap2d0 = (beta_wfs*ZELDAmapnm3d[imap0]+Apod2d_OPDmapnm\
                    +defo_ampl*Defo_mapnm2d\
                    +tipp_ampl*Tipp_mapnm2d\
                    +tilt_ampl*Tilt_mapnm2d)*1e-9

    t0 = time.time()
    if kw_aberr:
        if kw_saxo and kw_2nddate:

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

        else:
            direct_mono_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d0, poly=False)
            corono_mono_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d0, poly=False)
    else:
        direct_mono_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d)
        corono_mono_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d)    

    # computation of the averaged images
    direct_mono_img_f /= nmap
    corono_mono_img_f /= nmap

    for ilam in range(nlam):
        # image normalization
        direct_peak_val = direct_mono_img_f[ilam].max()
        direct_mono_img_f[ilam] /= direct_peak_val
        corono_mono_img_f[ilam] /= direct_peak_val

        # computation of the averaged and standard deviation profiles of the images   
        direct_mono_prf_avg_f[ilam], rad_direct = imutils.profile(direct_mono_img_f[ilam], type='mean')
        corono_mono_prf_avg_f[ilam], rad_corono = imutils.profile(corono_mono_img_f[ilam], type='mean')
        direct_mono_prf_std_f[ilam], rad_direct = imutils.profile(direct_mono_img_f[ilam], type='std')
        corono_mono_prf_std_f[ilam], rad_corono = imutils.profile(corono_mono_img_f[ilam], type='std')

    #%% saving of the images
    """
    ### File saving
    """
    fits.writeto(fpath_direct_mono_img_f, direct_mono_img_f, overwrite=True)
    fits.writeto(fpath_corono_mono_img_f, corono_mono_img_f, overwrite=True)

    fits.writeto(fpath_direct_mono_prf_avg_f, direct_mono_prf_avg_f, overwrite=True)
    fits.writeto(fpath_corono_mono_prf_avg_f, corono_mono_prf_avg_f, overwrite=True)
    fits.writeto(fpath_direct_mono_prf_std_f, direct_mono_prf_std_f, overwrite=True)
    fits.writeto(fpath_corono_mono_prf_std_f, corono_mono_prf_std_f, overwrite=True)

    #%%
    t_end = time.time()
    print('\ntime usage:   {0:.2f}s for nlam={1:03} and nmap={2:05d}'.format(t_end-t_ini,nlam,nmap))
    print('\nmemory usage: {0:.2f}Mb  for nlam={1:03} and nmap={2:05d}'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,nlam,nmap))