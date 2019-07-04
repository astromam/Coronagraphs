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
import numpy as np
import os
import time
from pyzelda.utils import aperture, imutils, zernike
from pathlib import Path
from astropy.io import fits
import corono as coro
import ctypes
import multiprocessing


#%%
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
               direct_poly_img_cube_i, direct_poly_img_cube_shape_i, corono_poly_img_cube_i, corono_poly_img_cube_shape_i):
    '''
    Thread pool initialization
    '''
    global OPDmap2d0, SAXOmapnm3d, corono0, direct_poly_img_cube, direct_poly_img_cube_shape, corono_poly_img_cube, corono_poly_img_cube_shape

    OPDmap2d0   = OPDmap2d0_i
    SAXOmapnm3d = SAXOmapnm3d_i
    corono0     = corono0_i
    direct_poly_img_cube       = direct_poly_img_cube_i
    direct_poly_img_cube_shape = direct_poly_img_cube_shape_i
    corono_poly_img_cube       = corono_poly_img_cube_i
    corono_poly_img_cube_shape = corono_poly_img_cube_shape_i

    
def compute_corono_image(img_index, saxo_i, saxo_f):
    '''
    Compute a series of coronagraphic images
    '''
    global OPDmap2d0, SAXOmapnm3d, corono0, direct_poly_img_cube, direct_poly_img_cube_shape, corono_poly_img_cube, corono_poly_img_cube_shape

    # shared arrays
    direct_poly_img_cube_np = array_to_numpy(direct_poly_img_cube, direct_poly_img_cube_shape)
    corono_poly_img_cube_np = array_to_numpy(corono_poly_img_cube, corono_poly_img_cube_shape)
    
    # create temporary images
    direct_poly_img_f = np.zeros((nImg2d, nImg2d))
    corono_poly_img_f = np.zeros((nImg2d, nImg2d))

    # loop on phase screens
    saxo_i = int(saxo_i)
    saxo_f = int(saxo_f)
    nmap = saxo_f - saxo_i + 1
    for imap in range(nmap):
        OPDmap2d = OPDmap2d0 + SAXOmapnm3d[saxo_i+imap]*1e-9
        direct_poly_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d)
        corono_poly_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d)

        t_mean = (time.time()-t0) / (imap+1)
        if (imap+1) % 10 == 0:
            print('map {1}/{2}, average computation time: {0:.2f}s'.format(t_mean, imap+1, nmap))

    # save result in shared arrays
    direct_poly_img_cube_np[img_index] = direct_poly_img_f
    corono_poly_img_cube_np[img_index] = corono_poly_img_f


if __name__ == '__main__':
    #%% APLC2d tests
    """
    ### Parameters
    """
    # Coronagraph type
    corono_name   = 'APLC'  # 'SP' or 'APLC' or DZPM
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
    mas2rad   = np.pi/(180.*3600)    # Conversion factor from mas to rads
    rMask_m   = 287e-6/2.            # mask size in m
    rMask     = rMask_m/(wv*Fratio)  # mask size in lam0/D
    rMask_mas = 1000.*rMask * (wv/dAper)/mas2rad
    print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))

    # sampling
    pixel  = 12.25  # IRDIS pixel sampling [mas/pix]
    nPup   = 384    # pupil
    nFPM   = 200    # focal plane mask
    nImg2d = 350    # final image plane 

    # compute spatial frequencies in the final image plane
    loD    = wv/dAper*180/np.pi*3600*1000/pixel
    Fmax2d = nImg2d/loD    # spatial frequencies in the final image plane
    
    # wavelength sampling
    nlam   = 5
    bw     = width/wv 

    # simulation configuration   
    kw_aberr     = True
    kw_2nddate   = bool(eval(sys.argv[1]))
    kw_skyobs    = True     # related to ZELDA map
    kw_aftercorr = bool(eval(sys.argv[2]))
    kw_saxo      = True
    saxomap_i    = 0               # saxo first screen
    saxomap_f    = int(30*1380)    # saxo last screen

    # seeing for on-sky observations
    seeing = 0.8
    
    # multi-processing
    nproc = multiprocessing.cpu_count()//2 - 1
    
    # make sure we have a number of phase screens multiple of the number of CPUs
    nsaxomap  = saxomap_f - saxomap_i + 1
    nsaxomap  = nsaxomap - (nsaxomap % nproc)

    ndefo = 21
    defo_ampl_arr = [np.float(sys.argv[3])]
    # defo_ampl_arr = -100 + 10.*np.arange(21)
    tipp_ampl = 0
    tilt_ampl = 0 

    #%%
    """
    ### Directories
    """    
    if kw_aberr is False:
        str_aberr = 'wo_aberr'
        str_date  = '2018-04-01'
        if kw_skyobs is True:
            str_obs   = 'sky'
        else:
            str_obs   = 'internal'
        str_corr  = ''
        str_saxo  = ''
        str_saxoset = ''
        nmap      = 1 
    else:
        str_aberr = 'with_aberr'    
        str_date  = '2018-04-01'
        str_obs   = 'internal'
        str_corr  = 'before_correction'
        str_saxo  = ''
        imap0     = 0
        nmap      = 1
        # beta_wfs  = 1/0.95
        beta_wfs  = 1/0.80
        str_saxo_tmp  = 'wo_saxo'
        if kw_2nddate is True:
            str_date = '2018-04-03'
        if kw_skyobs is True:
            str_obs  = 'sky'
        if kw_aftercorr is True:
            str_corr = 'after_correction'
            imap0    = 3
            # beta_wfs = 1/0.95
            beta_wfs = 1/0.80
        if kw_saxo is True:
            str_saxo = 'with_saxo'
            nmap     = nsaxomap*1
            # beta_wfs = 1/0.64
            beta_wfs = 1/0.64*1/0.8

    #%%
    # fdir = Path('~/GitHub/Coronagraphs/').expanduser()
    # fdir = Path('~/Work/GitHub/Coronagraphs/').expanduser()
    fdir = Path('/Users/mndiaye/Dropbox/python/Coronagraphs/')
    fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
    fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
    fdir_saxo    = fdir / 'data' / '2D' / 'ZELDA' / str_date

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
    # fname_Apod2d     = 'vlt_APLC_obs=0.14_lsid=0.28_lsod=1.00_IWA=2.0_OWA=20.0_BW=0.20_nlam=05_1D_N=0384_nFPM=50.000000_rMask=2.252MaxContrastL1_tau=0.756_stdgrb.fits'
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

        if kw_saxo is True:
            fname_SAXOmapnm3d = '2018-04-04T00:41:34-saxo_residual_turbulence_time=30.0sec_seeing={:.1f}as_tiptilt=1_gains=0_fitting=1_alias=1.fits'.format(seeing)
    
    #%% Filepaths for the file sources
    fpath_Apod2d          = fdir_pupils / fname_Apod2d
    fpath_Apod2d_OPDmapnm = fdir_pupils / fname_Apod2d_OPDmapnm
    fpath_Ampmap2d        = fdir_zelda / fname_Ampmap2d

    if kw_aberr is True:
        fpath_ZELDAmapnm3d = fdir_zelda  / fname_ZELDAmapnm3d   
        if kw_saxo is True:
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
    #if kw_aberr is True:
    Ampmap2d = fits.getdata(fpath_Ampmap2d)

    #%% Phase errors
    if kw_aberr is True:
        ZELDAmapnm3d = fits.getdata(fpath_ZELDAmapnm3d)

        if kw_saxo is True:
            # SAXO pupils
            pupil_tmp = aperture.sphere_saxo_pupil()
            pupil = np.round(imutils.scale(pupil_tmp, 0, new_dim=(nPup, nPup), method='interp'))

            # read SAXO phase residuals
            SAXOmapnm3d_tmp = fits.getdata(fpath_SAXOmapnm3d)

            # select only phase screens that will be actually used
            SAXOmapnm3d_tmp = SAXOmapnm3d_tmp[saxomap_i:saxomap_f]

            # rescale NCPA map
            print('Rescaling SPARTA phase screens')
            SAXOmapnm3d = np.empty((nmap, nPup, nPup))
            for i in range(nmap):
                SAXOmapnm3d[i] = imutils.scale(SAXOmapnm3d_tmp[i], 0, new_dim=(nPup, nPup), method='interp')
                if (i+1) % 1000 == 0:
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

    params = coro.to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d=Fmax2d, nFPM=nFPM,
                          rMask=rMask,
                          Pupil2dSym=Pupil2dSym, 
                          Pupil2d=Pupil2d, LyotStop2d=LyotStop2d, 
                          CtrBtwnPix=CtrBtwnPix,
                          CtrBtwnPix2=CtrBtwnPix2, 
                          nlam=nlam, bw=bw, wv=wv,
                          OPDmap2d=None, Ampmap2d=None,
                          OPDmap2d_post=None)

    #%%
    """
    ### Filepaths for the file results
    """
    if kw_aberr is True:
        params  = coro.update_params(params, OPDmap2d=None,
                                      Ampmap2d=Ampmap2d, LyotStop2d=LyotStop2d)
        corono0  = coro.design.APLC2d(**params)
    else:
        params  = coro.update_params(params, OPDmap2d=None, 
                                     Ampmap2d=Ampmap2d, LyotStop2d=LyotStop2d)
        corono0 = coro.design.APLC2d(**params)    


    for i, defo_ampl in enumerate(defo_ampl_arr):
        print('defo={0}nm rms'.format(defo_ampl))    
        fname_direct_poly_img_f     = 'direct_poly_img_nmap={:05d}_defo={:.1f}_tip={:.1f}_tilt={:.1f}_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
        fname_corono_poly_img_f     = 'corono_poly_img_nmap={:05d}_defo={:.1f}_tip={:.1f}_tilt={:.1f}_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
        fpath_direct_poly_img_f     = fdir_results / fname_direct_poly_img_f
        fpath_corono_poly_img_f     = fdir_results / fname_corono_poly_img_f

        #%%
        fname_direct_poly_prf_avg_f = 'direct_poly_prf_nmap={:05d}_defo={:.1f}_tip={:.1f}_tilt={:.1f}_avg_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
        fname_corono_poly_prf_avg_f = 'corono_poly_prf_nmap={:05d}_defo={:.1f}_tip={:.1f}_tilt={:.1f}_avg_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
        fname_direct_poly_prf_std_f = 'direct_poly_prf_nmap={:05d}_defo={:.1f}_tip={:.1f}_tilt={:.1f}_std_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
        fname_corono_poly_prf_std_f = 'corono_poly_prf_nmap={:05d}_defo={:.1f}_tip={:.1f}_tilt={:.1f}_std_f.fits'.format(nmap, defo_ampl, tipp_ampl, tilt_ampl)
        fpath_direct_poly_prf_avg_f = fdir_results / fname_direct_poly_prf_avg_f
        fpath_corono_poly_prf_avg_f = fdir_results / fname_corono_poly_prf_avg_f
        fpath_direct_poly_prf_std_f = fdir_results / fname_direct_poly_prf_std_f
        fpath_corono_poly_prf_std_f = fdir_results / fname_corono_poly_prf_std_f

        #%%
        # definition of the coronagraph class
        if kw_aberr is True:
            OPDmap2d0 = (beta_wfs*ZELDAmapnm3d[imap0]+Apod2d_OPDmapnm \
                        + defo_ampl*Defo_mapnm2d \
                        + tipp_ampl*Tipp_mapnm2d \
                        + tilt_ampl*Tilt_mapnm2d)*1e-9

        t0 = time.time()
        if kw_aberr is True:
            if kw_saxo is True:
                # create shared arrays
                direct_poly_img_cube_shape = (nproc, nImg2d, nImg2d)
                direct_poly_img_cube_data  = multiprocessing.RawArray(ctypes.c_double, int(np.prod(direct_poly_img_cube_shape)))
                direct_poly_img_cube_np    = array_to_numpy(direct_poly_img_cube_data, direct_poly_img_cube_shape)
                
                corono_poly_img_cube_shape = (nproc, nImg2d, nImg2d)
                corono_poly_img_cube_data  = multiprocessing.RawArray(ctypes.c_double, int(np.prod(corono_poly_img_cube_shape)))
                corono_poly_img_cube_np    = array_to_numpy(corono_poly_img_cube_data, corono_poly_img_cube_shape)

                # create thread pool
                print('Create thread pool')
                tpool = multiprocessing.Pool(processes=nproc, initializer=tpool_init,
                                             initargs=(OPDmap2d0, SAXOmapnm3d, corono0, direct_poly_img_cube_data, direct_poly_img_cube_shape,
                                                       corono_poly_img_cube_data, corono_poly_img_cube_shape))
                # tpool_init(OPDmap2d0, SAXOmapnm3d, corono0, direct_poly_img_cube_data, direct_poly_img_cube_shape,
                #            corono_poly_img_cube_data, corono_poly_img_cube_shape)

                # create tasks
                print('Create tasks')
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

                direct_poly_img_cube_np = array_to_numpy(direct_poly_img_cube_data, direct_poly_img_cube_shape)
                corono_poly_img_cube_np = array_to_numpy(corono_poly_img_cube_data, corono_poly_img_cube_shape)
                
                direct_poly_img_f += direct_poly_img_cube_np.sum(axis=0)
                corono_poly_img_f += corono_poly_img_cube_np.sum(axis=0)

            else:
                direct_poly_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d0)
                corono_poly_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d, OPDmap2d=OPDmap2d0)
        else:
            direct_poly_img_f += corono0.compute_direct_intensity_2d_bis(Apod2d)
            corono_poly_img_f += corono0.compute_corono_intensity_2d_bis(Apod2d)    

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
        
    print(fpath_corono_poly_prf_std_f)
    print('ok')

# import matplotlib.pyplot as plt
# import matplotlib.colors as colors

# data_psf = fits.getdata('/Users/avigan/data/ZELDA/2018-04-03_night/analysis/2018-04-03_night_aplc_test2_psf_zel_image.fits')
# data_p_avg = fits.getdata('/Users/avigan/data/ZELDA/2018-04-03_night/analysis/2018-04-03_night_aplc_test2_coro_zel_profile_mean.fits')
# data_p_avg = data_p_avg.mean(axis=0) / data_psf.max()

# data_p_std = fits.getdata('/Users/avigan/data/ZELDA/2018-04-03_night/analysis/2018-04-03_night_aplc_test2_coro_zel_profile_std.fits')
# data_p_std = data_p_std.mean(axis=0) / data_psf.max()

# data_sep = np.arange(data_p_avg.size)*12.25

# plt.figure(0, figsize=(21, 7))
# plt.clf()

# plt.subplot(131)
# plt.imshow(direct_poly_img_f, norm=colors.LogNorm(), vmin=1e-6, vmax=1)

# plt.subplot(132)
# plt.imshow(corono_poly_img_f, norm=colors.LogNorm(), vmin=1e-6, vmax=1e-2)

# plt.subplot(133)
# sep = np.arange(corono_poly_prf_avg_f.size)*12.25
# plt.plot(sep, direct_poly_prf_avg_f, label='simu psf', color='C0')
# plt.plot(sep, corono_poly_prf_avg_f, label='simu coro avg', color='C1')
# plt.plot(sep, corono_poly_prf_std_f, label='simu coro std', color='C2')
# plt.plot(data_sep, data_p_avg, label='data coro avg', color='C1', linestyle='--')
# plt.plot(data_sep, data_p_std, label='data coro std', color='C2', linestyle='--')
# plt.xlabel('Separation [mas]')
# plt.xlim(0, 2000)
# plt.ylabel('Contrast')
# plt.ylim(1e-6, 1)
# plt.yscale('log')

# plt.tight_layout()

# plt.legend(loc='upper right')

# plt.show()

# plt.savefig(fdir_results / 'results.pdf')
