#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on nov 2026

@author: mndiaye, asimmonnin, asp
"""

# explore coronagraph parameters ranges and compute throughput, integrated 
# energy of the star and planet psf in the field, with or without coro, at 
# all wavelengths in Y,J and H bands for a given spaxel size

#%%
"""
### Initialization
"""

import multiprocessing as mp
from multiprocessing import shared_memory
# from multiprocessing import cpu_count

import hcipy as hci
from astropy.io import fits
import numpy as np

import os
from pathlib import Path
from datetime import datetime

# donow = datetime.now().strftime("%Y%m%d%H%M%S")
# print(donow)

rad2mas = np.pi/(180.*3600.*1000.)
mas2rad = 1./rad2mas

def compute_one_lambda_shared_full(args):

    (i, lam, shm_results_name, shape_results, dtype_results,
     shm_coro_thr_name, shape_coro_thr, dtype_coro_thr,
     shm_lyot_int_name, shape_lyot_int, dtype_lyot_int,
     shm_eelt_int_name, shape_eelt_int, dtype_eelt_int,
     shm_eta_tel_name, shape_eta_tel, dtype_eta_tel,
     shm_lyot_max_name, shape_lyot_max, dtype_lyot_max,
     telescope_pupil, pupil_grid, D, focal_length,
     ap_lst, nA, dL_lst, nO, obs_lst, nI, ch_lst, nC,
     sf_x, aee, rng, nPup, nImg, nFPM, lam_ref, pscale, rad2mas) = args

    # --- reconnect shared arrays ---

    shm_results = shared_memory.SharedMemory(name=shm_results_name)
    results = np.ndarray(shape_results, dtype=dtype_results,
                         buffer=shm_results.buf)


    shm_coro_thr = shared_memory.SharedMemory(name=shm_coro_thr_name)
    coro_thr = np.ndarray(shape_coro_thr, dtype=dtype_coro_thr,
                          buffer=shm_coro_thr.buf)


    shm_lyot_int = shared_memory.SharedMemory(name=shm_lyot_int_name)
    lyot_int = np.ndarray(shape_lyot_int, dtype=dtype_lyot_int,
                          buffer=shm_lyot_int.buf)


    shm_eelt_int = shared_memory.SharedMemory(name=shm_eelt_int_name)
    eelt_int = np.ndarray(shape_eelt_int, dtype=dtype_eelt_int,
                          buffer=shm_eelt_int.buf)


    shm_eta_tel = shared_memory.SharedMemory(name=shm_eta_tel_name)
    eta_tel = np.ndarray(shape_eta_tel, dtype=dtype_eta_tel,
                         buffer=shm_eta_tel.buf)

    shm_lyot_max = shared_memory.SharedMemory(name=shm_lyot_max_name)
    lyot_max = np.ndarray(shape_lyot_max, dtype=dtype_lyot_max,
                         buffer=shm_lyot_max.buf)

    # throughput over field    
    int_a = telescope_pupil**2 # np.abs(Fld_AA0)**2
    ee_a = int_a.sum() # np.sum(int_a)

    for Ld in range(nO):
        
        diam = dL_lst[Ld]
        
        lyot_diameter = D * diam
        
        fov = nImg * pscale * rad2mas / (lam_ref * 2. / lyot_diameter)
        # in lambda reference over lyot_diameter
    
        # pixels witdh, lambda of reference / lyot_diameter
        px2resEl = nImg / ( fov * 2. )
        # print("dpw2resEl :", px2resEl - lam_ref / (lyot_diameter * pscale * rad2mas))


        #%%
        '''
        focal plane
        '''
        # focal plane grid
        focal_grid = hci.make_focal_grid(q=px2resEl,
                                         num_airy=fov,
                                         pupil_diameter=D,
                                         focal_length=focal_length,
                                         reference_wavelength=lam_ref)
                
        
        #%%
        '''
        propagator
        '''
        f_prop = hci.FraunhoferPropagator(pupil_grid, focal_grid,
                                       focal_length=focal_length)

        # energy/throughput at fourier plane in fov and spaxel
        wf_elt = hci.Wavefront(telescope_pupil, lam)
        Fld_elt = f_prop.forward(wf_elt)
        Int_elt = Fld_elt.intensity
        no_coro_norm = Int_elt.sum() # np.sum(np.abs(Int_elt)**2.)

        for ob in range(nI):
            
            obst=obs_lst[ob]

            # Lyot stop
            lyot_stop = hci.make_obstructed_circular_aperture(
                D * diam, obst/diam,
                num_spiders=6, spider_width=0.01)(pupil_grid)

            lyot_stop *= hci.make_linear_interpolator(
                telescope_pupil,fill_value=0)(pupil_grid)
            
            # # PERFECT PSF: elt + Lyot stop propagation, no FPM
            Fld_AA0 = telescope_pupil * lyot_stop
            wf00 = hci.Wavefront(Fld_AA0, lam)
            Fld_DD0 = f_prop.forward(wf00)
            # Int_DD0 = Fld_DD0.intensity
            
            int_c = Fld_AA0**2 # np.abs(Fld_CC0)**2
            ee_c = int_c.sum() # np.sum(int_c)
            coro_thr[Ld,ob,i] = ee_c/ee_a

            lyot_max [Ld,ob,i] = Fld_DD0.intensity.max()
            lyot_int[Ld,ob,i] = Fld_DD0.intensity.sum()
            eelt_int[Ld,ob,i] = no_coro_norm
            eta_tel[Ld,ob,i] = np.sum(np.reshape(Int_elt,(nImg,nImg))[aee[0]])

            norm_peakDD0 = 1./no_coro_norm

            for s in range(nC) :
                
                charge = ch_lst[s]

                # Int_DD0 *= norm_peakDD0
                
                # vortex definition
                v_coro = hci.VortexCoronagraph(pupil_grid, charge)
                
                lyot_apod = hci.Apodizer(lyot_stop)
                
                Fld_AA = telescope_pupil
                wf11 = hci.Wavefront(Fld_AA, lam)
                Fld_LL = v_coro.forward(wf11)
                Fld_DD = f_prop.forward(lyot_apod(Fld_LL))
                Int_DD = Fld_DD.intensity
                Int_DD *= norm_peakDD0
                
                Fld_AA = telescope_pupil
                wf11 = hci.Wavefront(Fld_AA, lam)
                Fld_LL = v_coro.forward(wf11)
                Fld_DD = f_prop.forward(Fld_LL)
                Int_DD = Fld_DD.intensity
                Int_DD *= norm_peakDD0

                for a in range(nA):
                    
                    results[0,a,Ld,ob,s,i] = np.sum(np.reshape(
                        Int_DD,(nImg,nImg))[rng[a]])
                
                for a in range(nA):
                    
                    phase = (hci.field.NewStyleField(np.exp(
                        2j * np.pi * ap_lst[a] * rad2mas * 
                        D * diam * sf_x / ( nPup * lam)).ravel(),
                        pupil_grid))

                    Fld_AA = (telescope_pupil * phase)
                    wf1 = hci.Wavefront(Fld_AA, lam)
                    Fld_LL = v_coro.forward(wf1)
                    Fld_DD = f_prop.forward(lyot_apod(Fld_LL))
                    Int_DD = Fld_DD.intensity
                                
                    # Normalized intensity
                    Int_DD *= norm_peakDD0
                    
                    results[1,a,Ld,ob,s,i] = np.sum(np.reshape(
                        Int_DD,(nImg,nImg))[aee[a+1]])
                
    
    shm_results.close()
    shm_coro_thr.close()
    shm_lyot_int.close()
    shm_eelt_int.close()
    shm_eta_tel.close()
    shm_lyot_max.close()
    shm_results.unlink()
    shm_coro_thr.unlink()
    shm_lyot_int.unlink()
    shm_eelt_int.unlink()
    shm_eta_tel.unlink()
    shm_lyot_max.unlink()
    
    return


if __name__ == '__main__':
    
    
    # datetime of script execution
    donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now
    print('date of now : ', donow)

    # defaults parameters
    nFPM = 100  # Sampling of the coronagraph focal plane mask
    nImg = 400  # last focal plane image size
    
    wvl = 'YJHK'
    lam_ref = 2200e-9 # 1600e-9  # 2200e-9 reference wvl
    lam_min = 950e-9  # minimal value of wvl
    lam_itv = 31  # 18  # # of intervals
    lam_stp = 50e-9  # interval step
    
    pscale = 0.3  # image plate scale in last focal plane in mas / pixel
    focal_length = 1e-6 / (pscale * rad2mas)

    # nominal (=inner/inscribed?) 38.542/ hcipy outer/circumscribed 39.14634
    # D = 39.14634 # in m, __use hcipy outer value__
    D = 38.54  #  main pupil diameter/ elt pupil diameter

    fnm_eltp = 'ELT_pupil_400.fits' # New pupil with new spider
    
    """
    ### Working directory
    """
    user = 'Alain'
    if user == 'Alain':
        fdir_dat = Path("D:/Andes/Data_corono/data/").resolve()
        fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()
        fdir_res = (fdir_res / donow)
        os.makedirs(fdir_res, exist_ok=True)   
    
    
    #%%
    '''
    wavelenghts, field of view, pixel to resolution element        
    '''

    # lam_lst = np.arange(lam_min,lam_min+(lam_itv+0.5)*lam_stp,lam_stp)
    lam_lst = np.arange(lam_min,lam_min+lam_itv*lam_stp+1e-9,lam_stp)
    # lam_lst = [950e-9, 1200e-9, 1350e-9, 1600e-9, 1850e-9]
    nL = len(lam_lst)

    #%%
    """
    apertures for photometry
    """
    
    # main angular separation of interest ('asoi'), 20 mas as of 05/2025
    as_oi = 30 
    # spaxel diameter as an aperture and in field locations
    
    ap_mas = 10.  # spaxel of 7 mas (05/2025->) / 10. -> 03/2025
    ap_min = 15
    ap_stp = 1
    ap_itv = 24
    ap_lst = np.arange(ap_min,ap_min+ap_stp*ap_itv+0.5,ap_stp)
    nA = len(ap_lst)
    
    
    #%%
    """
    Lyot coronagraph configuration
    """
    
    ### All Lyot coro parameters to test
    dL_min = 0.8
    dL_itv = 19
    dL_stp = 0.01
    dL_lst = np.arange(dL_min,dL_min+dL_stp*(dL_itv+0.5),dL_stp)
    nO = len(dL_lst)
    
    obs_min = 0.25
    obs_itv = 19
    obs_stp = 0.01
    obs_lst = np.arange(obs_min,obs_min+obs_stp*(obs_itv+0.5),obs_stp)
    nI = len(obs_lst)
    
    ch_min = 2.
    ch_itv = 3
    ch_stp = 2
    ch_lst = np.arange(ch_min,ch_min+ch_stp*(ch_itv+0.5),ch_stp)
    nC = len(ch_lst)
    
    
    #%%
    """
    pupil, pupil_grid, telescope_pupil field
    """
    # Directory for the pupils
    fdir_pupil = fdir_dat / 'Pupil'
    
    # Filename and path for the ELT pupil
    elt_pup_fnm = 'ELT_pupil_400.fits' # 2024
    fpath_elt = fdir_pupil / elt_pup_fnm
    
    # Read ELT pupil 
    pupil = hci.util.read_fits(fdir_pupil / fnm_eltp, extension=1)
    nPup = pupil.shape[0] # supersede input parameter

    pupil_grid = hci.make_pupil_grid(nPup, diameter=D)

    # ELT pupil as a Field
    telescope_pupil = hci.Field(pupil.ravel(),pupil_grid)
    
    # Pupil size
    nPup = pupil.shape[0] # 400    
    
    
    # stackoveflow...
    a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
    b_ = a_.copy()
    aa, bb = np.meshgrid(a_, b_)
    rad_mas = np.zeros((nImg,nImg,nA+1))
    rad_mas[:,:,0] = np.sqrt(aa**2 + bb**2) * pscale
    for a in range(nA):
        rad_mas[:,:,a+1] = np.sqrt(
            (aa * pscale - ap_lst[a])**2 + (bb * pscale)**2)
    
    #aee : aperture encircled energy
    aee = {a:np.where(rad_mas[:,:,a]<=ap_mas/2.) for a in range(nA+1)}
    rng = {a:np.where(
        np.abs(rad_mas[:,:,0]-ap_lst[a]*1.)<=ap_mas/2.) for a in range(nA)}
    
    
    ### results array: star/planet, #locations, #Ld, #obs, #fpm, #wvl
    results = np.zeros((2, nA, nO, nI, nC, nL))
    # planet_thr = np.zeros((nO, nI, nL))
    coro_thr = np.zeros((nO, nI, nL))
    lyot_int = np.zeros((nO, nI, nL))
    lyot_max = np.zeros((nO, nI, nL))
    eelt_int = np.zeros((nO, nI, nL))
    eta_tel = np.zeros((nO, nI, nL))
    
    # intensities = np.zeros((2, nO, nI, nB, nImg//2, nL))


    #%%
    
    # 2D array pupil slope for tilt
    sf_x = np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5
    
    # conversion lradian to mas
    rad2mas = np.pi/(180.*3600.*1000.)
    mas2rad = 1./rad2mas


#%%
    
    ### Loop over the parameters
    # for i in np.arange(nL):
    
    mp.set_start_method("spawn", force=True)

    # --- shared arrays ---

    shm_results = shared_memory.SharedMemory(create=True,
                                         size=results.nbytes)

    shm_results_arr = np.ndarray(results.shape,
                             dtype=results.dtype,
                             buffer=shm_results.buf)

    shm_results_arr[:] = results[:]


    shm_coro_thr = shared_memory.SharedMemory(create=True,
                                         size=coro_thr.nbytes)

    shm_coro_thr_arr = np.ndarray(coro_thr.shape,
                             dtype=coro_thr.dtype,
                             buffer=shm_coro_thr.buf)

    shm_coro_thr_arr[:] = coro_thr[:]


    shm_lyot_int = shared_memory.SharedMemory(create=True,
                                         size=lyot_int.nbytes)

    shm_lyot_int_arr = np.ndarray(lyot_int.shape,
                             dtype=lyot_int.dtype,
                             buffer=shm_lyot_int.buf)

    shm_lyot_int_arr[:] = lyot_int[:]


    shm_eelt_int = shared_memory.SharedMemory(create=True,
                                         size=eelt_int.nbytes)

    shm_eelt_int_arr = np.ndarray(eelt_int.shape,
                             dtype=eelt_int.dtype,
                             buffer=shm_eelt_int.buf)

    shm_eelt_int_arr[:] = eelt_int[:]

    shm_eta_tel = shared_memory.SharedMemory(create=True,
                                         size=eta_tel.nbytes)

    shm_eta_tel_arr = np.ndarray(eta_tel.shape,
                             dtype=eta_tel.dtype,
                             buffer=shm_eta_tel.buf)

    shm_eta_tel_arr[:] = eta_tel[:]


    shm_lyot_max = shared_memory.SharedMemory(create=True,
                                         size=lyot_max.nbytes)

    shm_lyot_max_arr = np.ndarray(lyot_max.shape,
                             dtype=lyot_max.dtype,
                             buffer=shm_lyot_max.buf)

    shm_lyot_max_arr[:] = lyot_max[:]
    
    
    # nproc = min(cpu_count(), nL)

    args_list = [
        (i, lam_lst[i],
         shm_results.name, results.shape, results.dtype,
         shm_coro_thr.name, coro_thr.shape, coro_thr.dtype,
         shm_lyot_int.name, lyot_int.shape, lyot_int.dtype,
         shm_eelt_int.name, eelt_int.shape, eelt_int.dtype,
         shm_eta_tel.name, eta_tel.shape, eta_tel.dtype,
         shm_lyot_max.name, lyot_max.shape, lyot_max.dtype,
         telescope_pupil, pupil_grid, D, focal_length,
         ap_lst, nA, dL_lst, nO, obs_lst, nI, ch_lst, nC,
         sf_x, aee, rng, nPup, nImg, nFPM, lam_ref, pscale, rad2mas)
        for i in range(nL)
    ]

    with mp.Pool() as pool:
        
        # res = pool.map(compute_one_lambda_shared_full, args_list)
        res = pool.imap_unordered(compute_one_lambda_shared_full, args_list)
        pool.close()
        pool.join()
    

    #%%
    wvl_as_spxl = (wvl+'_angSep_'+
                   str(int(as_oi))+'mas_spxl_'+
                   str(int(ap_mas))+'mas_as_'+
                   str(int(lam_ref*1e9))+'nm.fits')
    fpath_psf_lst=(
        # fdir_res / ('intensities_vs_params_'+wvl_as_spxl),
        fdir_res / ('results_vs_params_'+wvl_as_spxl),
        # fdir_res / ('Parameters_planet_thr_'+wvl_as_spxl),
        fdir_res / ('Parameters_coro_thr_'+wvl_as_spxl),
        fdir_res / ('eelt_intensities_vs_params_'+wvl_as_spxl),
        fdir_res / ('lyot_intensities_vs_params_'+wvl_as_spxl),
        fdir_res / ('eta_tel_vs_params_'+wvl_as_spxl),
        fdir_res / ('lyot_max_intensity_vs_params_'+wvl_as_spxl))
    
    # fits.writeto(fpath_psf_lst[0],intensities, overwrite=True)
    fits.writeto(fpath_psf_lst[0],shm_results_arr, overwrite=True)
    # fits.writeto(fpath_psf_lst[2],np.array(planet_thr), overwrite=True)
    fits.writeto(fpath_psf_lst[1],np.array(shm_coro_thr_arr), overwrite=True)
    fits.writeto(fpath_psf_lst[2],np.array(shm_eelt_int_arr), overwrite=True)
    fits.writeto(fpath_psf_lst[3],np.array(shm_lyot_int_arr), overwrite=True)
    fits.writeto(fpath_psf_lst[4],np.array(shm_eta_tel_arr), overwrite=True)
    fits.writeto(fpath_psf_lst[5],np.array(shm_lyot_max_arr), overwrite=True)

    for fpath in (fpath_psf_lst):
        
        fits.setval(fpath,'NPUP',value=nPup,comment='pupil size')
        fits.setval(fpath,'NFPM',value=nFPM,comment='FP coro. sampling')
        fits.setval(fpath,'NIMG',value=nImg,comment='image size')
        fits.setval(fpath,'LMIN',value=lam_min,comment='wavelength in meters')
        fits.setval(fpath,'LITV',value=lam_itv,comment='# of wvl intervals')
        fits.setval(fpath,'LSTP',value=lam_stp,comment='wvl step in meters')
        fits.setval(fpath,'LMBD',value=lam_ref,comment='wavelength in meters')
        fits.setval(fpath,'DIAM',value=D,comment='pupil dimater in meters')
        fits.setval(fpath,'PSCL',value=pscale,comment='plate scale in mas')
        fits.setval(fpath,'VTXC_MIN',value=ch_min,
                    comment='minim. FPM (lam/D), first focal plane')
        fits.setval(fpath,'VTXC_ITV',value=ch_itv,
                    comment='# of FPM intervals (lam/D), first focal plane')
        fits.setval(fpath,'VTXC_STP',value=ch_stp,
                    comment='step FPM (lam/D), first focal plane')
        fits.setval(fpath,'OBST_MIN',value=obs_min,
                    comment='minim. fractional pup. diameter')
        fits.setval(fpath,'OBST_ITV',value=obs_itv,
                    comment='# of fractional obscuration')
        fits.setval(fpath,'OBST_STP',value=obs_stp,
                    comment='step fractional obscuration')
        fits.setval(fpath,'DLYO_MIN',value=dL_min,
                    comment='minim. fractional Lyot pupil diameter')
        fits.setval(fpath,'DLYO_ITV',value=dL_itv,
                    comment='# of fractional Lyot pupil diameter')
        fits.setval(fpath,'DLYO_STP',value=dL_stp,
                    comment='step fractional Lyot pupil diameter')
        fits.setval(fpath,'D_SPAXEL',value=ap_mas,
                    comment='spaxel diameter in mas')
        fits.setval(fpath,'SPXL_ROI',value=as_oi,
                    comment='main spaxel location of interest in mas')
        fits.setval(fpath,'SPXL_MIN',value=ap_min,
                    comment='min. spaxel diameter in mas')
        fits.setval(fpath,'SPXl_ITV',value=ap_itv,
                    comment='# of spaxels intervals')
        fits.setval(fpath,'SPXL_STP',value=ap_stp,
                    comment='step between two spaxel locations')
        fits.setval(fpath,'EPUP_FNM',value=elt_pup_fnm,
                    comment='ELT pupil filename')
        
    shm_results.close()
    shm_results.unlink()
    shm_coro_thr.close()
    shm_coro_thr.unlink()
    shm_lyot_int.close()
    shm_lyot_int.unlink()
    shm_eelt_int.close()
    shm_eelt_int.unlink()
    shm_eta_tel.close()
    shm_eta_tel.unlink()
    shm_lyot_max.close()
    shm_lyot_max.unlink()

    print(donow, as_oi, ap_mas)

