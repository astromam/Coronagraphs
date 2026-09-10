#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import cpu_count

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from astropy.io import fits

import os
# import sys
from pathlib import Path
from datetime import datetime

from slow_fourier_transform import sft, isft
from uniform_disk import uniform_disk
from psf_profile import radial_profile

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!
    
os.environ["OMP_NUM_THREADS"] = "1"

def compute_one_lambda_shared_full(args):

    (i, lam,
     shm_opd_name, shape_opd, dtype_opd,
     shm_pupil_name, shape_pupil, dtype_pupil,
     shm_lyot_name, shape_lyot, dtype_lyot,
     mask2d, fpm_dec, dfc, sf_y,
     D, diam, nPup,
     nImg, nFPM, mB, lam_ref,
     mD_ref, disp, rad2mas) = args

    # --- reconnect shared arrays ---
    shm_opd = shared_memory.SharedMemory(name=shm_opd_name)
    OPD_arr = np.ndarray(shape_opd, dtype=dtype_opd, buffer=shm_opd.buf)

    shm_pupil = shared_memory.SharedMemory(name=shm_pupil_name)
    Pupil = np.ndarray(shape_pupil, dtype=dtype_pupil, buffer=shm_pupil.buf)

    shm_lyot = shared_memory.SharedMemory(name=shm_lyot_name)
    LyotStop2d = np.ndarray(shape_lyot, dtype=dtype_lyot, buffer=shm_lyot.buf)

    dLam = lam - lam_ref
    tilt = dLam * disp * rad2mas
    mD = mD_ref * lam_ref / lam

    nOPD = OPD_arr.shape[0]

    Int_D0 = np.zeros((nImg, nImg))
    Int_D  = np.zeros((nImg, nImg))
    Int_D0_prf = np.zeros((2, nImg//2))
    Int_D_prf = np.zeros((2, nImg//2))
    Int_DD0 = np.zeros((nImg, nImg))
    Int_DD = np.zeros((nImg, nImg))
    
    # PERFECT PSF
    Fld_AA0 = Pupil * LyotStop2d
    Fld_DD0 = sft(Fld_AA0, nImg, mD*diam)
    Int_DD0 = np.abs(Fld_DD0)**2
    norm_peakDD0 = 1/np.max(Int_DD0)
    Int_DD0 *= norm_peakDD0

    # perfect coronographic image
    
    Fld_AA = Pupil * 1.
    Fld_BB = mask2d*sft(Fld_AA, nFPM, mB*lam_ref/lam)
    Fld_CC = Fld_AA - isft(Fld_BB, nPup, mB*lam_ref/lam)
    Fld_LL = Fld_CC * LyotStop2d
    Fld_DD = sft(Fld_LL, nImg, mD*diam)
    Int_DD = np.abs(Fld_DD)**2
    Int_DD *= norm_peakDD0

    # LOOP OPD
    for k in range(nOPD):

        phase = (OPD_arr[k,:,:] + (fpm_dec + tilt) * D * diam * sf_y / nPup + dfc)

        Fld_A0 = (Pupil * np.exp(1j*2*np.pi * phase/lam) * LyotStop2d)

        Fld_D0 = sft(Fld_A0, nImg, mD*diam)
        Int_D0 += np.abs(Fld_D0)**2

        # CORO
        Fld_A0 = (Pupil * np.exp(1j * 2 * np.pi * phase / lam))

        Fld_B = mask2d*sft(Fld_A0, nFPM, mB*lam_ref/lam)
        Fld_C = Fld_A0 - sft(Fld_B, nPup, mB*lam_ref/lam, inv=True)
        Fld_L = Fld_C * LyotStop2d
        Fld_D = sft(Fld_L, nImg, mD*diam)

        Int_D += np.abs(Fld_D)**2

    Int_D0 /= nOPD
    Int_D  /= nOPD

    norm_peakD0 = 1/np.max(Int_D0)

    Int_D0 *= norm_peakD0
    Int_D  *= norm_peakD0
    
    """
    ### Compute the radial intensity profiles of the images
    """
    # computation of the averaged intensity profiles of the images   
    Int_D0_prf[1,:], rad_D0_prf_avg = radial_profile(Int_D0[:,:])
    Int_D_prf[1,:], rad_D_prf_avg = radial_profile(Int_D[:,:])

    # convert pixel scale into lam/D scale for the x-axis
    rad_D0_prf_avg_lamD = rad_D0_prf_avg * mD/nImg
    rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg
    
    # rad_D0_prf_avg_mas = rad_D0_prf_avg_lamD * lamD2mas
    Int_D0_prf[0,:] = rad_D0_prf_avg_lamD * (lam/D) / rad2mas
    # rad_D_prf_avg_mas = rad_D_prf_avg_lamD * lamD2mas
    Int_D_prf[0,:] = rad_D_prf_avg_lamD * (lam/D) / rad2mas

    shm_opd.close()
    shm_opd.unlink()
    shm_pupil.close()
    shm_pupil.unlink()
    shm_lyot.close()
    shm_lyot.unlink()

    return i, Int_D0, Int_D, Int_D0_prf, Int_D_prf, Int_DD0, Int_DD


if __name__ == "__main__":
    
    """
    ### Parameters
    """

    nFPM = 100  # Sampling of the coronagraph focal plane mask
    #nImg = 400  # last focal plane image size
    nImg = 400
    nOPD = 4000  # # of AO corrected phase residuals screens
    lam_ref = 1600e-9  # reference wvl
    lam_min = 950e-9 # 950e-9  # minimal value of wvl
    lam_itv = 18  # # of intervals
    lam_stp = 50e-9  # interval step
    D = 38.54  #  main pupil diameter/ elt pupil diameter
    # pscale = 0.3  # image plate scale in last focal plane
    pscale = 0.3
    # diam = 0.89 obs = 0.35, mB = 3.9 2025 with 'ELT_pupil_400.fits' YJH
    # diam = 0.90 obs = 0.37, mB = 4.0 2024 with 'ELT_pupil_400.fits' YJH @ 25 mas / 75% thr
    # diam = 0.96 obs = 0.30, mB = 4.5 2024 with 'ELT_pupil_400.fits' K
    # diam = 0.90 obs = 0.38 with 'Tel-Pupil.fits'
    diam = 0.89  # Lyot fractional outer diameter
    obst = 0.35  # Lyot fractional inner/obscuration diameter
    mB = 3.9        # Lyot focal plane mask diameter in lam_ref/D

    disp = 5e6  # dispersion mas/mu e.g 8e7 ou 80e6 = 80 mas / 1e-6 m
    fpm_dec_mas = 2  # psf to fpm decentering in mas
    ls_ape = 1  # lyot stop angular position error in degres
    ls_voe = 0  # lyot stop vertical - elevation - offset error in pixels
    ls_hoe = 4 # lyot stop horizontal - elevation - offset error in pixels
    ncpa_rms = 30  # scale of ncpa phase screens in nm RMS
    fpm_dfe_elt = 30  # scale of defocus at fpm in nm RMS for reference wvl
    
    user = 'Alain'
    usr_base = 'D:/Andes/Data_corono/'
    fnm_eltp = 'ELT_pupil_400.fits' # New pupil with new spider
    # unscaled unmasked ncpa file twice the size of the pupil size / nPup
    ncpa_fnm = ('ncpa_unscaled_x2_ELT_pupil_400.fits')
    dir_root = 'OPDs_PASSATA/OPD/WS/1kHzVarWS/'  # root of OA phase screens set
    # dir_root = 'OPDs_PASSATA/OPD/WS/500HzVarWS/'  # root of OA phase screens set

    # av = sys.argv
    # print('\n', av)
    # if len(av) > 1: input_parameters = av[1]  # PYTHONPATH update?
    
    # try:
        
    #     import input_parameters as ip
    #     if hasattr(ip, 'nFPM'): nFPM = ip.nFPM
    #     if hasattr(ip, 'nImg'): nImg = ip.nImg
    #     if hasattr(ip, 'nOPD'): nOPD = ip.nOPD
    #     if hasattr(ip, 'lam_ref'): lam_ref = ip.lam_ref
    #     if hasattr(ip, 'lam_min'):  lam_min = ip.lam_min
    #     if hasattr(ip, 'lam_itv'):  lam_itv = ip.lam_itv
    #     if hasattr(ip, 'lam_stp'):  lam_stp = ip.lam_stp
    #     if hasattr(ip, 'D'):    D = ip.D
    #     if hasattr(ip, 'pscale'):   pscale = ip.pscale
    #     if hasattr(ip, 'diam'): diam = ip.diam
    #     if hasattr(ip, 'obst'): obst = ip.obst
    #     if hasattr(ip, 'mB'):   mB = ip.mB
    #     if hasattr(ip, 'disp'): disp = ip.disp
    #     if hasattr(ip, 'fpm_dec_mas'):  fpm_dec_mas = ip.fpm_dec_mas
    #     if hasattr(ip, 'ls_ape'):   ls_ape = ip.ls_ape
    #     if hasattr(ip, 'ls_voe'):   ls_voe = ip.ls_voe
    #     if hasattr(ip, 'ls_hoe'):   ls_hoe = ip.ls_hoe
    #     if hasattr(ip, 'ncpa_rms'): ncpa_rms = ip.ncpa_rms
    #     if hasattr(ip, 'fpm_dfe_elt'):  fpm_dfe_elt = ip.fpm_dfe_elt
    #     if hasattr(ip, 'user'):user = ip.user
    #     if hasattr(ip, 'usr_base'): usr_base = ip.usr_base
    #     if hasattr(ip, 'fnm_eltp'): fnm_eltp = ip.fnm_eltp
    #     if hasattr(ip, 'ncpa_fnm'): ncpa_fnm = ip.ncpa_fnm
    #     if hasattr(ip, 'dir_root'): dir_root = ip.dir_root

        
    # except ModuleNotFoundError:
        
    #     print('use defaults parameters')
    
     
    lam_lst = np.arange(lam_min,lam_min+(lam_itv+0.5)*lam_stp,lam_stp)
    # lam_lst = [950e-9, 1200e-9, 1350e-9, 1600e-9, 1850e-9]

    nL = len(lam_lst)

    # conversions
    rad2mas = np.pi/(180.*3600*1000)
    mas2rad = 1/rad2mas

    # fov in mas
    fov_mas = nImg * pscale
    # field of view in radians
    fov_rdn = fov_mas * rad2mas
    # in multiple of reference lambda (lam_ref) over D
    mD_ref = fov_rdn / ( lam_ref / D )
    
    # Focal plane mask
    mask2d = uniform_disk(nFPM, nFPM/2.)
    
    # psf to fpm decentering in radians for legacy
    fpm_dec = fpm_dec_mas * rad2mas

    # fpm_dfe = 0 if fpm_dfe_elt = 0., computed dynamicaly otherwise
    fpm_dfe = fpm_dfe_elt * -1.
      
    # datetime of script execution
    donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now
    print('date of now : ', donow)
    
    
    #%%
    """
    ### Working directories
    """
    
    if user == 'Alain':
        fdir_dat = Path(usr_base+"/data/").resolve()
        fdir_res   = Path(usr_base+"/results/").resolve()
        fdir_plt   = Path(usr_base+"/plots/").resolve()
       
    # Directory for the pupils
    fdir_pupil = fdir_dat / 'Pupil'
    
    fdir_res = fdir_res / donow
    fdir_prfct = fdir_res / 'perfect'
    os.makedirs(fdir_prfct, exist_ok=True)
    
    fdir_plt = fdir_plt / donow
    
    
    #%%
    """
    data cubes for all psfs and profiles
    """
    
    Int_DD0 = np.zeros([nL, nImg, nImg])
    Int_DD = np.zeros([nL, nImg, nImg])
    Int_D0 = np.zeros([nL, nImg, nImg])
    Int_D = np.zeros([nL, nImg, nImg])
    Int_D0_prf_avg = np.zeros([nL, 2, nImg//2])
    Int_D_prf_avg = np.zeros([nL, 2, nImg//2])
    
    """
    ncpa's, pupils, ncpa, defocus
    """
    # Filename and path for the ELT pupil
    fpath_elt = fdir_pupil / fnm_eltp
    # Read ELT pupil 
    Pupil = fits.getdata(fpath_elt,)
    nPup = Pupil.shape[0]
    ipup = np.nonzero(Pupil)
    
    # 2D array pupil slope for tilt
    sf_x = np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5
    sf_y = np.transpose(sf_x.copy())[::-1,:]
    
    #2D array for distance to center pixel in pupil, in [0,1]
    kx = (np.arange(nPup)-nPup//2)/(nPup/2)
    ky = (np.arange(nPup)-nPup//2)/(nPup/2)
    kx2, ky2 = np.meshgrid(kx, ky)
    rho = np.sqrt(kx2**2 + ky2**2)
    
    if ncpa_rms != 0:
        
        ncpa_1 = fits.getdata(fdir_dat/ncpa_fnm)
        nMap = ncpa_1.shape[0]
        
        N=nMap//2
        hlf=N//2
        
    dfc = np.zeros((nPup,nPup))
    if fpm_dfe_elt > 0:
        
        # create circular pupil (elt circumcircle) phase with defocus
        dfc = uniform_disk(nPup,nPup//2)
        iok = np.nonzero(dfc)
        dfc[iok] = np.sqrt(3.)*(rho[iok]*rho[iok]-1.)
    
        # scale defocus inside elt pupil to required value
        dfc_temp = dfc.copy()
        dfc_temp *= Pupil.copy()
        mpu = np.mean(dfc_temp[ipup])
        dfc_temp -= mpu
        sp = np.std(dfc_temp[ipup])
        dfc_temp /= sp
        dfc_temp *= fpm_dfe_elt * 1e-9
    
        # compute input defocus over elt circum circle in front of elt pupil
        dfc -= mpu
        dfc /= sp
        dfc *= fpm_dfe_elt * 1e-9
        dfc -= np.mean(dfc[iok])
        fpm_dfe = np.std(dfc[iok]) * 1e9
        
        # update defocus phase screen to elt pupil shape with the required value
        dfc = dfc_temp.copy()    
        
    
    #%%
    '''
    rotate then shift Lyot Stop and defocus
    '''
    # Lyot stop
    LyotStop2d = Pupil*(uniform_disk(nPup, diam*nPup/2) -
                        uniform_disk(nPup, obst*nPup/2))
    
    if ls_ape != 0:
        
        pup_rot = ndimage.rotate(Pupil,ls_ape, reshape=False)
        pup_rot = pup_rot > 0.5
        LyotStop2d = pup_rot*(uniform_disk(nPup, diam*nPup/2) -
                              uniform_disk(nPup, obst*nPup/2))
    
    if ls_voe != 0 or ls_hoe != 0:
        
        LyotStop2d = np.roll(LyotStop2d,(ls_hoe,ls_voe),(1,0))
    
    
    #%%
    """
    ### working directory of the OPD files
    """
    
    # opds_dir=(dir_root+'1',  # )
    #           dir_root+'2', dir_root+'3', dir_root+'4', dir_root+'5',
    #           dir_root+'6', dir_root+'7', dir_root+'8', dir_root+'9',
    #           dir_root+'10')
    opds_dir = os.listdir(usr_base / fdir_dat / dir_root)
    # print(opds_dir)
    
    # for d_nb in range(len(opds_dir)):
    for d_nb, opd_set in enumerate(opds_dir):

        fdir_opd = usr_base / fdir_dat /  dir_root / opds_dir[d_nb]
        
        opd_set = os.path.basename(fdir_opd).split('.')[0]
        
        # opd_set='toto'
        os.makedirs(fdir_res / opd_set, exist_ok=True)
        os.makedirs(fdir_plt / opd_set, exist_ok=True)
        
        
        #%%
        
        # Filename and path for the OPD maps
        flist_opd = []
        for fnm in os.listdir(fdir_opd):
            if fnm.endswith(".fits"):
                flist_opd.append(fnm)
        
        flist_opd = sorted(flist_opd,key=len)
        # flist_opd = sorted(os.listdir(fdir_opd),key=len) 
        nof = len(flist_opd)
        # print("# of files in phase screens dir:", nof)
        fpath_opd = [fdir_opd / flist_opd[i] for i in range(nof)]
        # fpath_opd = sorted(fpath_opd)
        nW = int(np.floor((nof/nOPD)+1))
        if nW>2:
            fpath_opd = [fpath_opd[i] for i in range(1,nof,nW)]
        nOPD = len(fpath_opd)
        # print('sample size of OPD files:', nOPD)
        
        
        #%%
        """
        ### Read OPD files
        """
        
        # Read OPD maps for the nOPD files
        OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])
    
        # pour jeu fichier fits unique, e.g: OPDs_PASSATA/OPD/WS/ASI_*
        if nof < 2:
            start = int((OPD_arr.shape)[3]/5)
            # print("skip:", start)
            OPD_arr = OPD_arr[0,:,:,start:]
            OPD_arr = np.transpose(OPD_arr,(2,0,1))
            nOPD = OPD_arr.shape[0]
            # print("phase screen array shape from single file: ", OPD_arr.shape)
        else:
            print("phase screen array shape from set of files: ",OPD_arr.shape)
           
        OPD_arr *= 1e-9 # convert OPD from nm to m if new OPD with new pupil
        
        if ncpa_rms != 0:
            
            fnm = ('ncpa_ELT_pupil_400_30nm_4096screens.fits')
            ncpa_1 = fits.getdata(fdir_dat/fnm)             
            # rnd=np.random.randn(nOPD)
            # rnd /= 2.
            # xi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)
            
            # rnd=np.random.randn(nOPD)
            # rnd /= 2.
            # yi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)
    
            for n in range(nOPD):
                
                OPD_arr[n,:,:] += ncpa_1[n,:,:] * Pupil.copy()
                # temp = ((ncpa_1[hlf+xi[n]:hlf+N+xi[n],
                #                 hlf+yi[n]:hlf+N+yi[n]]).copy() * Pupil.copy())
                
                # temp -= np.mean(temp[ipup])
                # temp /= np.std(temp[ipup])
                # temp *= float(ncpa_rms) * 1e-9
                # OPD_arr[n,:,:] += temp.copy() * Pupil.copy()
                # temp *= 0.
    
        mp.set_start_method("spawn", force=True)
    
        # --- OPD shared ---
        shm_opd = shared_memory.SharedMemory(create=True,
                                             size=OPD_arr.nbytes)
    
        shm_opd_arr = np.ndarray(OPD_arr.shape,
                                 dtype=OPD_arr.dtype,
                                 buffer=shm_opd.buf)
    
        shm_opd_arr[:] = OPD_arr[:]
    
        # --- Pupil shared ---
        shm_pupil = shared_memory.SharedMemory(create=True,
                                               size=Pupil.nbytes)
    
        shm_pupil_arr = np.ndarray(Pupil.shape,
                                   dtype=Pupil.dtype,
                                   buffer=shm_pupil.buf)
    
        shm_pupil_arr[:] = Pupil[:]
    
        # --- Lyot shared ---
        shm_lyot = shared_memory.SharedMemory(create=True,
                                              size=LyotStop2d.nbytes)
    
        shm_lyot_arr = np.ndarray(LyotStop2d.shape,
                                  dtype=LyotStop2d.dtype,
                                  buffer=shm_lyot.buf)
    
        shm_lyot_arr[:] = LyotStop2d[:]
    
        nproc = min(cpu_count(), nL)
    
        args_list = [
            (i, lam_lst[i],
             shm_opd.name, OPD_arr.shape, OPD_arr.dtype,
             shm_pupil.name, Pupil.shape, Pupil.dtype,
             shm_lyot.name, LyotStop2d.shape, LyotStop2d.dtype,
             mask2d, fpm_dec, dfc, sf_y, D, diam, nPup,
             nImg, nFPM, mB, lam_ref, mD_ref, disp, rad2mas)
            for i in range(nL)
        ]
    
        with mp.Pool(processes=nproc) as pool:
            
            results = pool.map(compute_one_lambda_shared_full, args_list)
            
        for i, Int_D0_i, Int_D_i, Int_D0_prf, Int_D_prf, Int_DD0_i, Int_DD_i in results:
            
            Int_D0[i,:,:] = Int_D0_i
            Int_D[i,:,:]  = Int_D_i
            Int_D0_prf_avg[i,:,:] = Int_D0_prf
            Int_D_prf_avg[i,:,:] = Int_D_prf
            Int_DD0[i,:,:] = Int_DD0_i
            Int_DD[i,:,:] = Int_DD_i

        # filename for the direct and coronagraphic images and profiles
        fname_Int_D0 = 'ao_corr_psf_'+donow+'.fits'
        fname_Int_D = 'ao_corr_coro_psf_'+donow+'.fits'
        fname_Prf_D0 = 'ao_corr_psf_profile_'+donow+'.fits'
        fname_Prf_D = 'ao_corr_coro_psf_profile_'+donow+'.fits'
    
        # filepath for the direct and coronagraphic images
        fpath_Int_D0 = fdir_res / opd_set / fname_Int_D0
        fpath_Int_D  = fdir_res / opd_set / fname_Int_D
        fpath_Prf_D0 = fdir_res / opd_set / fname_Prf_D0
        fpath_Prf_D  = fdir_res / opd_set / fname_Prf_D
        # save the direct and coronagraphic images
    
        fits.writeto(fpath_Int_D0, Int_D0, overwrite=True)
        fits.writeto(fpath_Int_D, Int_D, overwrite=True)
        fits.writeto(fpath_Prf_D0, Int_D0_prf_avg, overwrite=True)
        fits.writeto(fpath_Prf_D, Int_D_prf_avg, overwrite=True)
        
        fpath_psf_lst=(fpath_Int_D0, fpath_Int_D, fpath_Prf_D0, fpath_Prf_D)

        if len(os.listdir(fdir_prfct))==0:
            fname_Int_DD0 = 'wonoise_psf_'+donow+'.fits'
            fname_Int_DD = 'wonoise_coro_psf_'+donow+'.fits'
            fpath_Int_DD0 = fdir_prfct/ fname_Int_DD0
            fpath_Int_DD  = fdir_prfct / fname_Int_DD
    
            if not os.path.isfile(fpath_Int_DD0):
                fits.writeto(fpath_Int_DD0, Int_DD0, overwrite=True)
                fpath_psf_lst += (fpath_Int_DD0,)
                
            if not os.path.isfile(fpath_Int_DD):
                fits.writeto(fpath_Int_DD, Int_DD, overwrite=True)
                fpath_psf_lst += (fpath_Int_DD,)
        
        hdr_keys = {'NPUP':(nPup,'pupil size'),
                    'NFPM':(nFPM,'FP coro. sampling'),
                    'NIMG':(nImg,'image size'),
                    'NOPD':(nOPD,'number of OPD files'),
                    'FOVS':(fov_mas,'field of view in mas'),
                    'LMIN':(lam_min,'wavelength in meters'),
                    'LITV':(lam_itv,'# of wvl intervals'),
                    'LSTP':(lam_stp,'wvl step in meters'),
                    'LMBD':(lam_ref,'reference wvl in meters'),
                    'DIAM':(D,'pupil dimater in meters'),
                    'PSCL':(pscale,'plate scale in mas'),
                    'SFPM':(mB,'FPM (LMBD/D), first focal plane'),
                    'FDIA':(diam,'fractional pup. diameter'),
                    'OBST':(obst,'fractional obscuration'),
                    'OPDS':(opd_set,'opd set creation date'),
                    'DISP':(disp,'achr. disp. in mas/m bw'),
                    'NCPA':(ncpa_rms,'ncpa rms in nanometers'),
                    'FDEC':(fpm_dec,'psf to fpm offset in radians'),
                    'FTLT':(fpm_dec_mas,'psf to fpm offset in mas'),
                    'LSAE':(ls_ape,'lyot stop angular position error in degrees'),
                    'LSVE':(ls_voe,'lyot stop vertical offset error in pixels'),
                    'LSHE':(ls_hoe,'lyot stop horizontal offset error in pixels'),
                    'ELT_DFOC':(fpm_dfe_elt,'elt pupil defocus in nm RMS at LMBD'),
                    'DIAM_DFC':(fpm_dfe,'pup. circumcirc. defocus nm RMS at LMBD'),
                    'EPUP_FNM':(fnm_eltp,'ELT pupil filename'),
                    'DATE_NOW':(donow,'date of now: script execution date'),
                    'DIR_ROOT':(dir_root,'root of OA residuals data'),
                    'NCPA_FNM':(ncpa_fnm,'filename for ncpa generation')}

        for fpath in fpath_psf_lst:
            for n, k in enumerate(hdr_keys):
                fits.setval(fpath,k,value=hdr_keys[k][0],comment=hdr_keys[k][1])
                    
    # datetime of script execution
    donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now
    
