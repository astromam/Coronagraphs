#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys

try:
    
    gpu_test = subprocess.check_output('nvidia-smi')
    
    import cupy as cp  # GPU Library
    
    print(cp.cuda.runtime.getDeviceProperties(0)['name'])
    
    def sft(A2, NB, m, inv=False, CtrBtwnPix=False):
        """
        GPU-accelerated Slow Fourier Transform (Matrix DFT).
        Handles both 2D (single image) and 3D (batch of images) inputs.
        """
        # A2 shape: (Batch, NA, NA) or (NA, NA)
        is_batch = (A2.ndim == 3)
        NA = A2.shape[-2]
        
        val = 0.5 if CtrBtwnPix else 0.0
        coeff = m / (NA * NB)
        sign = 1.0 if inv else -1.0
        
        # Create coordinate vectors on GPU
        range_NA = cp.arange(NA, dtype=cp.float64)
        range_NB = cp.arange(NB, dtype=cp.float64)
        
        # X and U vectors
        X = (range_NA - NA/2.0 + val).reshape(1, -1) * (1.0/NA)
        U = (range_NB - NB/2.0 + val).reshape(1, -1) * (m/NB)
        
        # Kernel creation: exp(i * 2pi * X.T @ U)
        # Using matrix mult for outer product
        XU = (2.0 * cp.pi) * cp.matmul(X.T, U)

        # Transformation matrices
        A3 = cp.exp(1j * sign * XU)
        A1 = A3.T
        
        if is_batch:
            # Batch Matrix Multiplication: 
            # We need B = A1 @ A2 @ A3
            # Step 1: 
            # temp = A2 @ A3.  (Batch, NA, NA) @ (NA, NB) -> (Batch, NA, NB)
            temp = cp.matmul(A2, A3)
            
            # Step 2: res = A1 @ temp. 
            # A1 is (NB, NA). temp is (Batch, NA, NB).
            # Standard matmul broadcasts:
            # (N,M) against (Batch, M, P) -> (Batch, N, P)
            B_out = cp.matmul(A1, temp)
        else:
            # Standard 2D
            B_out = A1 @ A2 @ A3
    
        return coeff * B_out
    
    def isft(A2, NB, m, CtrBtwnPix=False):
        return sft(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)
    
    print('Nvidia GPU detected!')

except FileNotFoundError:
    
    print('nvidia-smi is not available')       
    print('No Nvidia GPU in system!')
    sys.exit()
  
import numpy as np
from scipy import ndimage
from astropy.io import fits
from uniform_disk import uniform_disk
from psf_profile import radial_profile

import os
from pathlib import Path
from datetime import datetime

# conversion l radian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas


def compute_images_gpu_batch(c, elt_gpu, LyotStop2d_gpu, opds_gpu, mask2d_gpu,
                             lam_lst_gpu, nImg, mB, lam_ref, diam, obst,
                             mD_ref, nFPM, disp, rad2mas, fpm_dec, nPup):
    """
    Process all OPDs in chunks to calculate the average intensity images.
    """

    chnk_v = cp.ones(opds_gpu.shape[0])
    nL = lam_lst_gpu.shape[0]
    Int_D0 = cp.zeros((nL,nImg, nImg))
    Int_D  = cp.zeros((nL,nImg, nImg))
    Int_DD0 = cp.zeros((nL,nImg, nImg))
    Int_DD = cp.zeros((nL,nImg, nImg))
    
    for ilam in range(nL):
        
        # Sélection GPU (si plusieurs GPU disponibles)M
        dev_id = ilam % cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device(dev_id).use()

        lam = lam_lst_gpu[ilam]
        dLam = lam - lam_ref
        tilt = dLam * disp * rad2mas
        mD = mD_ref * lam_ref / lam
        
        # PERFECT PSF
        Fld_AA0 = elt_gpu * LyotStop2d_gpu
        Fld_DD0 = sft(Fld_AA0, nImg, mD*diam)
        Int_DD0[ilam,:,:] = cp.abs(Fld_DD0)**2
    
        # perfect coronographic image
        
        Fld_AA = elt_gpu * 1.
        Fld_BB = mask2d_gpu*sft(Fld_AA, nFPM, mB*lam_ref/lam)
        Fld_CC = Fld_AA - isft(Fld_BB, nPup, mB*lam_ref/lam)
        Fld_LL = Fld_CC * LyotStop2d_gpu
        Fld_DD = sft(Fld_LL, nImg, mD*diam)
        Int_DD[ilam,:,:] = cp.abs(Fld_DD)**2
    
        phase = (opds_gpu + dfc_gpu[None,:,:] + (fpm_dec + tilt) *
                 chnk_v[:,None,None] * D * diam * sf_y_gpu[None,:,:] / nPup)

        Fld_A0 = (elt_gpu * cp.exp(1j * 2 * cp.pi * phase / lam) *
                  LyotStop2d_gpu)
    
        Fld_D0 = sft(Fld_A0, nImg, mD*diam)
        Fld_D0 = cp.abs(Fld_D0)**2
        Int_D0[ilam,:,:] = cp.mean(Fld_D0, axis=0)
    
        # CORO
        Fld_A0 = (elt_gpu * cp.exp(1j*2*cp.pi * phase / lam))
    
        Fld_B = mask2d_gpu*sft(Fld_A0, nFPM, mB*lam_ref/lam)
        Fld_C = Fld_A0 - sft(Fld_B, nPup, mB*lam_ref/lam, inv=True)
        Fld_L = Fld_C * LyotStop2d_gpu
        Fld_D = sft(Fld_L, nImg, mD*diam)
        Fld_D = cp.abs(Fld_D)**2
        Int_D[ilam,:,:] = cp.mean(Fld_D, axis=0)
    
    return Int_D0, Int_D, Int_DD0, Int_DD


if __name__ == "__main__":
    
    """
    ### Parameters
    """

    nFPM = 100  # Sampling of the coronagraph focal plane mask
    nImg = 400  # last focal plane image size
    nOPD = 4000  # # of AO corrected phase residuals screens
    lam_ref = 1600e-9  # reference wvl
    lam_min = 950e-9  # minimal value of wvl
    lam_itv = 18  # # of intervals
    lam_stp = 50e-9  # interval step
    D = 38.54  #  main pupil diameter/ elt pupil diameter
    pscale = 0.3  # image plate scale in last focal plane

    # diam = 0.89 obs = 0.35, mB = 3.9 2025 with 'ELT_pupil_400.fits' YJH
    # diam = 0.90 obs = 0.37, mB = 4.0 2024 with 'ELT_pupil_400.fits' YJH @ 25 mas / 75% thr
    # diam = 0.96 obs = 0.30, mB = 4.5 2024 with 'ELT_pupil_400.fits' K
    # diam = 0.90 obs = 0.38 with 'Tel-Pupil.fits'
    diam = 0.89  # Lyot fractional outer diameter
    obst = 0.35  # Lyot fractional inner/obscuration diameter
    mB = 3.9        # Lyot focal plane mask diameter in lam_ref/D

    disp = 0  # dispersion mas/mu e.g 8e7 ou 80e6 = 80 mas / 1e-6 m
    fpm_dec_mas = 0  # psf to fpm decentering in mas
    ls_ape = 0  # lyot stop angular position error in degres
    ls_voe = 0  # lyot stop vertical - elevation - offset error in pixels
    ls_hoe = 0  # lyot stop horizontal - elevation - offset error in pixels
    ncpa_rms = 0  # scale of ncpa phase screens in nm RMS
    fpm_dfe_elt = 0  # scale of defocus at fpm in nm RMS for reference wvl
    
    user = 'Alain'
    usr_base = 'D:/Andes/Data_corono/'
    fnm_eltp = 'ELT_pupil_400.fits' # New pupil with new spider
    # unscaled unmasked ncpa file twice the size of the pupil size / nPup
    ncpa_fnm = ('ncpa_unscaled_x2_ELT_pupil_400.fits')
    dir_root = 'OPDs_PASSATA/OPD/WS/1kHzVarWS/'  # root of OA phase screens set
    
    av = sys.argv
    if len(av) > 1: input_parameters = av[1]  # PYTHONPATH update?
    print('\n', av)

    try:
        
        import input_parameters as ip
        if hasattr(ip, 'nFPM'): nFPM = ip.nFPM
        if hasattr(ip, 'nImg'): nImg = ip.nImg
        if hasattr(ip, 'nOPD'): nOPD = ip.nOPD
        if hasattr(ip, 'lam_ref'): lam_ref = ip.lam_ref
        if hasattr(ip, 'lam_min'):  lam_min = ip.lam_min
        if hasattr(ip, 'lam_itv'):  lam_itv = ip.lam_itv
        if hasattr(ip, 'lam_stp'):  lam_stp = ip.lam_stp
        if hasattr(ip, 'D'):    D = ip.D
        if hasattr(ip, 'pscale'):   pscale = ip.pscale
        if hasattr(ip, 'diam'): diam = ip.diam
        if hasattr(ip, 'obst'): obst = ip.obst
        if hasattr(ip, 'mB'):   mB = ip.mB
        if hasattr(ip, 'disp'): disp = ip.disp
        if hasattr(ip, 'fpm_dec_mas'):  fpm_dec_mas = ip.fpm_dec_mas
        if hasattr(ip, 'ls_ape'):   ls_ape = ip.ls_ape
        if hasattr(ip, 'ls_voe'):   ls_voe = ip.ls_voe
        if hasattr(ip, 'ls_hoe'):   ls_hoe = ip.ls_hoe
        if hasattr(ip, 'ncpa_rms'): ncpa_rms = ip.ncpa_rms
        if hasattr(ip, 'fpm_dfe_elt'):  fpm_dfe_elt = ip.fpm_dfe_elt
        if hasattr(ip, 'user'):user = ip.user
        if hasattr(ip, 'usr_base'):    usr_base = ip.usr_base
        if hasattr(ip, 'fnm_eltp'):    fnm_eltp = ip.fnm_eltp
        if hasattr(ip, 'ncpa_fnm'):ncpa_fnm = ip.ncpa_fnm
        if hasattr(ip, 'dir_root'): dir_root = ip.dir_root

        
    except ModuleNotFoundError:
        
        print('use defaults parameters')

    """
    ### Parameters
    """
    
    # trial and error
    GPU_BATCH_SIZE = 100 

    lam_lst = np.arange(lam_min,lam_min+(lam_itv+0.5)*lam_stp,lam_stp)
    nL = len(lam_lst)

    # conversion l radian to mas
    rad2mas = np.pi/(180.*3600*1000)
    mas2rad = 1/rad2mas
    
    # field of view in mas
    fov_mas = nImg * pscale
    # in radians
    fov_rdn = fov_mas * rad2mas
    # in multiple of reference lambda (lam_ref) over D
    mD_ref = fov_rdn / ( lam_ref / D )
    
    """
    ### Coronagraphic components
    """
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
    
    #  elif user == 'toto':
    
    # Directory for the pupils
    fdir_pupil = fdir_dat / 'Pupil'
    
    fdir_res = fdir_res / donow
    fdir_prfct = fdir_res / 'perfect'
    os.makedirs(fdir_prfct, exist_ok=True)
    
    fdir_plt = fdir_plt / donow
    
    
    #%%
    """
    ncpa's, pupils, ncpa, defocus
    """
    # Filename and path for the ELT pupil
    fpath_elt = fdir_pupil / fnm_eltp
    # Read ELT pupil 
    Pupil = fits.getdata(fpath_elt,)
    elt_gpu = cp.asarray(Pupil.astype(np.float64))
    nPup = Pupil.shape[0]
    ipup = np.nonzero(Pupil)
    
    # 2D array pupil slope for tilt
    sf_x = np.broadcast_to(np.arange(-nPup//2,nPup//2,1),(nPup,nPup)) + 0.5
    sf_y = np.transpose(sf_x.copy())
    sf_x_gpu = cp.asarray(sf_x.copy(), dtype=cp.float64)
    sf_y_gpu = cp.asarray(sf_y.copy(), dtype=cp.float64)

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
        dfc_gpu = cp.asarray(dfc_temp.copy(), dtype=cp.float64)    
            
    
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
    
    Pupil_gpu = cp.asarray(Pupil, dtype=cp.float64)
    LyotStop2d_gpu = cp.asarray(LyotStop2d, dtype=cp.float64)
    mask2d_gpu = cp.asarray(mask2d, dtype=cp.float64)
    lam_lst_gpu = cp.asarray(lam_lst, dtype=cp.float64)


    #%%
    """
    ### working directory of the OPD files
    """
    #%%
    
    # dir_root = 'OPDs_PASSATA/OPD/WS/1kHzVarWS/'
    # opds_dir=(root+'1',)
    opds_dir=(dir_root+'1', dir_root+'2', dir_root+'3', dir_root+'4',
              dir_root+'5', dir_root+'6', dir_root+'7', dir_root+'8',
              dir_root+'9', dir_root+'10')

    for dir_nb in range(len(opds_dir)):
    
        new_spider_flare = fdir_dat / opds_dir[dir_nb]
        fdir_opd   = new_spider_flare
        
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
        fpath_opd = [fdir_opd / flist_opd[i] for i in range(nof)]
        # fpath_opd = sorted(fpath_opd)
        nW = int(np.floor((nof/nOPD)+1))
        if nW>2:
            fpath_opd = [fpath_opd[i] for i in range(1,nof,nW)]
        nOPD = len(fpath_opd)
        
        
        #%%
        """
        ### Read OPD files
        """
        
        # Read OPD maps for the nOPD files
        OPD_arr = np.asarray([fits.getdata(fpath_opd[i]) for i in range(nOPD)])
    
        # pour jeu fichier fits unique, e.g: OPDs_PASSATA/OPD/WS/ASI_*
        if nof < 2:
            start = int((OPD_arr.shape)[3]/5)
            OPD_arr = OPD_arr[0,:,:,start:]
            OPD_arr = np.transpose(OPD_arr,(2,0,1))
            nOPD = OPD_arr.shape[0]
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
            
        nOPD = OPD_arr.shape[0]
        opd_arr_sz = OPD_arr.nbytes
        one_opd_sz = int(np.rint(float(opd_arr_sz) / float(nOPD)))
        
        totGlobCpuMem = float(
            cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'])
        
        n_time_one = 8 * 7.
        GPU_BATCH_SIZE = np.min(
            [np.rint(totGlobCpuMem / (one_opd_sz * n_time_one)), nOPD])
        if GPU_BATCH_SIZE < nOPD:
            while nOPD % GPU_BATCH_SIZE >1:
                GPU_BATCH_SIZE -= 1
                     
        nchk = int(np.rint(nOPD//GPU_BATCH_SIZE))
        GPU_BATCH_SIZE = int(np.rint(GPU_BATCH_SIZE))
        print('GPU_BATCH_SIZE: ',GPU_BATCH_SIZE, ', # of chunks:', nchk)
        
        Int_DD0 = np.zeros([nL, nImg, nImg])
        Int_DD = np.zeros([nL, nImg, nImg])
        Int_D0 = np.zeros([nL, nImg, nImg])
        Int_D = np.zeros([nL, nImg, nImg])
        Int_D0_prf_avg = np.zeros([nL, 2, nImg//2])
        Int_D_prf_avg = np.zeros([nL, 2, nImg//2])

        for c in range(nchk):

            opd_gpu = cp.asarray(
                OPD_arr[GPU_BATCH_SIZE*c:GPU_BATCH_SIZE*(c+1),:,:],
                dtype=cp.float64)
        
            gpu_ret = compute_images_gpu_batch(c,
                elt_gpu, LyotStop2d_gpu, opd_gpu, mask2d_gpu, lam_lst_gpu,
                nImg, mB, lam_ref, diam, obst, mD_ref, 
                nFPM, disp, rad2mas, fpm_dec, nPup )
            
            # Transfer result back to CPU RAM immediately
            Int_D0  += gpu_ret[0].get()
            Int_D   += gpu_ret[1].get()
            Int_DD0 += gpu_ret[2].get()
            Int_DD  += gpu_ret[3].get()

            # Clean up GPU memory for this minute
            del opd_gpu
            cp.get_default_memory_pool().free_all_blocks()
        
        norm_D0 = np.max(Int_D0, axis=(1,2))
        Int_D0  /= norm_D0[:,None,None]
        Int_D   /= norm_D0[:,None,None]
        
        norm_DD0 = np.max(Int_DD0, axis=(1,2))
        Int_DD0  /= norm_DD0[:,None,None]
        Int_DD   /= norm_DD0[:,None,None]
        
        for ilam in range(nL):
            
            """
            ### Compute the radial intensity profiles of the images
            """
            lam = lam_lst[ilam]
            mD = mD_ref * lam_ref / lam

            # conversion lam/D to mas
            lamD2mas = (lam / D) * mas2rad

            # computation of the averaged intensity profiles of the images   
            Int_D0_prf_avg[ilam,1,:], rad_D0_prf_avg = (
                radial_profile(Int_D0[ilam,:,:]))
            Int_D_prf_avg[ilam,1,:], rad_D_prf_avg = (
                radial_profile(Int_D[ilam,:,:]))
        
            # convert pixel scale into lam/D scale for the x-axis
            rad_D0_prf_avg_lamD = rad_D0_prf_avg * mD/nImg
            rad_D_prf_avg_lamD = rad_D_prf_avg * mD/nImg
            
            Int_D0_prf_avg[ilam,0,:] = rad_D0_prf_avg_lamD * lamD2mas
            Int_D_prf_avg[ilam,0,:] = rad_D_prf_avg_lamD * lamD2mas

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
                    'DATE_NOW':(donow,'date of now, script execution date'),
                    'DIR_ROOT':(dir_root,'root of OA residuals data'),
                    'NCPA_FNM':(ncpa_fnm,'filename for ncpa generation')}

        for fpath in fpath_psf_lst:
            for n, k in enumerate(hdr_keys):
                fits.setval(fpath,k,value=hdr_keys[k][0],comment=hdr_keys[k][1])

