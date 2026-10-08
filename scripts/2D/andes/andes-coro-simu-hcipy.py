# -*- coding: utf-8 -*-

import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import cpu_count

import os 
from pathlib import Path
from datetime import datetime

import numpy as np
# import matplotlib.pyplot as plt
import hcipy as hci
from astropy.io import fits


rad2mas = np.pi/(180.*3600.*1000.)
mas2rad = 1./rad2mas

os.environ["OMP_NUM_THREADS"] = "1"

def compute_one_lambda_shared_full(args):

    (i, lam, ytlt, dfoc, propagator, coro,
     telescope_pupil, lyot_stop, pupil_grid, focal_grid,
     shm_opd_name, shape_opd, dtype_opd,
     nImg, pscale, px2binSz) = args
     
    # # --- reconnect shared arrays ---
    shm_opd = shared_memory.SharedMemory(name=shm_opd_name)
    OPD_arr = np.ndarray(shape_opd, dtype=dtype_opd, buffer=shm_opd.buf)

    n_opd = OPD_arr.shape[2]

    Int_D0 = np.zeros((nImg * nImg))
    Int_D  = Int_D0.copy()
    Int_DD0 = Int_D0.copy()
    Int_DD = Int_D0.copy()
    Int_D0_prf = np.zeros((2, nImg//2))
    Int_D_prf = Int_D0_prf.copy()
    
    # # PERFECT PSF: elt + Lyot stop propagation, no FPM
    Fld_AA0 = telescope_pupil * lyot_stop
    wf00 = hci.Wavefront(Fld_AA0, lam)
    Fld_DD0 = propagator.forward(wf00)
    Int_DD0 = Fld_DD0.intensity
    norm_peakDD0 = 1./Int_DD0.max()
    Int_DD0 *= norm_peakDD0
    
    # with Lyot FPM: "Lyot propagation with small FPM: soummer 2007 a&a"
    Fld_AA = telescope_pupil
    wf11 = hci.Wavefront(Fld_AA, lam)
    Fld_LL = coro.forward(wf11)
    Fld_DD = propagator.forward(Fld_LL)
    Int_DD = Fld_DD.intensity
    Int_DD *= norm_peakDD0
    
    # LOOP OPD
    for n in range(n_opd):

        phase = (hci.field.NewStyleField(np.exp(
            2j * np.pi * (OPD_arr[:,:,n].ravel() + dfoc + ytlt) / lam),
            pupil_grid))

        Fld_A0 = (telescope_pupil * lyot_stop * phase)
        wf0 = hci.Wavefront(Fld_A0, lam)
        Fld_D0 = propagator.forward(wf0)
        Int_D0 += Fld_D0.intensity

        # CORO
        Fld_A = (telescope_pupil * phase)
        wf1 = hci.Wavefront(Fld_A, lam)
        Fld_L = coro.forward(wf1)
        Fld_D = propagator.forward(Fld_L)
        Int_D += Fld_D.intensity
    
    Int_D0 /= float(n_opd)
    Int_D  /= float(n_opd)

    norm_peakD0 = 1./Int_D0.max()
    Int_D0 *= norm_peakD0
    Int_D  *= norm_peakD0
    
    """
    ### Compute the radial intensity profiles of the images
    """
    # computation of the averaged intensity profiles of the images
    prf = hci.radial_profile(hci.Field(Int_D0, focal_grid), bin_size=px2binSz)
    Int_D0_prf[0,:] = (prf[0][0:nImg//2] / px2binSz - 0.5) * pscale
    Int_D0_prf[1,:] = prf[1][0:nImg//2]

    prf = hci.radial_profile(hci.Field(Int_D, focal_grid), bin_size=px2binSz)
    Int_D_prf[0,:] = (prf[0][0:nImg//2] / px2binSz - 0.5) * pscale
    Int_D_prf[1,:] = prf[1][0:nImg//2]

    shm_opd.close()
    shm_opd.unlink()
    
    return (i,
            Int_D0.reshape((nImg,nImg)), Int_D.reshape((nImg,nImg)),
            Int_D0_prf, Int_D_prf,
            Int_DD0.reshape((nImg,nImg)), Int_DD.reshape((nImg,nImg)))


if __name__ == '__main__':
    
    # datetime of script execution
    donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now
    print('date of now : ', donow)

    # defaults parameters
    nFPM = 100  # Sampling of the coronagraph focal plane mask
    nImg = 400  # last focal plane image size
    nOPD = 4000  # # of AO corrected phase residuals screens
    lam_ref = 1600e-9  # reference wvl
    lam_min = 950e-9  # minimal value of wvl
    lam_itv = 18  # # of intervals
    lam_stp = 50e-9  # interval step
    pscale = 0.3  # image plate scale in last focal plane

    # nominal (=inner/inscribed?) 38.542/ hcipy outer/circumscribed 39.14634
    # D = 39.14634 # in m, __use hcipy outer value__
    D = 38.54  #  main pupil diameter/ elt pupil diameter
    diam = 0.89  # Lyot fractional outer diameter
    obst = 0.35  # Lyot fractional inner/obscuration diameter
    mB = 3.9        # Lyot focal plane mask diameter in lam_ref/D
    lyot_diameter = D * diam

    disp = 5e6  # dispersion mas/mu e.g 8e7 ou 80e6 = 80 mas / 1e-6 m
    fpm_dec_mas = 2 # psf to fpm decentering in mas
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
    
    
    #%%
    '''
    wavelenghts, field of view, pixel to resolution element        
    '''

    lam_lst = np.arange(lam_min,lam_min+(lam_itv+0.5)*lam_stp,lam_stp)
    # lam_lst = [950e-9, 1200e-9, 1350e-9, 1600e-9, 1850e-9]
    nL = len(lam_lst)

    # for achromatic tilt / atmo. dispersion
    dLam = lam_lst - np.array(lam_ref)[None]
    # atmo. induced chromatic tilt
    atlt = dLam * disp * rad2mas

    fov = (nImg * pscale * rad2mas / (lam_ref * 2. / (lyot_diameter)))
    # in lambda reference over lyot_diameter

    px2resEl = nImg / (fov * 2.) # pixels witdh, lambda of reference / lyot_diameter
    
    focal_length = px2resEl * lyot_diameter * 1e-6 / lam_ref # 684.022 # m, nominal?

    # units conversion from meters to mas, * diam to satisfy fov_mas...
    meters2mas = focal_length * diam * rad2mas
    
    # pixel to size of one bin
    px2binSz = pscale * rad2mas * focal_length * diam
    
    # fov in mas
    fov_mas = nImg * pscale
    # field of view in radians
    fov_rdn = fov_mas * rad2mas
    # in multiple of reference lambda (lam_ref) over D
    # mD_ref = fov_rdn / ( lam_ref / D )

    # psf to fpm decentering in radians
    fpm_dec = fpm_dec_mas * rad2mas

    # Lyot stop angular error in radians
    ls_ape_rad = ls_ape * np.pi / 180.
    
    
    #%%
    """
    ### Working directories
    """
    
    if user == 'Alain':
        fdir_dat = Path(usr_base+"/data/").resolve()
        fdir_res   = Path(usr_base+"/results/").resolve()
        fdir_plt   = Path(usr_base+"/plots/").resolve()
    
    fdir_res = fdir_res / donow
    fdir_prfct = fdir_res / 'perfect'
    os.makedirs(fdir_prfct, exist_ok=True)
    
    fdir_plt = fdir_plt / donow


    #%%
    '''
    pupils, tilts, distances to center pixel and grids
    '''

    # input pupil
    fdir_pupil = fdir_dat / 'Pupil'
    pupil = hci.util.read_fits(fdir_pupil / fnm_eltp, extension=1)
    nPup = pupil.shape[0] # supersede input parameter

    pupil_grid = hci.make_pupil_grid(nPup, diameter=D)

    # ELT pupil as a Field
    telescope_pupil = hci.Field(pupil.ravel(),pupil_grid)
    ipup = np.nonzero(telescope_pupil)
    
    # wft = hci.Wavefront(telescope_pupil, lam_lst[0])

    # hci.imshow_field(telescope_pupil, cmap='gray')
    # plt.colorbar()
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.show()
    
    # Lyot stop
    scl = D / nPup # nImg

    lyot_stop = hci.make_shifted_aperture(
        hci.make_rotated_aperture(
            hci.make_obstructed_circular_aperture(
                D * diam, obst/diam, num_spiders=6, spider_width=0.01),
            np.pi/2. - ls_ape_rad), [ls_hoe * scl, ls_voe * scl])(pupil_grid)
    
    t_p = hci.make_shifted_aperture(
        hci.make_rotated_aperture(
            hci.make_linear_interpolator(telescope_pupil,fill_value=0),
            - ls_ape_rad), [ls_hoe * scl, ls_voe * scl])(pupil_grid)
    
    lyot_stop *= t_p
    
    # hci.imshow_field( lyot_stop, cmap='gray')
    # plt.colorbar()
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.show()
    

    #%%
    '''
    zernikes modes for input pupil: defocus and y tilt
    '''

    # ansi=False --> Noll but i is Noll - 1 
    # python index: 1 is x tilt, 2 is y tilt, 3 is focus, 4 y astig... 
    zern_bas = hci.mode_basis.make_zernike_basis(4, D, pupil_grid, ansi=False,
                                                 radial_cutoff=False,
                                                 use_cache=True, cache=None)
    # stop
    
    # for psf to fpm decentering as pupil vertical tilt
    zern_bas[2][:] *= 100.

    # adding vertical sight offset to atmo. dispersion as tilt and convert to m
    ytlt = ( (atlt[:,None] + fpm_dec) * lyot_diameter * zern_bas[2] *
            pupil.ravel() / nPup )

    # for defocus on fpm as pupil error
    zern_bas[3][:] -= np.mean(zern_bas[3][ipup])
    zern_bas[3][:] /= np.std(zern_bas[3][ipup])

    dfoc = (zern_bas[3] * fpm_dfe_elt * 1e-9) * pupil.ravel()
    
    
    #%%
    '''
    focal plane, FPM definition
    '''
    # focal plane grid
    focal_grid = hci.make_focal_grid(q=px2resEl,
                                     num_airy=fov,
                                     pupil_diameter=D,
                                     focal_length=focal_length,
                                     reference_wavelength=lam_ref)

    # FPM definition
    fpm_grid = hci.make_focal_grid(q=nFPM/mB,
                                   num_airy=mB/2.,
                                   pupil_diameter=D,
                                   focal_length=focal_length,
                                   reference_wavelength=lam_ref)
    
    fpm = 1 - hci.make_circular_aperture(
        mB * lam_ref * focal_length / D)(fpm_grid)


    #%%
    '''
    propagators
    '''
    
    propagator = hci.FraunhoferPropagator(pupil_grid, focal_grid,
                                   focal_length=focal_length)
    # toto = propagator(wft)
    # stop
    coro = hci.LyotCoronagraph(pupil_grid,
                               focal_plane_mask=fpm,
                               focal_plane_mask_grid=fpm_grid,
                               lyot_stop=lyot_stop,
                               focal_length=focal_length)
    
        
    #%%
    """
    ### ncpa's for testing and OPD files
    """
        
    if ncpa_rms!=0:
        ncpa = hci.util.read_fits(fdir_dat / 
                                  ('ncpa_ELT_pupil_400_30nm_4096screens.fits'))


    #%%
    '''
    opd data
    '''
    
    opds_dir=(dir_root+'1',  # )
              dir_root+'2', dir_root+'3', dir_root+'4', dir_root+'5',
              dir_root+'6', dir_root+'7', dir_root+'8', dir_root+'9',
              dir_root+'10')


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

            
    #%%
    '''
    iterations over opd data directories
    '''
    
    # if True:
    for ndir in range(len(opds_dir)):
        
        #%%
        """
        ### Read OPD files
        """

        fdir_opd = fdir_dat / opds_dir[ndir]
        
        opd_set = os.path.basename(fdir_opd).split('.')[0]
        
        os.makedirs(fdir_res / opd_set, exist_ok=True)
        os.makedirs(fdir_plt / opd_set, exist_ok=True)
     
        opd_start = 1000
        
        # Filename and path for the OPD maps
        flist_opd = []
        for fnm in os.listdir(fdir_opd):
            if fnm.endswith(".fits"):
                flist_opd.append(fnm)
        
        OPD_arr = hci.util.read_fits(fdir_opd / flist_opd[0]) 
        OPD_arr = OPD_arr[:,:,opd_start:]
        # OPD_arr = OPD_arr[:,:,0:99]
        OPD_arr *= 1e-9
        OPD_arr *= pupil.copy()[:,:,None]

        # OPD_arr *= 0.
        # print(OPD_arr.shape)
        nOPD = OPD_arr.shape[2]
                
        if ncpa_rms!=0:
            for n in range(nOPD):
                OPD_arr[:,:,n] += ncpa[n,:,:] * pupil.copy()

        mp.set_start_method("spawn", force=True)
    
        # --- OPD shared ---
        shm_opd = shared_memory.SharedMemory(create=True,
                                             size=OPD_arr.nbytes)
        shm_opd_arr = np.ndarray(OPD_arr.shape,
                                 dtype=OPD_arr.dtype,
                                 buffer=shm_opd.buf)
        shm_opd_arr[:] = OPD_arr[:]

        nproc = min(cpu_count(), nL)

    
        args_list = [
            (i, lam_lst[i], ytlt[i,:], dfoc, propagator, coro,
             telescope_pupil, lyot_stop, pupil_grid, focal_grid,
             shm_opd.name, OPD_arr.shape, OPD_arr.dtype,
             nImg, pscale, px2binSz)
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
                    'EPUP_FNM':(fnm_eltp,'ELT pupil filename'),
                    'DATE_NOW':(donow,'date of now: script execution date'),
                    'DIR_ROOT':(dir_root,'root of OA residuals data'),
                    'NCPA_FNM':(ncpa_fnm,'filename for ncpa generation')}

        for fpath in fpath_psf_lst:
            for n, k in enumerate(hdr_keys):
                fits.setval(fpath,k,value=hdr_keys[k][0],comment=hdr_keys[k][1])

