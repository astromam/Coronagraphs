#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:58:56 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%% Initialization
import numpy as np
import os
import time
from pyzelda.utils import aperture, imutils
from pathlib import Path
from astropy.io import fits
import corono as coro

#%% APLC2d tests
"""
tests on APLC 2d class
"""
corono_name   = 'APLC' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = True
CtrBtwnPix2 = False
SymPupil2d  = False
cDarkHole   = 6

wv        = 1.593e-6
width     = 52e-9

dAper     = 8

mas2rad   = np.pi/(180.*3600)

rMask_m   = 287e-6/2.
Fratio    = 40

rMask  = rMask_m/(wv*Fratio)
rMask_mas = 1000.*rMask * (wv/dAper)/mas2rad
print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))

rho0   = 5.
rho1   = 20.
nPup   = 384
nImg2d = 256
Fmax2d = nImg2d/(2*(wv/950e-9))
nlam   = 11
bw     = 0.2#width/wv 
nFPM   = 200
   
kw_aberr     = False
kw_2nddate   = True    
kw_skyobs    = True
kw_aftercorr = False
kw_saxo      = True

#%% 
if kw_aberr is False:
    str_aberr = 'wo_aberr_with_irdis_plate_scale'
    str_date  = ''
    str_obs   = ''
    str_corr  = ''
    str_saxo  = ''
    str_saxoset= ''
    imap0     = 0
    nmap      = 1 
else:
    str_aberr = 'with_aberr'    
    str_date  = '2018-04-01'
    str_obs   = 'internal'
    str_corr  = 'before_correction'
    str_saxo_tmp  = 'wo_saxo'
    if kw_2nddate is True:
        str_date = '2018-04-03'
    if kw_skyobs is True:
        str_obs  = 'sky'
    if kw_aftercorr is True:
        str_corr = 'after_correction'
        imap0    = 3
    if kw_saxo is True and kw_2nddate is True:
        str_saxo = 'with_saxo'

fdir = Path('../../').resolve()

fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_date / str_obs / str_saxo / str_corr  
fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
fdir_saxo    = fdir / 'data' / '2D' / 'ZELDA' / '2018-04-03'


#%%
if not os.path.exists(fdir_results):
    os.makedirs(fdir_results)

fname_Apod2d     = 'SPHERE_APO1_field_transmission_map.fits'
#fname_Apod2d = 'sphere_pupil_APO1_BH.fits'
fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
fname_LyotStop2d = 'sphere_stop_ST_ALC2.fits'

if kw_aberr is True:
    if kw_skyobs is True:
        fname_OPDmapnm3d = '2018-04-01_night_ncpa_loop_700modes_5_ncpa_loop_opd.fits'
        if kw_2nddate is True:
            fname_OPDmapnm3d = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
    else:
        fname_OPDmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'        
        if kw_2nddate is True:        
            fname_OPDmapnm3d = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'
    
    if kw_saxo is True and kw_2nddate is True:    
        fname_SAXOmapnm3d = '2018-04-04T03_06_15-saxo_residual_turbulence.fits'
        if kw_aftercorr is True:
            fname_SAXOmapnm3d = '2018-04-04T03_12_50-saxo_residual_turbulence.fits'

        
fpath_Apod2d     = fdir_pupils / fname_Apod2d
fpath_Ampmap2d   = fdir_pupils / fname_Ampmap2d

if kw_aberr is True:
    fpath_OPDmapnm3d = fdir_zelda  / fname_OPDmapnm3d   
    if kw_saxo is True and kw_2nddate is True:
        fpath_SAXOmapnm3d = fdir_saxo / fname_SAXOmapnm3d
    
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d

#%%

fname_direct_poly_img_t     = 'direct_poly_img_t.fits'
fname_corono_poly_img_t     = 'corono_poly_img_t.fits'
fpath_direct_poly_img_t     = fdir_results / fname_direct_poly_img_t
fpath_corono_poly_img_t     = fdir_results / fname_corono_poly_img_t

fname_direct_poly_prf_avg_t = 'direct_poly_prf_avg_t.fits'
fname_corono_poly_prf_avg_t = 'corono_poly_prf_avg_t.fits'
fname_direct_poly_prf_std_t = 'direct_poly_prf_std_t.fits'
fname_corono_poly_prf_std_t = 'corono_poly_prf_std_t.fits'
fpath_direct_poly_prf_avg_t = fdir_results / fname_direct_poly_prf_avg_t
fpath_corono_poly_prf_avg_t = fdir_results / fname_corono_poly_prf_avg_t
fpath_direct_poly_prf_std_t = fdir_results / fname_direct_poly_prf_std_t
fpath_corono_poly_prf_std_t = fdir_results / fname_corono_poly_prf_std_t

fname_direct_mono_img_t     = 'direct_mono_img_t.fits'
fname_corono_mono_img_t     = 'corono_mono_img_t.fits'
fpath_direct_mono_img_t     = fdir_results / fname_direct_mono_img_t
fpath_corono_mono_img_t     = fdir_results / fname_corono_mono_img_t

fname_direct_mono_prf_avg_t = 'direct_mono_prf_avg_t.fits'
fname_corono_mono_prf_avg_t = 'corono_mono_prf_avg_t.fits'
fname_direct_mono_prf_std_t = 'direct_mono_prf_std_t.fits'
fname_corono_mono_prf_std_t = 'corono_mono_prf_std_t.fits'
fpath_direct_mono_prf_avg_t = fdir_results / fname_direct_mono_prf_avg_t
fpath_corono_mono_prf_avg_t = fdir_results / fname_corono_mono_prf_avg_t
fpath_direct_mono_prf_std_t = fdir_results / fname_direct_mono_prf_std_t
fpath_corono_mono_prf_std_t = fdir_results / fname_corono_mono_prf_std_t

fname_direct_mono_lyot_re_t = 'direct_mono_lyot_t_re.fits'
fname_direct_mono_lyot_im_t = 'direct_mono_lyot_t_im.fits'
fname_corono_mono_lyot_re_t = 'corono_mono_lyot_t_re.fits'
fname_corono_mono_lyot_im_t = 'corono_mono_lyot_t_im.fits'
fpath_direct_mono_lyot_re_t = fdir_results / fname_direct_mono_lyot_re_t
fpath_direct_mono_lyot_im_t = fdir_results / fname_direct_mono_lyot_im_t
fpath_corono_mono_lyot_re_t = fdir_results / fname_corono_mono_lyot_re_t
fpath_corono_mono_lyot_im_t = fdir_results / fname_corono_mono_lyot_im_t

#%% Entrance pupil
Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)

#%% Apodization
Apod2d = fits.getdata(fpath_Apod2d)

#%% Amplitude errors
if kw_aberr is True:
    Ampmap2d = fits.getdata(fpath_Ampmap2d)

#%% Phase errors
if kw_aberr is True:
    OPDmapnm3d = fits.getdata(fpath_OPDmapnm3d)
#    nmap = len(OPDmapnm3d)
    nmap = 1

    if kw_saxo is True and kw_2nddate is True:
        SAXOmapnm3d_tmp = fits.getdata(fpath_SAXOmapnm3d)
        nsaxo_all = len(SAXOmapnm3d_tmp)
    
        nmap = 10
        
        pupil_tmp = aperture.sphere_saxo_pupil()
        pupil = np.round(imutils.scale(pupil_tmp, 0, new_dim=(384,384), method='interp'))

        # rescale NCPA map
        SAXOmapnm3d = []
        for i in range(nmap):
            SAXOmapnm3d.append(imutils.scale(SAXOmapnm3d_tmp[i], 0, new_dim=(384,384), method='interp'))
        
        SAXOmapnm3d = np.asarray(SAXOmapnm3d)

# import pylab as pl
#pl.figure(0)
#pl.imshow(OPDmapnm3d[1])



#%% Lyot Stop
LyotStop2d = fits.getdata(fpath_LyotStop2d)

#%%
if corono_name != 'APLC':
    raise NameError('Check the name of the coronagraph!')

params = coro.to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                 rMask = rMask,
                 SymPupil2d = SymPupil2d, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                 CtrBtwnPix=CtrBtwnPix,
                 CtrBtwnPix2 = CtrBtwnPix2, 
                 nlam=nlam, bw = bw, wv =wv,
                 rho0   = rho0, rho1 = rho1, cDarkHole = cDarkHole,
                 OPDmap2d = None, Ampmap2d = None)

#%%
direct_poly_img_t = np.zeros((nmap, nImg2d, nImg2d))
corono_poly_img_t = np.zeros((nmap, nImg2d, nImg2d))
direct_mono_img_t = np.zeros((nmap, nlam, nImg2d, nImg2d))
corono_mono_img_t = np.zeros((nmap, nlam, nImg2d, nImg2d))

direct_poly_prf_avg_t = np.zeros((nmap, nImg2d//2))
corono_poly_prf_avg_t = np.zeros((nmap, nImg2d//2))
direct_mono_prf_avg_t = np.zeros((nmap, nlam, nImg2d//2))
corono_mono_prf_avg_t = np.zeros((nmap, nlam, nImg2d//2))

direct_poly_prf_std_t = np.zeros((nmap, nImg2d//2))
corono_poly_prf_std_t = np.zeros((nmap, nImg2d//2))
direct_mono_prf_std_t = np.zeros((nmap, nlam, nImg2d//2))
corono_mono_prf_std_t = np.zeros((nmap, nlam, nImg2d//2))

direct_mono_lyot_t = np.zeros((nmap, nlam, nPup, nPup), dtype='complex128')
corono_mono_lyot_t = np.zeros((nmap, nlam, nPup, nPup), dtype='complex128')

#%%

for imap in range(nmap):
    t0 = time.time()
    if kw_aberr is True:
        OPDmap2d = OPDmapnm3d[imap0]*1e-9
        if kw_saxo is True and kw_2nddate is True:
            OPDmap2d += SAXOmapnm3d[imap]*1e-9
        
        params   = coro.update_params(params, OPDmap2d = OPDmap2d, Ampmap2d = Ampmap2d, LyotStop2d = LyotStop2d)
        corono0  = coro.design.APLC2d(**params)
    else:
        params  = coro.update_params(params, OPDmap2d = None, Ampmap2d = None, LyotStop2d = LyotStop2d)
        corono0 = coro.design.APLC2d(**params)    
                
    direct_poly_img_t[imap] = corono0.compute_direct_intensity_2d(Apod2d)
    corono_poly_img_t[imap] = corono0.compute_corono_intensity_2d(Apod2d)
    direct_mono_img_t[imap] = corono0.compute_direct_intensity_2d(Apod2d, poly=False)
    corono_mono_img_t[imap] = corono0.compute_corono_intensity_2d(Apod2d, poly=False)
    
    direct_poly_prf_avg_t[imap], rad_direct = imutils.profile(direct_poly_img_t[imap], type='mean')
    corono_poly_prf_avg_t[imap], rad_corono = imutils.profile(corono_poly_img_t[imap], type='mean')
    direct_poly_prf_std_t[imap], rad_direct = imutils.profile(direct_poly_img_t[imap], type='std')
    corono_poly_prf_std_t[imap], rad_corono = imutils.profile(corono_poly_img_t[imap], type='std')
    for i in range(corono0.nlam):
        direct_mono_prf_avg_t[imap, i], rad_direct = imutils.profile(direct_mono_img_t[imap, i], type='mean')
        corono_mono_prf_avg_t[imap, i], rad_corono = imutils.profile(corono_mono_img_t[imap, i], type='mean')
        direct_mono_prf_std_t[imap, i], rad_direct = imutils.profile(direct_mono_img_t[imap, i], type='std')
        corono_mono_prf_std_t[imap, i], rad_corono = imutils.profile(corono_mono_img_t[imap, i], type='std')

    direct_mono_lyot_t[imap] = corono0.compute_direct_lyot_field_2d(Apod2d)
    corono_mono_lyot_t[imap] = corono0.compute_corono_lyot_field_2d(Apod2d)
    t1 = time.time()
    print('map {1}/{2}, computation time: {0:.2f}s'.format(t1-t0, imap+1, nmap))


#%% saving of the images
fits.writeto(fpath_direct_poly_img_t, direct_poly_img_t, overwrite=True)
fits.writeto(fpath_corono_poly_img_t, corono_poly_img_t, overwrite=True)

fits.writeto(fpath_direct_poly_prf_avg_t, direct_poly_prf_avg_t, overwrite=True)
fits.writeto(fpath_corono_poly_prf_avg_t, corono_poly_prf_avg_t, overwrite=True)
fits.writeto(fpath_direct_poly_prf_std_t, direct_poly_prf_std_t, overwrite=True)
fits.writeto(fpath_corono_poly_prf_std_t, corono_poly_prf_std_t, overwrite=True)

fits.writeto(fpath_direct_mono_img_t, direct_mono_img_t, overwrite=True)
fits.writeto(fpath_corono_mono_img_t, corono_mono_img_t, overwrite=True)

fits.writeto(fpath_direct_mono_prf_avg_t, direct_mono_prf_avg_t, overwrite=True)
fits.writeto(fpath_corono_mono_prf_avg_t, corono_mono_prf_avg_t, overwrite=True)
fits.writeto(fpath_direct_mono_prf_std_t, direct_mono_prf_std_t, overwrite=True)
fits.writeto(fpath_corono_mono_prf_std_t, corono_mono_prf_std_t, overwrite=True)
 
fits.writeto(fpath_direct_mono_lyot_re_t, direct_mono_lyot_t.real, overwrite=True)
if kw_aberr is True:
    fits.writeto(fpath_direct_mono_lyot_im_t, direct_mono_lyot_t.imag, overwrite=True)
fits.writeto(fpath_corono_mono_lyot_re_t, corono_mono_lyot_t.real, overwrite=True)
fits.writeto(fpath_corono_mono_lyot_im_t, corono_mono_lyot_t.imag, overwrite=True)    
#    
#%%
print('ok')



#%% tests
#if kw_saxo is True:        
#    import pylab as pl
#    
#    pl.figure(1)
#    pl.clf()
#    pl.imshow(pupil_tmp, cmap = 'inferno')
#    
#    pl.figure(2)
#    pl.clf()
#    pl.imshow(SAXOmapnm3d_tmp[0], cmap = 'inferno')
#    
#    pupil_tmp = aperture.sphere_saxo_pupil()
#    idx_pupil_tmp = (pupil_tmp != 0)
#    
#    val_tmp_t = []
#    for i in range(nsaxo_all):
#        saxo_map_tmp = SAXOmapnm3d_tmp[i]  
#        val_tmp_t.append(np.std(saxo_map_tmp[idx_pupil_tmp]))
#    #    print('tmp val: {0} nm rms'.format(val_tmp))
#    
#    pl.figure(3)
#    pl.clf()
#    pl.imshow(pupil, cmap = 'inferno')
#    
#    pl.figure(4)
#    pl.clf()
#    pl.imshow(SAXOmapnm3d[0], cmap = 'inferno')
#    
#    idx_pupil = (pupil != 0)
#    
#    val_t = []
#    for i in range(nmap):
#        saxo_map = SAXOmapnm3d[i]
#        val = np.std(saxo_map[idx_pupil])
#    #    print('tmp val: {0} nm rms'.format(val))
#
#
#%%
#if kw_saxo is True:
#    val_tmp2_t = np.asarray(val_tmp_t)
#    frame_rate = 1380.
#    
#    time_t = (1./frame_rate)*np.arange(nsaxo_all)
#    
#    fname = 'saxo_data'
#    
#    fname = '2018-04-04T03_06_15-saxo_residual_turbulence_plt.pdf'
#    str_dataset = '- 1st data set'
#    if kw_aftercorr is True:
#        fname = '2018-04-04T03_12_50-saxo_residual_turbulence_plt.pdf'
#        str_dataset = '- 2nd data set'
#    
#    val_mean   = np.mean(val_tmp2_t)
#    val_std    = np.std(val_tmp2_t) 
#    val_median = np.median(val_tmp2_t)
#    
#    print(r'mean nm rms wfe: {0} \pm {1}'.format(val_mean, val_std))
#    print(r'median nm rms wfe: {0}'.format(val_median))
#       
#    fpath =  fdir_saxo / fname 
#    
#    pl.figure(5)
#    pl.clf()
#    pl.plot(time_t, val_tmp2_t)
#    pl.xlabel('Time [s]')
#    pl.ylabel('WFE [nm rms]')
#    pl.ylim(75,305)
#    pl.title('SAXO data ' + str_dataset)
#    pl.axhline(val_mean, xmin=time_t.min(), xmax=time_t.max(), 
#               linewidth=1, color='r', linestyle='-')
#    pl.axhline(val_mean+val_std, xmin=time_t.min(), xmax=time_t.max(), 
#               linewidth=1, color='r', linestyle='--')
#    pl.axhline(val_mean-val_std, xmin=time_t.min(), xmax=time_t.max(), 
#               linewidth=1, color='r', linestyle='--')
#    
#    
#    pl.tight_layout()
#    pl.savefig(str(fpath))
