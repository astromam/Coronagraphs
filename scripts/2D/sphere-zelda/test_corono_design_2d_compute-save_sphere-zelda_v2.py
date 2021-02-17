#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:20:52 2018

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
nImg2d = 600
Fmax2d = 60
nlam   = 5
bw     = width/wv 
nFPM   = 200
   
kw_aberr     = False
kw_2nddate   = True    
kw_skyobs    = True
kw_aftercorr = False
kw_saxo      = True

#%% 
if kw_aberr is False:
    str_aberr = 'wo_aberr'
    str_date  = ''
    str_obs   = ''
    str_corr  = ''
    str_saxo  = ''
    str_saxoset= ''
    nmap      = 1 
else:
    str_aberr = 'with_aberr'    
    str_date  = '2018-04-01'
    str_obs   = 'internal'
    str_corr  = 'before_correction'
    imap0     = 0
    nmap      = 1 
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
        nmap     = 1

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
            fname_ZELDAmapnm3d = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
    else:
        fname_OPDmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'        
        if kw_2nddate is True:        
            fname_ZELDAmapnm3d = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'
    
    if kw_saxo is True and kw_2nddate is True:    
        fname_SAXOmapnm3d = '2018-04-04T03_06_15-saxo_residual_turbulence.fits'
        if kw_aftercorr is True:
            fname_SAXOmapnm3d = '2018-04-04T03_12_50-saxo_residual_turbulence.fits'

        
fpath_Apod2d     = fdir_pupils / fname_Apod2d
fpath_Ampmap2d   = fdir_pupils / fname_Ampmap2d

if kw_aberr is True:
    fpath_ZELDAmapnm3d = fdir_zelda  / fname_ZELDAmapnm3d   
    if kw_saxo is True and kw_2nddate is True:
        fpath_SAXOmapnm3d = fdir_saxo / fname_SAXOmapnm3d
    
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d

#%%

fname_direct_poly_img_t     = 'direct_poly_img_nmap={0:05d}_t.fits'.format(nmap)
fname_corono_poly_img_t     = 'corono_poly_img_nmap={0:05d}_t.fits'.format(nmap)
fpath_direct_poly_img_t     = fdir_results / fname_direct_poly_img_t
fpath_corono_poly_img_t     = fdir_results / fname_corono_poly_img_t

fname_direct_poly_img_f     = 'direct_poly_img_nmap={0:05d}_f.fits'.format(nmap)
fname_corono_poly_img_f     = 'corono_poly_img_nmap={0:05d}_f.fits'.format(nmap)
fpath_direct_poly_img_f     = fdir_results / fname_direct_poly_img_f
fpath_corono_poly_img_f     = fdir_results / fname_corono_poly_img_f

fname_direct_poly_prf_avg_f = 'direct_poly_prf_nmap={0:05d}_avg_f.fits'.format(nmap)
fname_corono_poly_prf_avg_f = 'corono_poly_prf_nmap={0:05d}_avg_f.fits'.format(nmap)
fname_direct_poly_prf_std_f = 'direct_poly_prf_nmap={0:05d}_std_f.fits'.format(nmap)
fname_corono_poly_prf_std_f = 'corono_poly_prf_nmap={0:05d}_std_f.fits'.format(nmap)
fpath_direct_poly_prf_avg_f = fdir_results / fname_direct_poly_prf_avg_f
fpath_corono_poly_prf_avg_f = fdir_results / fname_corono_poly_prf_avg_f
fpath_direct_poly_prf_std_f = fdir_results / fname_direct_poly_prf_std_f
fpath_corono_poly_prf_std_f = fdir_results / fname_corono_poly_prf_std_f


#%% Entrance pupil
Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)

#%% Apodization
Apod2d = fits.getdata(fpath_Apod2d)

#%% Amplitude errors
if kw_aberr is True:
    Ampmap2d = fits.getdata(fpath_Ampmap2d)

#%% Phase errors
if kw_aberr is True:
    ZELDAmapnm3d = fits.getdata(fpath_ZELDAmapnm3d)

    if kw_saxo is True and kw_2nddate is True:
        SAXOmapnm3d_tmp = fits.getdata(fpath_SAXOmapnm3d)
        nsaxo_all = len(SAXOmapnm3d_tmp)        
        pupil_tmp = aperture.sphere_saxo_pupil()
        pupil = np.round(imutils.scale(pupil_tmp, 0, new_dim=(384,384), method='interp'))

        # rescale NCPA map
        SAXOmapnm3d = []
        for i in range(nmap):
            SAXOmapnm3d.append(imutils.scale(SAXOmapnm3d_tmp[i], 0, new_dim=(384,384), method='interp'))
        
        SAXOmapnm3d = np.asarray(SAXOmapnm3d)


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

direct_poly_img_f = np.zeros((nImg2d, nImg2d))
corono_poly_img_f = np.zeros((nImg2d, nImg2d))

direct_poly_prf_avg_f = np.zeros((nImg2d//2))
corono_poly_prf_avg_f = np.zeros((nImg2d//2))
direct_poly_prf_std_f = np.zeros((nImg2d//2))
corono_poly_prf_std_f = np.zeros((nImg2d//2))

#%%

for imap in range(nmap):
    t0 = time.time()
    if kw_aberr is True:
        OPDmap2d = ZELDAmapnm3d[imap0]*1e-9
        if kw_saxo is True and kw_2nddate is True:
            OPDmap2d += SAXOmapnm3d[imap]*1e-9
        
        params   = coro.update_params(params, OPDmap2d = OPDmap2d, Ampmap2d = Ampmap2d, LyotStop2d = LyotStop2d)
        corono0  = coro.design.APLC2d(**params)
    else:
        params  = coro.update_params(params, OPDmap2d = None, Ampmap2d = None, LyotStop2d = LyotStop2d)
        corono0 = coro.design.APLC2d(**params)    
                
    direct_poly_img_t[imap] = corono0.compute_direct_intensity_2d(Apod2d)
    corono_poly_img_t[imap] = corono0.compute_corono_intensity_2d(Apod2d)

    t1 = time.time()
    print('map {1}/{2}, computation time: {0:.2f}s'.format(t1-t0, imap+1, nmap))

if nmap > 1:
    direct_poly_img_f = np.mean(direct_poly_img_t, axis=0)
    corono_poly_img_f = np.mean(corono_poly_img_t, axis=0)
else:
    direct_poly_img_f = direct_poly_img_t[0]
    corono_poly_img_f = corono_poly_img_t[0]
    
direct_poly_prf_avg_f, rad_direct = imutils.profile(direct_poly_img_f, type='mean')
corono_poly_prf_avg_f, rad_corono = imutils.profile(corono_poly_img_f, type='mean')
direct_poly_prf_std_f, rad_direct = imutils.profile(direct_poly_img_f, type='std')
corono_poly_prf_std_f, rad_corono = imutils.profile(corono_poly_img_f, type='std')


#%% saving of the images
fits.writeto(fpath_direct_poly_img_t, direct_poly_img_t, overwrite=True)
fits.writeto(fpath_corono_poly_img_t, corono_poly_img_t, overwrite=True)

fits.writeto(fpath_direct_poly_img_f, direct_poly_img_f, overwrite=True)
fits.writeto(fpath_corono_poly_img_f, corono_poly_img_f, overwrite=True)

fits.writeto(fpath_direct_poly_prf_avg_f, direct_poly_prf_avg_f, overwrite=True)
fits.writeto(fpath_corono_poly_prf_avg_f, corono_poly_prf_avg_f, overwrite=True)
fits.writeto(fpath_direct_poly_prf_std_f, direct_poly_prf_std_f, overwrite=True)
fits.writeto(fpath_corono_poly_prf_std_f, corono_poly_prf_std_f, overwrite=True)

#%%
#
##%%
#import pylab as pl
#
#map_t = [ZELDAmapnm3d[imap0], SAXOmapnm3d[0], OPDmap2d*1e9]
#label_t = ['ZELDA map', 'SAXO map', 'Total map']
#ncase = np.shape(map_t)[0]
#
#
#if ncase <= 10: 
#    f2 = pl.figure(30, figsize=(8,4.5))
#    pl.clf()
#    for i, val in enumerate(map_t):
#        exec('ax{0} = f2.add_subplot(1,{1},{0})'.format(i+1,ncase))
#        if i < ncase: 
#            exec('im = ax{0}.imshow(val, cmap = "inferno", vmin=-300, vmax=300)'.format(i+1))
#            exec('ax{0}.text(nPup/2, 0.1*nPup, "{1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,label_t[i]))
#        exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
#        exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))
#        
#    f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                        wspace=0.02, hspace=0.02)
#    
#    f2.subplots_adjust(right=0.8)
#    cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
#    cbar    = f2.colorbar(im, cax=cbar_ax)
#    cbar.ax.set_ylabel('phase map in nm', rotation=270, labelpad = 10)
##        pl.savefig(str(fpath_image_plane_disp), transparent=True)
#    pl.tight_layout()
#    pl.show()
#   
#
##%%
#from pyzelda import ztools
#psd_2d_zel_tmp, psd_1d_zel_tmp, freq_zel = ztools.compute_psd(ZELDAmapnm3d[imap0], mask = Pupil2d)
##psd_2d_sax_tmp, psd_1d_sax_tmp, freq_sax = ztools.compute_psd(SAXOmapnm3d_tmp[0], mask = pupil_tmp)
#psd_2d_sax_tmp, psd_1d_sax_tmp, freq_sax = ztools.compute_psd(SAXOmapnm3d[0], mask = Pupil2d)
#psd_2d_sum_tmp, psd_1d_sum_tmp, freq_sum = ztools.compute_psd(OPDmap2d*1e9, mask = Pupil2d)
#
#dim_zel = np.shape(psd_2d_zel_tmp)[0]
#dim_sax = np.shape(psd_2d_sax_tmp)[0]
#dim_sum = np.shape(psd_2d_sum_tmp)[0]
#
##nSax = np.shape(SAXOmapnm3d_tmp)[1]
#nSax = np.shape(SAXOmapnm3d)[1]
#
#psd_2d_zel = psd_2d_zel_tmp[(dim_zel-nPup)//2:(dim_zel+nPup)//2, (dim_zel-nPup)//2:(dim_zel+nPup)//2]
#psd_2d_sax = psd_2d_sax_tmp[(dim_sax-nSax)//2:(dim_sax+nSax)//2, (dim_sax-nSax)//2:(dim_sax+nSax)//2]
#psd_2d_sum = psd_2d_sum_tmp[(dim_sum-nPup)//2:(dim_sum+nPup)//2, (dim_sum-nPup)//2:(dim_sum+nPup)//2]
#
##%%
#
#psd_t = [psd_2d_zel, psd_2d_sax, psd_2d_sum]
#label_t = ['zelda map psd', 'saxo map psd', 'total map psd']
#
#fname_t = ['zelda_map_psd.pdf', 'saxo_map_psd.pdf', 'total_map_psd.pdf']
#
#if ncase <= 10: 
#    f2 = pl.figure(31, figsize=(8,4.5))
#    pl.clf()
#    for i, val in enumerate(psd_t):
#        exec('ax{0} = f2.add_subplot(1,{1},{0})'.format(i+1,ncase))
#        if i < ncase: 
#            exec('im = ax{0}.imshow(np.log10(val), cmap = "inferno", vmin=-3, vmax=2)'.format(i+1))
#            exec('ax{0}.text(nPup/2, 0.1*nPup, "{1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,label_t[i]))
#        exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
#        exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))
#      
#        f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                            wspace=0.02, hspace=0.02) 
#        
#        f2.subplots_adjust(right=0.8)
#        cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
#        cbar    = f2.colorbar(im, cax=cbar_ax)
#        cbar.ax.set_ylabel('psd in log scale [nm RMS/(cycle/pup)]', rotation=270, labelpad = 10)
#    #    pl.savefig(str(fdir_results / fname_t[i]), transparent=True)
#        pl.tight_layout()
#        pl.show()
#
##%%
#
#psd_zel_avg_f, rad_zel = imutils.profile(psd_2d_zel, type='mean')
#psd_zel_std_f, rad_zel = imutils.profile(psd_2d_zel, type='std')
#psd_zel_avg_f[psd_zel_avg_f <= 1e-10] = 0
#psd_zel_std_f[psd_zel_std_f <= 1e-10] = 0
#
#psd_sax_avg_f, rad_sax = imutils.profile(psd_2d_sax, type='mean')
#psd_sax_std_f, rad_sax = imutils.profile(psd_2d_sax, type='std')
#psd_sax_avg_f[psd_sax_avg_f <= 1e-10] = 0
#psd_sax_std_f[psd_sax_std_f <= 1e-10] = 0
#
#psd_sum_avg_f, rad_sum = imutils.profile(psd_2d_sum, type='mean')
#psd_sum_std_f, rad_sum = imutils.profile(psd_2d_sum, type='std')
#psd_sum_avg_f[psd_sum_avg_f <= 1e-10] = 0
#psd_sum_std_f[psd_sum_std_f <= 1e-10] = 0
#
##%%
##pl.figure(20)
##pl.clf()
##pl.imshow(pupil_tmp)
#
#fname = 'avg_psd_prf.pdf'
#fpath = fdir_results / fname
#
#pl.figure(21)
#pl.clf()
#pl.plot(rad_zel*nPup/dim_zel, np.log10(psd_zel_avg_f), label='zelda')
#pl.plot(rad_sax*nPup/dim_sax, np.log10(psd_sax_avg_f), label='saxo')
#pl.plot(rad_sum*nPup/dim_sum, np.log10(psd_sum_avg_f), label='total')
#pl.title('psd mean profile')
#pl.xlabel('spatial frequency in cycle/pup')
#pl.ylabel('psd in log scale [nm RMS/(cycle/pup)]')
#pl.savefig(str(fpath))
#pl.legend()
#
#
#fname = 'std_psd_prf.pdf'
#fpath = fdir_results / fname
#
#pl.figure(22)
#pl.clf()
#pl.plot(rad_zel*nPup/dim_zel, np.log10(psd_zel_std_f), label='zelda')
#pl.plot(rad_sax*nPup/dim_sax, np.log10(psd_sax_std_f), label='saxo')
#pl.plot(rad_sum*nPup/dim_sum, np.log10(psd_sum_std_f), label='total')
#pl.title('psd std profile')
#pl.xlabel('spatial frequency in cycle/pup')
#pl.ylabel('psd in log scale [nm RMS/(cycle/pup)]')
#pl.savefig(str(fpath))
#pl.legend()
#
##    
##%%
#print('ok')
