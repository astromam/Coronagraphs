#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:27:14 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%% Initialization
import numpy as np
import os
import pylab as pl
from pathlib import Path
from astropy.io import fits
from pyzelda.utils import aperture, imutils, zernike
import corono as coro

import time

#%% APLC2d tests
"""
tests on APLC 2d class
"""
corono_name = 'APLC' # 'SP' or 'APLC' or DZPM
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
nlam   = 11
bw     = 0.20#width/wv 
nFPM   = 200

kw_correction = True

# DO NOT CHANGE THESE VALUES
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
    imap0     = 0
else:
    str_aberr = 'with_aberr'    
    str_date  = '2018-04-01'
    str_obs   = 'internal'
    str_corr  = 'before_correction'
    imap0     = 0
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

fdir_plots    = fdir / 'results' / '2D' / 'plots' / 'SPHERE' / str_aberr / str_date / str_obs / str_corr  
fdir_pupimages = fdir_results / 'pupimages'

#%%
if not os.path.exists(fdir_results):
    os.makedirs(fdir_results)
    
if not os.path.exists(fdir_plots):
    os.makedirs(fdir_plots)    

if not os.path.exists(fdir_pupimages):
    os.makedirs(fdir_pupimages)  

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
    if kw_saxo is True:
        fpath_SAXOmapnm3d = fdir_saxo / fname_SAXOmapnm3d
    
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d

#%%

label_lst = ['no errors']
ncase = len(label_lst)

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

fname_direct_mono_lyot_re_t     = 'direct_mono_lyot_t_re.fits'
fname_direct_mono_lyot_im_t     = 'direct_mono_lyot_t_im.fits'
fname_corono_mono_lyot_re_t     = 'corono_mono_lyot_t_re.fits'
fname_corono_mono_lyot_im_t     = 'corono_mono_lyot_t_im.fits'
fpath_direct_mono_lyot_re_t     = fdir_results / fname_direct_mono_lyot_re_t
fpath_direct_mono_lyot_im_t     = fdir_results / fname_direct_mono_lyot_im_t
fpath_corono_mono_lyot_re_t     = fdir_results / fname_corono_mono_lyot_re_t
fpath_corono_mono_lyot_im_t     = fdir_results / fname_corono_mono_lyot_im_t

#%%
fname_image_plane_plot   = 'corono_poly_prf_std_t_OPDmap={0}_plot.pdf'.format(imap0)
fname_image_plane_mono_plot   = 'corono_poly_prf_std_t_OPDmap={0}_mono_plot.pdf'.format(imap0)
fname_image_allmaps_plot = 'corono_poly_prf_std_t_OPDmap=all_plot.pdf'
fname_image_allmaps_plot_comp = 'corono_poly_prf_std_t_OPDmap=all_plot_comp.pdf'
fname_image_plane_disp   = 'corono_poly_img_t_OPDmap={0}_disp.pdf'.format(imap0)
fname_pupil_plane_disp   = 'corono_poly_lyot_t_OPDmap={0}_disp.pdf'.format(imap0)

fpath_image_plane_plot   = fdir_plots / fname_image_plane_plot
fpath_image_plane_mono_plot   = fdir_plots / fname_image_plane_mono_plot
fpath_image_allmaps_plot = fdir_plots / fname_image_allmaps_plot
fpath_image_allmaps_plot_comp = fdir_plots / fname_image_allmaps_plot_comp
fpath_image_plane_disp   = fdir_plots / fname_image_plane_disp
fpath_pupil_plane_disp   = fdir_plots / fname_pupil_plane_disp


#%% Entrance pupil
Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)

fpath = fdir_pupimages / 'Aperture.pdf'

pl.figure(1)
pl.clf()
pl.imshow(Pupil2d, cmap = 'inferno')
pl.title('Entrance pupil')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#%% Apodization
Apod2d = fits.getdata(fpath_Apod2d)

fpath = fdir_pupimages / 'Apodizer.pdf'

pl.figure(2)
pl.clf()
pl.imshow(Apod2d*Pupil2d, cmap = 'inferno')
pl.title('Apodized entrance pupil')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#%% Amplitude errors
fpath = fdir_pupimages / 'AmplMap.pdf'
Ampmap2d = fits.getdata(fpath_Ampmap2d)

pl.figure(3)
pl.clf()
pl.imshow(Pupil2d*Ampmap2d, cmap = 'inferno')
pl.title('Amplitude map')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#%% Phase errors
nmap = 1
if kw_aberr is True:
    OPDmapnm3d = fits.getdata(fpath_OPDmapnm3d)
    OPDmapnm2d = OPDmapnm3d[imap0]
    OPDmap2d   = OPDmapnm2d*1e-9
        
    if kw_saxo is True:
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

        pl.figure(6)
        pl.clf()
        pl.imshow(SAXOmapnm3d[0])
        pl.title('Saxo phase map')

    pl.figure(4)
    pl.clf()
    pl.imshow(OPDmap2d, cmap = 'inferno')
    pl.title('Phase map')

#%% Lyot Stop
LyotStop2d = fits.getdata(fpath_LyotStop2d)

fpath = fdir_pupimages / 'LyotStop.pdf'

pl.figure(5)
pl.clf()
pl.imshow(LyotStop2d, cmap = 'inferno')
pl.title('Lyot stop')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

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
corono00 = coro.design.APLC2d(**params)

#%%
direct_poly_img_t = fits.getdata(fpath_direct_poly_img_t)
corono_poly_img_t = fits.getdata(fpath_corono_poly_img_t)

direct_poly_prf_avg_t = fits.getdata(fpath_direct_poly_prf_avg_t)
corono_poly_prf_avg_t = fits.getdata(fpath_corono_poly_prf_avg_t)
direct_poly_prf_std_t = fits.getdata(fpath_direct_poly_prf_std_t)
corono_poly_prf_std_t = fits.getdata(fpath_corono_poly_prf_std_t)

direct_mono_img_t = fits.getdata(fpath_direct_mono_img_t)
corono_mono_img_t = fits.getdata(fpath_corono_mono_img_t)

direct_mono_prf_avg_t = fits.getdata(fpath_direct_mono_prf_avg_t)
if kw_aberr is True:
    corono_mono_prf_avg_t = fits.getdata(fpath_corono_mono_prf_avg_t)
direct_mono_prf_std_t = fits.getdata(fpath_direct_mono_prf_std_t)
corono_mono_prf_std_t = fits.getdata(fpath_corono_mono_prf_std_t)

#%%    
direct_mono_lyot_re_t = fits.getdata(fpath_direct_mono_lyot_re_t)
if kw_aberr is True:
    direct_mono_lyot_im_t = fits.getdata(fpath_direct_mono_lyot_im_t)
corono_mono_lyot_re_t = fits.getdata(fpath_corono_mono_lyot_re_t)
corono_mono_lyot_im_t = fits.getdata(fpath_corono_mono_lyot_im_t) 

direct_mono_lyot_t = direct_mono_lyot_re_t*1
if kw_aberr is True:
    direct_mono_lyot_t  = 1j*corono_mono_lyot_im_t + direct_mono_lyot_re_t
corono_mono_lyot_t  = 1j*corono_mono_lyot_im_t + corono_mono_lyot_re_t


#%% Intensity profiles of the direct and coronagraphic images
rad_corono = np.arange(nImg2d//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,nmap))

i0 = 0

pl.figure(11)
pl.clf()
pl.semilogy(rad_corono*Fmax2d/nImg2d, corono_poly_prf_std_t[i0]/direct_poly_img_t[i0].max(),
        label='map {0}'.format(imap0), color = colors_cor[i0])

pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono00.xi2d.min(), xmax=corono00.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'1$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
pl.tight_layout()
pl.savefig(str(fpath_image_plane_plot), transparent=True)

#%% plot displays at multiple wavelengths

values = range(nlam)
colors_wv = pl.cm.rainbow(np.linspace(0,1,nlam))

if corono00.nlam > 1:
    idx_lam_peak = (corono00.nlam+1)//2
else:
    idx_lam_peak = 0

# Intensity profiles of the direct and coronagraphic images
pl.figure(12)
pl.clf()
for i in range(corono00.nlam):
    pl.semilogy(rad_corono*Fmax2d/nImg2d, 5*corono_mono_prf_std_t[i0, i]/direct_mono_img_t[i0, idx_lam_peak].max(),
                label=r'{0:.3f}$\lambda_0$'.format(corono00.lam_t[i]), color = colors_wv[i])
    
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono00.xi2d.min(), xmax=corono00.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'5$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title('Intensity profile in monochromatic light'.format(imap0))
pl.tight_layout()
pl.savefig(str(fpath_image_plane_mono_plot), transparent=True)

pl.show()

#%%
ilam0 = (nlam +1)//2

pl.figure(13)
pl.clf()
if np.ndim(direct_mono_lyot_t) > 2:
    pl.imshow(np.abs(direct_mono_lyot_t[i0, ilam0])**2, cmap = 'inferno')
else:
    pl.imshow(np.abs(direct_mono_lyot_t[i0])**2, cmap = 'inferno')
pl.title('Lyot plane intensity before stop (direct field) for map {0}'.format(i0))

pl.figure(14)
pl.clf()
if np.ndim(direct_mono_lyot_t) > 2:
    pl.imshow(np.abs(direct_mono_lyot_t[i0, ilam0])**2*LyotStop2d, cmap = 'inferno')
else:
    pl.imshow(np.abs(direct_mono_lyot_t[i0])**2*LyotStop2d, cmap = 'inferno')   
pl.title('Lyot plane intensity after stop (direct field) for map {0}'.format(i0))


pl.figure(15)
pl.clf()
pl.imshow(np.abs(corono_mono_lyot_t[i0, ilam0])**2, cmap = 'inferno')
pl.title('Lyot plane intensity before stop (corono field) for map {0}'.format(i0))

pl.figure(16)
pl.clf()
pl.imshow(np.abs(corono_mono_lyot_t[i0, ilam0])**2*LyotStop2d, cmap = 'inferno')
pl.title('Lyot plane intensity after stop (corono field) for map {0}'.format(i0))

pl.show()

#%%

if nmap <= 10:
    f1 = pl.figure(20, figsize=(8,4.5))
    pl.clf()
    for i in range(nmap):
        exec('ax{0} = f1.add_subplot(1,{1},{0})'.format(i+1,nmap))
        exec('im = ax{0}.imshow(np.abs(direct_mono_lyot_t[{1}, ilam0])**2, cmap = "inferno", vmin=0, vmax=2)'.format(i+1,i))
        exec('ax{0}.text(nPup/2, 0.1*nPup, "map {1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,i))
        exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
        exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))
    
    f1.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    f1.subplots_adjust(right=0.8)
    cbar_ax = f1.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar    = f1.colorbar(im, cax=cbar_ax)
    pl.suptitle('Lyot plane intensity before stop (direct field)')
    pl.tight_layout()

#%%
if nmap <= 10:
    f2 = pl.figure(21, figsize=(8,4.5))
    pl.clf()
    for i in range(2*nmap):
        exec('ax{0} = f2.add_subplot(2,{1},{0})'.format(i+1,nmap))
        if i < nmap: 
            exec('im = ax{0}.imshow(np.log10(np.abs(corono_mono_lyot_t[{1}, ilam0])**2), cmap = "inferno", vmin=-3, vmax=0)'.format(i+1,i))
            exec('ax{0}.text(nPup/2, 0.1*nPup, "map {1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,i))
        else:
            exec('im = ax{0}.imshow(np.log10(np.abs(corono_mono_lyot_t[{1}, ilam0])**2*LyotStop2d), cmap = "inferno", vmin=-3, vmax=0)'.format(i+1,(i) % nmap))        
        exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
        exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))
    
    f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    f2.subplots_adjust(right=0.8)
    cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar    = f2.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
    #pl.suptitle('Lyot plane intensity')
    pl.savefig(str(fpath_pupil_plane_disp), transparent=True)
    pl.tight_layout()
    pl.show()

#%%
if nmap <= 10: 
    f2 = pl.figure(22, figsize=(8,4.5))
    pl.clf()
    for i in range(nmap):
        exec('ax{0} = f2.add_subplot(1,{1},{0})'.format(i+1,nmap))
        if i < nmap: 
            exec('im = ax{0}.imshow(np.log10(corono_poly_img_t[{1}]/direct_poly_img_t[{1}].max()), cmap = "inferno", vmin=-7.5, vmax=-3.5)'.format(i+1,i))
            exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "map {1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,i))
        exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
        exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))
    
    f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    f2.subplots_adjust(right=0.8)
    cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar    = f2.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
    pl.savefig(str(fpath_image_plane_disp), transparent=True)
    pl.tight_layout()
    pl.show()

print('ok')

#%% Intensity profiles of the direct and coronagraphic images
rad_corono = np.arange(nImg2d//2)
colors_map = pl.cm.rainbow(np.linspace(0,1,nmap))

pl.figure(31)
pl.clf()
for imap in range(nmap):
    pl.semilogy(rad_corono*Fmax2d/nImg2d, 5*corono_poly_prf_std_t[imap]/direct_poly_img_t[imap].max(),
                    label='map {0}'.format(imap0), color = colors_map[imap])

pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono00.xi2d.min(), xmax=corono00.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'5$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
pl.tight_layout()
pl.savefig(str(fpath_image_allmaps_plot), transparent=True)

pl.show()

#%%
"""
Robustness to NPCA
"""
direct_poly_img_f = direct_poly_img_t[0]

fdir_ncpa_int  = Path('/Users/mndiaye/Dropbox/python/Coronagraphs/data/2D/ZELDA/2018-04-03/internal') 
fname_ncpa_int = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'
fpath_ncpa_int = fdir_ncpa_int / fname_ncpa_int

fdir_ncpa_sky  = Path('/Users/mndiaye/Dropbox/python/Coronagraphs/data/2D/ZELDA/2018-04-03/sky') 
fname_ncpa_sky = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
fpath_ncpa_sky = fdir_ncpa_sky / fname_ncpa_sky

OPDmap2d_int_t = fits.getdata(fpath_ncpa_int)
OPDmap2d_sky_t = fits.getdata(fpath_ncpa_sky)

if kw_correction is False:
    imap_int = 0
    imap_sky = 0
    str_correction = 'correc=0'
else:
    imap_int = 3
    imap_sky = 2
    str_correction = 'correc=1'
    
    
OPDmap2d_int = OPDmap2d_int_t[imap_int]*1e-9
OPDmap2d_sky = OPDmap2d_sky_t[imap_sky]*1e-9
       
OPDmap2d_t = [OPDmap2d_int, OPDmap2d_sky]
 
#%%
# number of coronagraph configuration
ncorono      = len(OPDmap2d_t)
print('# of coronagraph configurations: {0}'.format(ncorono))

#%%
      
# list of parameters for each coronagraph configuration
params_t = []
for k in range(ncorono):
    params_t.append(coro.update_params(params, OPDmap2d=OPDmap2d_t[k], Ampmap2d = Ampmap2d)) 

corono_t = []

params2_t  = []
for k in range(ncorono):
    params2_t.append(coro.update_params(params_t[k], nlam=nlam, Fmax2d = Fmax2d, nImg2d = nImg2d)) 

if corono_name == 'SP':
    corono_t.append(coro.design.SP2d(**params2_t[0]))
elif corono_name == 'APLC':
    for k in range(ncorono):
        corono_t.append(coro.design.APLC2d(**params2_t[k])) 
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
val = 0
if nImg2d%2 == 0:
    val = 1/2

sepbis=2.5
septer=5.0


# array of angular distances in the final image plane
xx,yy  = np.meshgrid(np.arange(nImg2d)-nImg2d//2+val, np.arange(nImg2d)-nImg2d//2+val)
mydist = (Fmax2d/nImg2d)*np.hypot(yy,xx)        
resbis = (mydist <= sepbis +0.5)*(mydist >= sepbis -0.5)
rester = (mydist <= septer +0.5)*(mydist >= septer -0.5)

pl.figure(3)
pl.clf()
pl.imshow(resbis)

#%%
direct_poly_img_aberr_t = [] 
corono_poly_img_aberr_t = []

values = range(nlam)
colors = pl.cm.rainbow(np.linspace(0,1,nlam))

for k in range(ncorono):
    if corono_name == 'APLC':
        direct_poly_img_aberr_t.append(corono_t[k].compute_direct_intensity_2d(Apod2d))
    else:
        direct_poly_img_aberr_t.append(corono_t[k].compute_direct_intensity_2d(corono_t[k].Pupil2d))
    corono_poly_img_aberr_t.append(corono_t[k].compute_corono_intensity_2d(Apod2d))

direct_poly_img_aberr_t = np.asarray(direct_poly_img_aberr_t)
corono_poly_img_aberr_t = np.asarray(corono_poly_img_aberr_t)

direct_poly_img_aberr_prf_avg_t = np.zeros((ncorono, nImg2d//2))
corono_poly_img_aberr_prf_avg_t = np.zeros((ncorono, nImg2d//2))
direct_poly_img_aberr_prf_std_t = np.zeros((ncorono, nImg2d//2))
corono_poly_img_aberr_prf_std_t = np.zeros((ncorono, nImg2d//2))

for i in range(ncorono):
    direct_poly_img_aberr_prf_avg_t[i], rad_direct = imutils.profile(direct_poly_img_aberr_t[i], type='mean')
    corono_poly_img_aberr_prf_avg_t[i], rad_corono = imutils.profile(corono_poly_img_aberr_t[i], type='mean')
    direct_poly_img_aberr_prf_std_t[i], rad_direct = imutils.profile(direct_poly_img_aberr_t[i], type='std')
    corono_poly_img_aberr_prf_std_t[i], rad_corono = imutils.profile(corono_poly_img_aberr_t[i], type='std')


#%%
pl.figure(1)
pl.clf()
pl.imshow(rester)


#%%
corono_poly_avg_resbis_aberr_t = []
corono_poly_avg_rester_aberr_t = []

for i in range(ncorono):
    corono_poly_avg_resbis_aberr_t.append(np.mean(corono_poly_img_aberr_t[i, resbis != 0])/direct_poly_img_f.max())
    corono_poly_avg_rester_aberr_t.append(np.mean(corono_poly_img_aberr_t[i, rester != 0])/direct_poly_img_f.max())
    
    

#%%
ncpa_name = ['int', 'on-sky']

fname_image_plane_disp = 'corono_poly_ncpa_sensitivity_{0}_disp.pdf'.format(str_correction)
fpath_image_plane_disp = fdir_plots / fname_image_plane_disp

if len(corono_t) <= 9:
    f2 = pl.figure(30, figsize=(10,4.5))
    pl.clf()
    for i in range(ncorono):
        exec('ax{2} = f2.add_subplot({0},{1},{2})'.format(1,ncorono,i+1))
        exec('im = ax{0}.imshow(np.log10(corono_poly_img_aberr_t[{1}]/direct_poly_img_aberr_t[{1}].max()), cmap = "inferno", vmin=-7.5, vmax=-3.5)'.format(i+1,i))
        exec('ax{0}.text(nImg2dbis/2, 0.1*nImg2dbis, "{1} NCPA", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,ncpa_name[i]))
        exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
        exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))
        
    f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    f2.subplots_adjust(right=0.85)
    cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar    = f2.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('corono image', rotation=270, labelpad = 10)
    pl.savefig(str(fpath_image_plane_disp), transparent=True)
    pl.tight_layout()
    pl.show()

#%%

colors_ncpa = pl.cm.rainbow(np.linspace(0,1,ncorono+1))

fname_ncpa_plot = 'corono_poly_ncpa_sensitivity_{0}_plot.pdf'.format(str_correction)
fpath_ncpa_plot = fdir_plots / fname_ncpa_plot

plot_lines = []

pl.figure(32)
pl.clf()
pl.semilogy(rad_corono*Fmax2d/nImg2d, 5*corono_poly_prf_std_t[0]/direct_poly_img_f.max(),
                    label='No NCPA'.format(0), color = colors_ncpa[0])
for i in range(ncorono):
    pl.semilogy(rad_corono*Fmax2d/nImg2d, 5.*corono_poly_img_aberr_prf_std_t[i]/direct_poly_img_aberr_t[i].max(),
                    label='{0} NCPA'.format(ncpa_name[i]), color = colors_ncpa[i+1])


pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono00.xi2d.min(), xmax=corono00.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'5$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
pl.tight_layout()
pl.savefig(str(fpath_ncpa_plot), transparent=True)

pl.show()
