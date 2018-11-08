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
from pyzelda.utils import aperture, imutils
import corono as coro

import time

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
nlam   = 11
bw     = 0.20#width/wv 
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
if kw_aberr is True:

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
results from Paranal run
"""
#%%
#path_root     = Path('/Users/mndiaye/Dropbox/python/zelda/ZELDA-2018')
#path_data_int = path_root / '2018-04-03' / 'data'
#path_data_sky = path_root / '2018-04-03_night' / 'data'
#pixel_irdis = 12.25
#
##%%
#prefix = 'aplc_internal_source'
#
## read images for normalisation
#psf_def_img  = fits.getdata(path_data_int / '{:s}_psf_ref_image.fits'.format(prefix))
#psf_def_norm_int = psf_def_img.max()
#
#psf_zel_img = fits.getdata(path_data_int / '{:s}_psf_zel_image.fits'.format(prefix))
#psf_zel_norm_int = psf_zel_img.max()
#
## read profiles
#psf_def_prf_int = fits.getdata(path_data_int / '{:s}_psf_ref_profile.fits'.format(prefix))
#psf_zel_prf_int = fits.getdata(path_data_int / '{:s}_psf_zel_profile.fits'.format(prefix))
#
#cor_def_prf_int = fits.getdata(path_data_int / '{:s}_coro_ref_profile_std.fits'.format(prefix))
#cor_zel_prf_int = fits.getdata(path_data_int / '{:s}_coro_zel_profile_std.fits'.format(prefix))
#
## separations
#psf_sep_int = np.arange(len(psf_def_prf_int))*pixel_irdis
#cor_sep_int = np.arange(cor_def_prf_int.shape[1])*pixel_irdis

#%% tools for throughput
#%% 
def generate_phot_aperture(mD=50, nImg=500, sep=5.0, ee_rad=0.7, axis=0):
    ''' --------------------------------------------------------------
    Compute the list of points with a given area in the final image plane 
    of the coronagraph. The area is defined by a circle centered at a given 
    angular separation from the star.
    
    Parameters:
    ---------- 
    
    - mD     : spatial frequency range in the final image plane D in lam0/D
    - nImg   : linear number of points in the final image plane D
    - sep    : angular separation from the star for the area 
    - ee_rad : angular radius of the photometric aperture
    - axis   : axis along which the photometric aperture is set (0: y-axis, 1: x-axis)
    
    Output:
    ----------
    
     - res    : 2D array with 1 and 0 for points inside and outside the area in the 
    coronagraphic image
    
    -------------------------------------------------------------- '''
    # conversion of values to float
    mD     = float(mD)
    nImg   = float(nImg)
    
    # array of angular distances in the final image plane
    sep_pix = sep*(nImg/mD)
    if axis == 0:
        xx,yy   = np.meshgrid(np.arange(nImg)-nImg/2-sep_pix, np.arange(nImg)-nImg/2)
    else:
        xx,yy   = np.meshgrid(np.arange(nImg)-nImg/2, np.arange(nImg)-nImg/2-sep_pix)
    mydist  = (mD/nImg)*np.hypot(yy,xx)
    
    # array with 1 and 0 for points inside and outside the area in the coronagraphic image
    res    = np.zeros_like(mydist)
    res[(mydist <= ee_rad)] = 1.0
    return res

#%% 
def estimate_contrast(image, res):
    ''' --------------------------------------------------------------
    Estimate the averaged intensity inside a given area in the final image plane.
    
    Parameters:
    ---------- 
    
    - image   : 2D array of the image in the final image plane
    - res     : linear number of points in the final image plane D
    
    Output:
    ----------
    
    - 2D array with averaged intensity inside a given area in the coronagraphic image
    
    -------------------------------------------------------------- ''' 
    return np.mean(image[res == 1.0]), np.std(image[res == 1.0])
    
#%% 
def estimate_energy(image, res):
    ''' --------------------------------------------------------------
    Estimate the energy inside a given area in the final image plane.
    
    Parameters:
    ---------- 
    
    - image   : 2D array of the image in the final image plane
    - res     : linear number of points in the final image plane D
    
    Output:
    ----------
    
    - 2D array with energy inside a given area in the coronagraphic image
    
    -------------------------------------------------------------- ''' 
    return np.sum(image[res == 1.0])

#%%
# vector of angular separation for the computation of planet transmission
sep_min  = 0.0
sep_max  = 11.0
sep_stp  = 0.5
nsep     = int(round(1+(sep_max-sep_min)/sep_stp))
sep_vec  = sep_min + sep_stp*np.arange(nsep)
print('{0:5d} sep'.format(nsep))

'''
### read aperture PSF
'''
int_D0 = np.empty((nImg2d, nImg2d))
int_D0 = direct_poly_img_t[0]/np.max(direct_poly_img_t[0])

'''
### estimate energy within photometric aperture w/o coronagraph
'''
planet_area0 = generate_phot_aperture(mD=Fmax2d, nImg=nImg2d, sep=0., ee_rad=0.7)
energy_D0    = estimate_energy(int_D0, planet_area0)
print('energy w/o  coronagraph: {0}'.format(energy_D0))
energy_0     = np.sum(int_D0)
print('total energy: {0}'.format(energy_0))


#%%

'''
### computation of a tip mode for planet location
''' 
xx,yy   = np.meshgrid(np.arange(nPup)-nPup/2, np.arange(nPup)-nPup/2)
rr      = (2./np.float(nPup))*np.hypot(yy,xx)
theta   = np.arctan2(yy,xx)
Z       = 2.*rr*np.cos(theta)   

#%%
'''
### computation of the coronagraphic image
'''
int_D1_arr = np.empty((nsep, nImg2d, nImg2d))       
for l in range(nsep):
    sep = sep_vec[l]
    opd = sep*(1./4.) * Z
    params   = coro.update_params(params, OPDmap2d = opd, Ampmap2d = None, LyotStop2d = LyotStop2d)
    corono0  = coro.design.APLC2d(**params)
    int_D1_arr[l] = corono0.compute_corono_intensity_2d(Apod2d)
    int_D1_arr[l] /= np.max(direct_poly_img_t[0])

#%%
pl.figure(30)
pl.clf()
pl.imshow(np.log10(int_D1_arr[0]), cmap = 'inferno')


    
#%%
'''
### estimate energy within photometric aperture with coronagraph at a given location
# eta_S: fraction of star light in the ROI (region of interest: photometric aperture of 0.7lam/D)
# eta_P: fraction of planet light in the ROI
'''
eta_P = np.empty((nsep))
eta_S = np.empty((nsep))
t0 = time.time()
for l in range(nsep):
    sep = sep_vec[l]
    ROI = generate_phot_aperture(mD=Fmax2d, nImg=nImg2d, sep=sep, ee_rad=0.7)

    print('{0:05d}/{1:05d} computation'.format(l+1,nsep))
    index_S = 0
    index_P = l
    energy_S = estimate_energy(int_D1_arr[index_S], ROI)
    energy_P = estimate_energy(int_D1_arr[index_P], ROI)
    eta_S[l] = energy_S/energy_0  
    eta_P[l] = energy_P/energy_0 
    print('fraction of star   light in the ROI: {0}'.format(eta_S[l])) 
    print('fraction of planet light in the ROI: {0}'.format(eta_P[l]))
t1 = time.time()
print('exec time: {0}s'.format(t1-t0))
print('')

#%%
'''
array of angular distances in x and y
'''
xx1,yy1   = np.meshgrid(np.arange(nImg)-nImg/2., np.arange(nImg)-nImg/2.)
mydist1   = (mD/float(nImg))*np.hypot(yy1,xx1)

'''
### 1d and 2D fit of the planet transmission eta_P
'''
nsample     = 50
sep_vec_new = np.arange(0, sep_max, sep_stp/nsample)
eta_P_1dfit = np.empty((np.shape(sep_vec_new)[0]))
eta_S_1dfit = np.empty((np.shape(sep_vec_new)[0]))
eta_P_2dfit = np.empty((ndesign, nImg, nImg))
eta_S_2dfit = np.empty((ndesign, nImg, nImg))
for i in range(nOD):
    for j in range(nID):  
        for k in range(nfpm):
            tck_P = interpolate.splrep(sep_vec, eta_P[i,j,k], s=0)
            tck_S = interpolate.splrep(sep_vec, eta_S[i,j,k], s=0)            
            eta_P_1dfit[i,j,k] = interpolate.splev(sep_vec_new, tck_P, der=0)
            eta_S_1dfit[i,j,k] = interpolate.splev(sep_vec_new, tck_S, der=0)
            index_sub = i*nID*nfpm+j*nfpm+k
            eta_P_2dfit[index_sub] = interpolate.splev(mydist1, tck_P, der=0)
            eta_S_2dfit[index_sub] = interpolate.splev(mydist1, tck_S, der=0)


#%%
'''
### check radial planet transmission profiles
'''
lamD_vec   = (Fmax2d/float(nImg2d))*np.arange(nImg2d/2)


fig = pl.figure(5)
pl.clf()
ax  = fig.add_subplot(111)
#ax.set_yscale('log')
#ax.set_color_cycle([cm0(1.*i/(ncurves-1)) for i in range(ncurves)])
ax.plot(sep_vec, eta_P, label = 'corono')
#ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.set_xlabel(r'Ang. sep. in $\lambda$/D')
ax.set_ylabel(r'Normalized intensity')
ax.legend(loc = 0, labelspacing=0.01, title = 'images')
fig.tight_layout()


