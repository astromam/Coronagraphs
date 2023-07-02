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
ftsz = 16 
pl.rcParams.update({'font.size': ftsz})
from matplotlib.patches import Circle

from pathlib import Path
from astropy.io import fits
from pyzelda.utils import aperture, imutils
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

bw     = 0.20#width/wv 
nFPM   = 50

nImg2dbis = 256
Fmax2dbis = nImg2dbis/(2*(wv/950e-9))
nlambis   = 11

kw_aberr     = False
kw_2nddate   = True    
kw_skyobs    = True
kw_aftercorr = False
kw_saxo      = True

do_plot = True 
do_fits = True 

Dtel = 8
lam0D2mas = (wv/Dtel)*(180.*3600*1000/np.pi)

lam02um = wv*1e6

val = 0
if nImg2dbis%2 == 0:
    val = 1/2

#%%
"""
### File directories
"""
if kw_aberr is False:
    str_aberr = 'wo_aberr_with_irdis_plate_scale'
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

#fdir = Path('../../').resolve()
fdir = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/Coronagraphs').resolve()

fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_date / str_obs / str_saxo / str_corr  
fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
fdir_saxo    = fdir / 'data' / '2D' / 'ZELDA' / '2018-04-03'

fdir_plots    = fdir / 'results' / '2D' / 'plots' / 'SPHERE' / str_aberr / str_date / str_obs / str_corr  
fdir_pupimages = fdir_results / 'pupimages'

#%%
"""
### Filepaths
"""

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
"""
### Filenames to be read
"""
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
"""
### Filenames to be saved
"""
fname_image_plane_plot   = 'corono_poly_prf_std_t_OPDmap={0}_plot.pdf'.format(imap0)
fname_image_plane_mono_plot   = 'corono_poly_prf_std_t_OPDmap={0}_mono_plot.pdf'.format(imap0)
fname_image_allmaps_plot = 'corono_poly_prf_std_t_OPDmap=all_plot.pdf'
fname_image_allmaps_plot_comp = 'corono_poly_prf_std_t_OPDmap=all_plot_comp.pdf'
fname_image_plane_disp   = 'corono_poly_img_t_OPDmap={0}_disp.pdf'.format(imap0)
fname_pupil_plane_disp   = 'corono_poly_lyot_t_OPDmap={0}_disp.pdf'.format(imap0)
fname_image_plane_disp_all   = 'corono_poly_img_t_OPDmap={0}_disp_all0.pdf'.format(imap0)

fpath_image_plane_plot   = fdir_plots / fname_image_plane_plot
fpath_image_plane_mono_plot   = fdir_plots / fname_image_plane_mono_plot
fpath_image_allmaps_plot = fdir_plots / fname_image_allmaps_plot
fpath_image_allmaps_plot_comp = fdir_plots / fname_image_allmaps_plot_comp
fpath_image_plane_disp   = fdir_plots / fname_image_plane_disp
fpath_pupil_plane_disp   = fdir_plots / fname_pupil_plane_disp
fpath_image_plane_disp_all   = fdir_plots / fname_image_plane_disp_all


#%% Entrance pupil
"""
### Display plots
"""

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
"""
### Coronagraph object
"""

if corono_name != 'APLC':
    raise NameError('Check the name of the coronagraph!')
    
params = coro.to_dict(nPup=nPup, nImg2d=nImg2dbis, Fmax2d = Fmax2dbis, nFPM = nFPM,
                 rMask = rMask,
                 SymPupil2d = SymPupil2d, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                 CtrBtwnPix=CtrBtwnPix,
                 CtrBtwnPix2 = CtrBtwnPix2, 
                 nlam=nlambis, bw = bw, wv =wv,
                 rho0   = rho0, rho1 = rho1, cDarkHole = cDarkHole,
                 OPDmap2d = None, Ampmap2d = None)
corono00 = coro.design.APLC2d(**params)

#%%
"""
### Image plane intensity
"""
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
"""
### Pupil plane intensity 
"""    

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
"""
###Intensity profiles of the direct and coronagraphic images
"""
rad_corono = np.arange(nImg2dbis//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,nmap))

i0 = 0

pl.figure(11)
pl.clf()
pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, corono_poly_prf_std_t[i0]/direct_poly_img_t[i0].max(),
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

values = range(nlambis)
colors_wv = pl.cm.rainbow(np.linspace(0,1,nlambis))

if corono00.nlam > 1:
    idx_lam_peak = (corono00.nlam+1)//2
else:
    idx_lam_peak = 0

# Intensity profiles of the direct and coronagraphic images
pl.figure(12)
pl.clf()
for i in range(corono00.nlam):
    pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_mono_prf_std_t[i0, i]/direct_mono_img_t[i0, idx_lam_peak].max(),
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

#%% Lyot plane intensity
ilam0 = (nlambis +1)//2

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

#%% Lyot plane intensity

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
            exec('ax{0}.text(nImg2dbis/2, 0.1*nImg2dbis, "map {1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,i))
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
"""
### Intensity profiles of the direct and coronagraphic images
"""

rad_corono = np.arange(nImg2dbis//2)
colors_map = pl.cm.rainbow(np.linspace(0,1,nmap))

pl.figure(31)
pl.clf()
for imap in range(nmap):
    pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_prf_std_t[imap]/direct_poly_img_t[imap].max(),
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
Robustness to spectral bandwidth
"""

#%%
"""
### Star image in monochormatic light
"""   
nlam_ter = 101
bw_ter   = 0.92


#%%
"""
### Computation of the normalization factor
"""
# Parameters for the monochromatic PSF for a pupil without apodizer nor Lyot stop   
params_N = coro.update_params(params, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis, 
                                nlam = 1, bw = bw_ter, Pupil2d = Pupil2d, 
                                LyotStop2d = Pupil2d)

# Coronagraph object definition including previous parameters
if corono_name != 'APLC':
    raise NameError('{0}: Not an APLC!'.format(corono_name))
corono_N = coro.design.APLC2d(**params_N)

# Computation of the monochromatic psf of the telescope aperture
direct_mono_img_N = corono_N.compute_direct_intensity_2d(Pupil2d, poly=False)

# normalization term wrt to the PSF of the pupil without apodizer nor Lyot stop  
direct_mono_img_N_peak = direct_mono_img_N[0].max()

#%%
"""
### Computation of the monochromatic direct and coronagraphic images
"""
# filepaths of the monochromatic direct and monochromatic images
fname_imgs_dir = 'current_aplc_poly_bw={0}_nlam={1}_nPup={2}_mono_direct.fits'.format(bw_ter, nlam_ter, nPup)
fname_imgs_cor = 'current_aplc_poly_bw={0}_nlam={1}_nPup={2}_mono_corono.fits'.format(bw_ter, nlam_ter, nPup)
fpath_imgs_dir = fdir_plots / fname_imgs_dir
fpath_imgs_cor = fdir_plots / fname_imgs_cor

# Parameters for the coronagraph object 
params_S    = coro.update_params(params, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis, 
                                nlam = nlam_ter, bw = bw_ter)

# coronagraph object
if corono_name != 'APLC':
    raise NameError('{0}: Not an APLC!'.format(corono_name))
corono_S = coro.design.APLC2d(**params_S)

# computation or reading of the direct and coronagraphic images
print('computation of the star coronagraphic image')
t0 = time.time()
if do_fits:
    # broadband images
    #direct_poly_img3 = corono_S.compute_direct_intensity_2d(Apod2d)
    #corono_poly_img3 = corono_S.compute_corono_intensity_2d(Apod2d)
      
    # monochromatic images
    direct_mono_img_S = corono_S.compute_direct_intensity_2d(Apod2d, poly=False)
    corono_mono_img_S = corono_S.compute_corono_intensity_2d(Apod2d, poly=False)

    # normalized direct and coronagraphic images in monochromatic light
    direct_mono_img_S /= direct_mono_img_N_peak
    corono_mono_img_S /= direct_mono_img_N_peak

    fits.writeto(fpath_imgs_dir, direct_mono_img_S, overwrite= True)
    fits.writeto(fpath_imgs_cor, corono_mono_img_S, overwrite= True)
else:
    direct_mono_img_S = fits.getdata(fpath_imgs_dir)
    corono_mono_img_S = fits.getdata(fpath_imgs_cor)  
    
t1 = time.time()
print('computation time: {0:.2f}s'.format(t1-t0))    

#%%
"""
### Planet image in monochormatic light
"""   
# vector of angular separation for the computation of planet transmission
sep_min  = 0.0
sep_max  = Fmax2dbis//2
sep_stp  = Fmax2dbis/nImg2dbis#0.5

# array of angular separations
nsep     = int(round(1+(sep_max-sep_min)/sep_stp))
sep_arr  = sep_min + sep_stp*np.arange(nsep)
print('{0:5d} sep'.format(nsep))

# Zernike mode for the tip mode
xx,yy   = np.meshgrid(np.arange(nPup)-nPup/2, np.arange(nPup)-nPup/2)
rr      = (2./float(nPup))*np.hypot(yy,xx)
theta   = np.arctan2(yy,xx)
Z       = 2.*rr*np.cos(theta)*Pupil2d   

# OPD map to generate a planet
opd_arr = [sep*wv*(1./4.) * Z for l, sep in enumerate(sep_arr)]

#%%
# computation of the coronagraphic image of the planet
corono_mono_img_P = np.zeros((nsep, nlam_ter, nImg2dbis, nImg2dbis))
print('computation of the planet coronagraphic image')
t0 = time.time()
for l in range(nsep):
    params_P = coro.update_params(params_S, OPDmap2d = opd_arr[l]) 
    corono_P = coro.design.APLC2d(**params_P)
    corono_mono_img_P[l]  = corono_P.compute_corono_intensity_2d(Apod2d, poly = False)
t1 = time.time()
print('computation time: {0:.2f}s'.format(t1-t0))

# normalization of the planet coronagraphic image
corono_mono_img_P /= direct_mono_img_N_peak

#%% display of the planet image

Z0P = np.log10(corono_mono_img_P[2, nlam_ter//2])

pl.figure(30)
pl.clf()
pl.imshow(Z0P, cmap = 'inferno', vmin = -7.5, vmax = -3.5)    
pl.show()    

#%%
"""
### Eta_S and Eta_P computation
"""
# photometric aperture circular or ring radius in lam/D
phot_rad = 0.7

# array of angular distances in the final image plane
xxi,yyi  = np.meshgrid(np.arange(nImg2dbis)-nImg2dbis//2+val, np.arange(nImg2dbis)-nImg2dbis//2+val)
mydist = (Fmax2dbis/nImg2dbis)*np.hypot(yyi,xxi)        

eta_S = np.zeros((nsep, nlam_ter))
eta_P = np.zeros((nsep, nlam_ter))

ring_aper = np.zeros((nImg2dbis, nImg2dbis))
circ_aper = np.zeros((nImg2dbis, nImg2dbis))

sum_ring_aper = np.zeros((nsep, nlam_ter))
sum_circ_aper = np.zeros((nsep, nlam_ter))

print('eta_S and eta_P computation')
t0 = time.time()
for i in range(nsep):
    for j in range(nlam_ter):
    # compute averaged starlight intensity
        ind_S = (mydist <= sep_arr[i] + phot_rad)*(mydist >= sep_arr[i] -phot_rad)
        ring_aper[ind_S] = 1.
        sum_ring_aper[i,j] = np.sum(ring_aper)
        eta_S[i,j] = np.sum(ring_aper*corono_mono_img_S[j])/np.sum(ring_aper)
        ring_aper[ind_S] = 0.
    
        # compute averaged intensity of the planet at its location within a photmetric aperture of 0.7lam/D
        mydist_P = (Fmax2dbis/nImg2dbis)*np.hypot(yyi,xxi-sep_arr[i]*nImg2dbis/Fmax2dbis)        
        ind_P = (mydist_P <= phot_rad)
        circ_aper[ind_P] = 1.
        sum_circ_aper[i,j] = np.sum(circ_aper)
        eta_P[i,j] = np.sum(circ_aper*corono_mono_img_P[i, j])/np.sum(circ_aper)
        circ_aper[ind_P] = 0.
t1 = time.time()
print('computation time: {0}s'.format(t1-t0))
    
#%%
"""
### Eta_S/Eta_P ratio
"""
etaSP_ratio = eta_S/eta_P

#%%

colors_wv = pl.cm.rainbow(np.linspace(0,1,nlam_ter-1))

pl.figure(34)
pl.clf()
for i in range(nlam_ter-1):
    pl.plot(sep_arr, np.log10(etaSP_ratio[:, i]), color = colors_wv[i])
pl.xlabel(r'angular separation in $\lambda$/D') 
pl.ylabel(r'$\eta_S/\eta_P$')    
pl.title(r'$\eta_S/\eta_P$ vs wavelength $\lambda$')
pl.show()

#%%
"""
Contour plot for spectral bandwidth robustnesswith eta_S/eta_P
""" 

#%%
# normalization term    
#direct_mono_img_f3_peak = direct_mono_img_t3[nlam_ter//2].max()

lam0D_corono = rad_corono*Fmax2dbis/nImg2dbis

wv_ind = wv*corono_S.lam_t >= 0.95e-6
lam0D_ind = sep_arr >= 1.0

# bounds for the separations
lam0D_min = sep_arr[lam0D_ind].min()
lam0D_max = sep_arr[lam0D_ind].max()

# bounds for the wavelengths
lam_min0 = corono_S.lam_t.min()
lam_min = corono_S.lam_t[wv_ind].min()
lam_max = corono_S.lam_t[wv_ind].max()

# bounds for the optimization wavelength 
lam0 = 1.
lam_opt_min = lam0 - 0.5*bw#corono0.lam_t.min()
lam_opt_max = lam0 + 0.5*bw#corono0.lam_t.max()

#%%
fname_image_plane_f_disp = 'current_aplc_poly_bw_sensitivity_contour_nPup={0}_etaS-etaP_disp.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_plots / fname_image_plane_f_disp


# line width parameter
lw0 = 2.5

tmp = etaSP_ratio[:, wv_ind]

Z0 = np.log10(tmp[lam0D_ind]).T
extent0 = [lam0D_min, lam0D_max, lam_min, lam_max]

f2 = pl.figure(32, figsize=(8,4.5))
pl.clf()
ax0 = f2.add_subplot(111)

im = ax0.imshow(Z0, 
                cmap = "inferno", 
                vmin=-7.5, vmax=-3.5,
                extent = extent0,
                origin = 'lower',
                )

cs = ax0.contour(Z0, [-7., -6., -5.], colors = 'white',
            extent = extent0,linestyles = '-')
ax0.clabel(cs, inline=1, fontsize=ftsz, fmt = '%1.1f')

ax0.set_xlabel(r'Angular separation in $\lambda_0$/D')
ax0.set_ylabel(r'Wavelength $\lambda$ in $\lambda_0$')
ax0.set_aspect('auto')
#ax0.set_title(r'New APLC design')
ax0.text(10**(np.log10(lam0D_max)/2), 1.4, "Current APLC design", fontsize=ftsz, 
         horizontalalignment="center", color = "blue")
#exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(1,))
#exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(1,))



ax0.axvline(x=rMask, ymin=-12, ymax =2, linewidth=lw0, color='r', linestyle='--')
ax0.axvline(x=rho0, ymin=-12, ymax =2, linewidth=lw0, color='b', linestyle='--')
ax0.axvline(x=rho1, ymin=-12, ymax =2, linewidth=lw0, color='b', linestyle='--')

ax0.axhline(y=1.0, xmin=0., xmax =lam0D_max, 
            linewidth=lw0, color='g', linestyle=':')
ax0.axhline(y=lam_opt_min, xmin=0., xmax =lam0D_max, 
            linewidth=lw0, color='k', linestyle=':')
ax0.axhline(y=lam_opt_max, xmin=0., xmax =lam0D_max, 
            linewidth=lw0, color='k', linestyle=':')

ax0.autoscale(False)
ax0.set_xscale("log")
#
ax2 = ax0.twinx()
ax2.set_ylim(lam_min*lam02um, lam_max*lam02um)
ax2.set_ylabel(r'$\lambda$ in $\mu$m ($\lambda_0={0}\mu$m)'.format(wv*1e6), 
               rotation=270, labelpad = ftsz)
#
##
ax3 = ax0.twiny()
ax3.set_xlim(lam0D_min*lam0D2mas, lam0D_max*lam0D2mas)
##ax3.set_yticks()
ax3.set_xlabel(r'Angular separation in mas')
ax3.set_xscale("log")



f2.subplots_adjust(bottom=0.13, top=0.87, left=0.1, right=0.75,
                    wspace=0.02, hspace=0.02)
#
#f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.86, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Raw contrast $\eta_S/\eta_P$ in log scale', rotation=270, labelpad = 16)
if do_plot is True:
    pl.savefig(str(fpath_image_plane_f_disp), transparent=True)
pl.tight_layout()
pl.show()



    

