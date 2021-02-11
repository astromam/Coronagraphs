#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:55:49 2018

@author: mndiaye
"""

#%% Initialization
import numpy as np
import os
import pylab as pl
from pathlib import Path
from astropy.io import fits
from pyzelda.utils import aperture
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

#%%
fdir = Path('.').resolve()

fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / 'tests' 
fdir_plots    = fdir / 'results' / '2D' / 'plots' / 'SPHERE'

if not os.path.exists(fdir_results):
    os.makedirs(fdir_results)
    
if not os.path.exists(fdir_plots):
    os.makedirs(fdir_plots)    

fname_Apod2d     = 'SPHERE_APO1_field_transmission_map.fits'
#fname_Apod2d = 'sphere_pupil_APO1_BH.fits'
fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
fname_OPDmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'
fname_LyotStop2d = 'sphere_stop_ST_ALC2.fits'

fpath_Apod2d     = fdir_pupils / fname_Apod2d
fpath_Ampmap2d   = fdir_pupils / fname_Ampmap2d
fpath_OPDmapnm3d = fdir_pupils / fname_OPDmapnm3d
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d

#%%

fname_direct_poly_img_t     = 'direct_poly_img_t.fits'
fname_corono_poly_img_t     = 'corono_poly_img_t.fits'
fname_direct_poly_prf_avg_t = 'direct_poly_prf_avg_t.fits'
fname_corono_poly_prf_avg_t = 'corono_poly_prf_avg_t.fits'
fname_direct_poly_prf_std_t = 'direct_poly_prf_std_t.fits'
fname_corono_poly_prf_std_t = 'corono_poly_prf_std_t.fits'

fpath_direct_poly_img_t     = fdir_results / fname_direct_poly_img_t
fpath_corono_poly_img_t     = fdir_results / fname_corono_poly_img_t
fpath_direct_poly_prf_avg_t = fdir_results / fname_direct_poly_prf_avg_t
fpath_corono_poly_prf_avg_t = fdir_results / fname_corono_poly_prf_avg_t
fpath_direct_poly_prf_std_t = fdir_results / fname_direct_poly_prf_std_t
fpath_corono_poly_prf_std_t = fdir_results / fname_corono_poly_prf_std_t

fname_direct_mono_img_t     = 'direct_mono_img_t.fits'
fname_corono_mono_img_t     = 'corono_mono_img_t.fits'
fname_direct_mono_prf_avg_t = 'direct_mono_prf_avg_t.fits'
fname_corono_mono_prf_avg_t = 'corono_mono_prf_avg_t.fits'
fname_direct_mono_prf_std_t = 'direct_mono_prf_std_t.fits'
fname_corono_mono_prf_std_t = 'corono_mono_prf_std_t.fits'

fpath_direct_mono_img_t     = fdir_results / fname_direct_mono_img_t
fpath_corono_mono_img_t     = fdir_results / fname_corono_mono_img_t
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
imap0 = 0
fname_image_plane_plot   = 'corono_poly_prf_std_t_OPDmap={0}_plot.pdf'.format(imap0)
fname_image_allmaps_plot = 'corono_poly_prf_std_t_OPDmap=all_plot.pdf'
fname_image_plane_disp   = 'corono_poly_img_t_OPDmap={0}_disp.pdf'.format(imap0)
fname_pupil_plane_disp   = 'corono_poly_lyot_t_OPDmap={0}_disp.pdf'.format(imap0)

fpath_image_plane_plot   = fdir_plots / fname_image_plane_plot
fpath_image_allmaps_plot = fdir_plots / fname_image_allmaps_plot
fpath_image_plane_disp   = fdir_plots / fname_image_plane_disp
fpath_pupil_plane_disp   = fdir_plots / fname_pupil_plane_disp


#%% Entrance pupil
#Pupil2d = fits.getdata(fpath_Pupil2d)
#Pupil2d = coro.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)
#Pupil2d = aperture.vlt_pupil(nPup, nPup, spiders_thickness=4*0.008)
#Pupil2d = aperture.sphere_irdis_pupil()

pl.figure(1)
pl.clf()
pl.imshow(Pupil2d)
pl.title('Entrance pupil')

#%% Apodization
Apod2d = fits.getdata(fpath_Apod2d)

pl.figure(2)
pl.clf()
pl.imshow(Apod2d*Pupil2d, cmap = 'inferno')
pl.title('Apodized entrance pupil')

#%% Phase errors
OPDmapnm3d = fits.getdata(fpath_OPDmapnm3d)
OPDmapnm2d = OPDmapnm3d[imap0]
OPDmap2d   = OPDmapnm2d*1e-9

nmap = len(OPDmapnm3d)

pl.figure(4)
pl.clf()
pl.imshow(OPDmap2d)
pl.title('Phase map')

#%% Amplitude errors
Ampmap2d = fits.getdata(fpath_Ampmap2d)

pl.figure(3)
pl.clf()
pl.imshow(Apod2d*Pupil2d*Ampmap2d, cmap = 'inferno')
pl.title('Amplitude map')

#%% Lyot Stop
LyotStop2d = fits.getdata(fpath_LyotStop2d)

pl.figure(4)
pl.clf()
pl.imshow(LyotStop2d)
pl.title('Lyot stop')

#%% 
LyotStop2dth = aperture.vlt_pupil(nPup, 0.96*nPup, dead_actuator_diameter=0, spiders_thickness=4*0.008)

test = aperture.annulus(nPup, 1.52*0.14*nPup/2, 0.965*nPup/2)*1.

LyotStop2dth *= test.astype(LyotStop2dth.dtype) 

pl.figure(5)
pl.clf()
pl.imshow(LyotStop2dth)
pl.title('Lyot stop (th)')

pl.show()

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

"#%%
label_lst  = ['all errors', 'phase errors only', 'ampl. errors only', 'no errors', 'no errors - th']

ncase = len(label_lst)

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
corono_mono_prf_avg_t = fits.getdata(fpath_corono_mono_prf_avg_t)
direct_mono_prf_std_t = fits.getdata(fpath_direct_mono_prf_std_t)
corono_mono_prf_std_t = fits.getdata(fpath_corono_mono_prf_std_t)

#%%    
direct_mono_lyot_re_t = fits.getdata(fpath_direct_mono_lyot_re_t)
direct_mono_lyot_im_t = fits.getdata(fpath_direct_mono_lyot_im_t)
corono_mono_lyot_re_t = fits.getdata(fpath_corono_mono_lyot_re_t)
corono_mono_lyot_im_t = fits.getdata(fpath_corono_mono_lyot_im_t) 

direct_mono_lyot_t  = 1j*direct_mono_lyot_im_t + direct_mono_lyot_re_t
corono_mono_lyot_t  = 1j*corono_mono_lyot_im_t + corono_mono_lyot_re_t


#%% Intensity profiles of the direct and coronagraphic images
icase0    = 0

rad_corono = np.arange(nImg2d//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,ncase))

pl.figure(21)
pl.clf()
for icase in range(ncase):
    pl.semilogy(rad_corono*Fmax2d/nImg2d, corono_poly_prf_std_t[icase, imap0]/direct_poly_img_t[icase, imap0].max(),
                label=label_lst[icase], color = colors_cor[icase])

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
pl.savefig(str(fpath_image_plane_plot))

#%% plot displays

values = range(nlam)
colors_wv = pl.cm.rainbow(np.linspace(0,1,nlam))

if corono00.nlam > 1:
    idx_lam_peak = (corono00.nlam+1)//2
else:
    idx_lam_peak = 0

# Intensity profiles of the direct and coronagraphic images
pl.figure(22)
pl.clf()
for i in range(corono00.nlam):
    pl.semilogy(rad_corono*Fmax2d/nImg2d, corono_mono_prf_std_t[icase0,imap0, i]/direct_mono_img_t[icase0, imap0, idx_lam_peak].max(),
                label=r'{0:.3f}$\lambda_0$'.format(corono00.lam_t[i]), color = colors_wv[i])
    
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono00.xi2d.min(), xmax=corono00.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'1$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title('Intensity profile in monochromatic light')

pl.show()

#%%
icase0 = 0
ilam0  = (nlam +1)//2

pl.figure(6)
pl.clf()
pl.imshow(np.abs(direct_mono_lyot_t[icase0, imap0, ilam0])**2, cmap = 'inferno')
pl.title('Lyot plane intensity before stop (direct field)')

pl.figure(7)
pl.clf()
pl.imshow(np.abs(direct_mono_lyot_t[icase0, imap0, ilam0])**2*LyotStop2d, cmap = 'inferno')
pl.title('Lyot plane intensity after stop (direct field)')


pl.figure(8)
pl.clf()
pl.imshow(np.abs(corono_mono_lyot_t[icase0, imap0, ilam0])**2, cmap = 'inferno')
pl.title('Lyot plane intensity before stop (corono field)')

pl.figure(9)
pl.clf()
pl.imshow(np.abs(corono_mono_lyot_t[icase0, imap0, ilam0])**2*LyotStop2d, cmap = 'inferno')
pl.title('Lyot plane intensity after stop (corono field)')

pl.show()

#%%

f1 = pl.figure(10, figsize=(8,2))
pl.clf()
for i in range(ncase):
    exec('ax{0} = f1.add_subplot(1{1}{0})'.format(i+1,ncase))
    exec('im = ax{0}.imshow(np.abs(direct_mono_lyot_t[{1}, imap0, ilam0])**2, cmap = "inferno", vmin=0, vmax=2)'.format(i+1,ncase-1-i))
    exec('ax{0}.text(nPup/2, 0.1*nPup, "{1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,label_lst[ncase-1-i]))
    exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
    exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))

f1.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

f1.subplots_adjust(right=0.8)
cbar_ax = f1.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = f1.colorbar(im, cax=cbar_ax)
pl.suptitle('Lyot plane intensity before stop (direct field)')

#%%
f2 = pl.figure(11, figsize=(8,4))
pl.clf()
for i in range(2*ncase):
    exec('ax{0} = f2.add_subplot(2,{1},{0})'.format(i+1,ncase))
    if i < ncase: 
        exec('im = ax{0}.imshow(np.log10(np.abs(corono_mono_lyot_t[{1}, imap0, ilam0])**2), cmap = "inferno", vmin=-3, vmax=0)'.format(i+1,ncase-1-i))
        exec('ax{0}.text(nPup/2, 0.1*nPup, "{1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,label_lst[ncase-1-i]))
    else:
        exec('im = ax{0}.imshow(np.log10(np.abs(corono_mono_lyot_t[{1}, imap0, ilam0])**2*LyotStop2d), cmap = "inferno", vmin=-3, vmax=0)'.format(i+1,(ncase - 1 -i) % ncase))        
    exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
    exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))



f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
#pl.suptitle('Lyot plane intensity')
pl.savefig(str(fpath_pupil_plane_disp))

pl.show()

#%%
f2 = pl.figure(12)
pl.clf()
for i in range(ncase):
    exec('ax{0} = f2.add_subplot(1{1}{0})'.format(i+1,ncase))
    if i < ncase: 
        exec('im = ax{0}.imshow(np.log10(corono_poly_img_t[{1}, imap0]/direct_poly_img_t[{1}, imap0].max()), cmap = "inferno", vmin=-7.5, vmax=-3.5)'.format(i+1,ncase-1-i))
        exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "{1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,label_lst[ncase-1-i]))
    exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
    exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))



f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
pl.savefig(str(fpath_image_plane_disp))

pl.show()

print('ok')

#%% Intensity profiles of the direct and coronagraphic images
icase0    = 3

rad_corono = np.arange(nImg2d//2)
colors_map = pl.cm.rainbow(np.linspace(0,1,nmap))

pl.figure(31)
pl.clf()
for imap in range(nmap):
    pl.semilogy(rad_corono*Fmax2d/nImg2d, corono_poly_prf_std_t[icase0, imap]/direct_poly_img_t[icase0, imap].max(),
                label='map {0}'.format(imap), color = colors_map[imap])

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
pl.savefig(str(fpath_image_allmaps_plot))

pl.show()


#%%
"""
results from Paranal run
"""
#%%
path_root     = Path('/Users/mndiaye/Dropbox/python/zelda/ZELDA-2018')
path_data_int = path_root / '2018-04-03' / 'data'
path_data_sky = path_root / '2018-04-03_night' / 'data'
pixel_irdis = 12.25

#%%
prefix = 'aplc_internal_source'

# read images for normalisation
psf_def_img  = fits.getdata(path_data_int / '{:s}_psf_ref_image.fits'.format(prefix))
psf_def_norm_int = psf_def_img.max()

psf_zel_img = fits.getdata(path_data_int / '{:s}_psf_zel_image.fits'.format(prefix))
psf_zel_norm_int = psf_zel_img.max()

# read profiles
psf_def_prf_int = fits.getdata(path_data_int / '{:s}_psf_ref_profile.fits'.format(prefix))
psf_zel_prf_int = fits.getdata(path_data_int / '{:s}_psf_zel_profile.fits'.format(prefix))

cor_def_prf_int = fits.getdata(path_data_int / '{:s}_coro_ref_profile_std.fits'.format(prefix))
cor_zel_prf_int = fits.getdata(path_data_int / '{:s}_coro_zel_profile_std.fits'.format(prefix))

# separations
psf_sep_int = np.arange(len(psf_def_prf_int))*pixel_irdis
cor_sep_int = np.arange(cor_def_prf_int.shape[1])*pixel_irdis

#%%
#
# sky
#

prefix = 'aplc_test2'

# read images for normalisation
psf_def_img  = fits.getdata(path_data_sky / '{:s}_psf_ref_image.fits'.format(prefix))
psf_def_norm_sky = psf_def_img.max()

psf_zel_img = fits.getdata(path_data_sky / '{:s}_psf_zel_image.fits'.format(prefix))
psf_zel_norm_sky = psf_zel_img.max()

# read profiles
psf_def_prf_sky = fits.getdata(path_data_sky / '{:s}_psf_ref_profile.fits'.format(prefix))
psf_zel_prf_sky = fits.getdata(path_data_sky / '{:s}_psf_zel_profile.fits'.format(prefix))

cor_def_prf_sky = fits.getdata(path_data_sky / '{:s}_coro_ref_profile_std.fits'.format(prefix))
cor_zel_prf_sky = fits.getdata(path_data_sky / '{:s}_coro_zel_profile_std.fits'.format(prefix))

# separations
psf_sep_sky = np.arange(len(psf_def_prf_sky))*pixel_irdis
cor_sep_sky = np.arange(cor_def_prf_sky.shape[1])*pixel_irdis

#%%
# plot
fig = pl.figure(0, figsize=(5, 4))
pl.clf()

#plt.semilogy(psf_sep, psf_def_prf / psf_def_norm, color='C0', label='Default refslp')
#plt.semilogy(psf_sep, psf_zel_prf / psf_zel_norm, color='C1', label='ZELDA refslp')
    
# sky
for idx, p in enumerate(cor_def_prf_sky):
    if idx == 0:
        pl.semilogy(cor_sep_sky, p / psf_def_norm_sky, color='C0', alpha=1, label='Default [sky]')
    else:
        pl.semilogy(cor_sep_sky, p / psf_def_norm_sky, color='C0', alpha=1)
#plt.semilogy(cor_sep, np.mean(cor_def_prf, axis=0) / psf_def_norm, color='C2', lw=3, label='Default refslp')
#plt.semilogy(cor_sep, cor_def_prf[imin_def] / psf_def_norm, color='C2', lw=3, label='Default refslp')

for idx, p in enumerate(cor_zel_prf_sky):
    if idx == 0:
        pl.semilogy(cor_sep_sky, p / psf_zel_norm_sky, color='C1', alpha=1, label='ZELDA [sky]')
    else:
        pl.semilogy(cor_sep_sky, p / psf_zel_norm_sky, color='C1', alpha=1)
#plt.semilogy(cor_sep, np.mean(cor_zel_prf, axis=0) / psf_def_norm, color='C3', lw=3, label='ZELDA refslp')
#plt.semilogy(cor_sep, cor_zel_prf[imin_zel] / psf_def_norm, color='C3', lw=3, label='ZELDA refslp')

# internal source
for p in cor_def_prf_int:
    pl.semilogy(cor_sep_int, p / psf_def_norm_int, color='C2', alpha=1, lw=2, label='Default [int. source]')

for p in cor_zel_prf_int:
    pl.semilogy(cor_sep_int, p / psf_zel_norm_int, color='C3', alpha=1, lw=2, label='ZELDA [int. source]')

pl.legend(loc='upper right')

pl.xlim(0, 1000)
pl.ylim(1e-6, 1e-3)

pl.xlabel('Angular separation [as]')
pl.ylabel('Contrast')

pl.grid(which='both')
pl.tight_layout()

#%%
cor_def_prf_sky_avg = np.mean(cor_def_prf_sky, axis=0)
cor_zel_prf_sky_avg = np.mean(cor_zel_prf_sky, axis=0)
cor_def_prf_int_avg = np.mean(cor_def_prf_int, axis=0)
cor_zel_prf_int_avg = np.mean(cor_zel_prf_int, axis=0)

pl.figure(30)
pl.clf()
#pl.semilogy(cor_sep_sky, cor_zel_prf_sky_avg/psf_zel_norm_sky)
pl.semilogy(cor_sep_int, cor_zel_prf_int_avg/psf_zel_norm_int)

pl.show()

#%% Intensity profiles of the direct and coronagraphic images
icase0    = 4
imap0     = 3

rad_corono = np.arange(nImg2d//2)
colors_map = pl.cm.rainbow(np.linspace(0,1,nmap))

pl.figure(32)
pl.clf()
for imap in {imap0}:
    pl.semilogy(rad_corono*Fmax2d/nImg2d*1000.*(wv/dAper)/mas2rad, corono_poly_prf_std_t[icase0, imap]/direct_poly_img_t[icase0, imap].max(),
                '--', label='simulation - map #{0}/{1}'.format(imap, nmap-1), color = colors_map[imap])

#pl.semilogy(cor_sep_sky, cor_zel_prf_sky_avg/psf_zel_norm_sky)
if imap0 == 0:    
    pl.semilogy(cor_sep_int, cor_def_prf_int_avg/psf_def_norm_int, 'k', label='int. source (def)')
elif imap0 == 3:
    pl.semilogy(cor_sep_int, cor_zel_prf_int_avg/psf_zel_norm_int, 'k', label='int. source (zel)')
else:
    pass

pl.axvline(x=rMask*1000.*(wv/dAper)/mas2rad, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0*1000.*(wv/dAper)/mas2rad, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1*1000.*(wv/dAper)/mas2rad, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono00.xi2d.min(), xmax=corono00.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in mas')
pl.ylabel(r'1$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
pl.tight_layout()
pl.savefig(str(fpath_image_allmaps_plot))

pl.show()




