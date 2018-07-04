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
from corono import corono_design as cd
from corono.utils import to_dict, uniform_disk, radius_disk

from pathlib import Path
from astropy.io import fits

from pyzelda.utils import aperture


#%% APLC2d tests
"""
tests on APLC 2d class
"""
corono_name   = 'APLC' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = True
CtrBtwnPix2 = False
SymPupil2d  = False
cDarkHole   = 6

it_OPDmap = 3

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

fdir_pupils = fdir / 'pupils' / '2D' / 'SPHERE' 
fdir_data   = fdir / 'results' / '2D' / 'SPHERE' / 'data'
fdir_plot   = fdir / 'results' / '2D' / 'SPHERE' / 'plot'

if not os.path.exists(fdir_data):
    os.makedirs(fdir_data)
    
if not os.path.exists(fdir_plot):
    os.makedirs(fdir_plot)    

fname_Apod2d     = 'SPHERE_APO1_field_transmission_map.fits'
#fname_Apod2d = 'sphere_pupil_APO1_BH.fits'
fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
fname_OPDmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'
fname_LyotStop2d = 'sphere_stop_ST_ALC2.fits'

fpath_Apod2d     = fdir_pupils / fname_Apod2d
fpath_Ampmap2d   = fdir_pupils / fname_Ampmap2d
fpath_OPDmapnm3d = fdir_pupils / fname_OPDmapnm3d
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d

fname_direct_poly_img_t     = 'direct_poly_img_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_corono_poly_img_t     = 'corono_poly_img_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_direct_poly_prf_avg_t = 'direct_poly_prf_avg_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_corono_poly_prf_avg_t = 'corono_poly_prf_avg_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_direct_poly_prf_std_t = 'direct_poly_prf_std_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_corono_poly_prf_std_t = 'corono_poly_prf_std_t_OPDmap={0}.fits'.format(it_OPDmap)

fpath_direct_poly_img_t     = fdir_data / fname_direct_poly_img_t
fpath_corono_poly_img_t     = fdir_data / fname_corono_poly_img_t
fpath_direct_poly_prf_avg_t = fdir_data / fname_direct_poly_prf_avg_t
fpath_corono_poly_prf_avg_t = fdir_data / fname_corono_poly_prf_avg_t
fpath_direct_poly_prf_std_t = fdir_data / fname_direct_poly_prf_std_t
fpath_corono_poly_prf_std_t = fdir_data / fname_corono_poly_prf_std_t

fname_direct_mono_img_t     = 'direct_mono_img_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_corono_mono_img_t     = 'corono_mono_img_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_direct_mono_prf_avg_t = 'direct_mono_prf_avg_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_corono_mono_prf_avg_t = 'corono_mono_prf_avg_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_direct_mono_prf_std_t = 'direct_mono_prf_std_t_OPDmap={0}.fits'.format(it_OPDmap)
fname_corono_mono_prf_std_t = 'corono_mono_prf_std_t_OPDmap={0}.fits'.format(it_OPDmap)

fpath_direct_mono_img_t     = fdir_data / fname_direct_mono_img_t
fpath_corono_mono_img_t     = fdir_data / fname_corono_mono_img_t
fpath_direct_mono_prf_avg_t = fdir_data / fname_direct_mono_prf_avg_t
fpath_corono_mono_prf_avg_t = fdir_data / fname_corono_mono_prf_avg_t
fpath_direct_mono_prf_std_t = fdir_data / fname_direct_mono_prf_std_t
fpath_corono_mono_prf_std_t = fdir_data / fname_corono_mono_prf_std_t

fname_direct_mono_lyot_re_t     = 'direct_mono_lyot_t_re_OPDmap={0}.fits'.format(it_OPDmap)
fname_direct_mono_lyot_im_t     = 'direct_mono_lyot_t_im_OPDmap={0}.fits'.format(it_OPDmap)
fname_corono_mono_lyot_re_t     = 'corono_mono_lyot_t_re_OPDmap={0}.fits'.format(it_OPDmap)
fname_corono_mono_lyot_im_t     = 'corono_mono_lyot_t_im_OPDmap={0}.fits'.format(it_OPDmap)

fpath_direct_mono_lyot_re_t     = fdir_data / fname_direct_mono_lyot_re_t
fpath_direct_mono_lyot_im_t     = fdir_data / fname_direct_mono_lyot_im_t
fpath_corono_mono_lyot_re_t     = fdir_data / fname_corono_mono_lyot_re_t
fpath_corono_mono_lyot_im_t     = fdir_data / fname_corono_mono_lyot_im_t

#%%
fname_image_plane_plot = 'corono_poly_prf_std_t_OPDmap={0}_plot.pdf'.format(it_OPDmap)
fname_image_plane_disp = 'corono_poly_img_t_OPDmap={0}_disp.pdf'.format(it_OPDmap)
fname_pupil_plane_disp = 'corono_poly_lyot_t_OPDmap={0}_disp.pdf'.format(it_OPDmap)

fpath_image_plane_plot = fdir_plot / fname_image_plane_plot
fpath_image_plane_disp = fdir_plot / fname_image_plane_disp
fpath_pupil_plane_disp = fdir_plot / fname_pupil_plane_disp


#%% Entrance pupil
#Pupil2d = fits.getdata(fpath_Pupil2d)
#Pupil2d = uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
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
OPDmapnm2d = OPDmapnm3d[it_OPDmap]
OPDmap2d   = OPDmapnm2d*1e-9

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

##%%
if corono_name != 'APLC':
    raise NameError('Check the name of the coronagraph!')
    
params = to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                 rMask = rMask,
                 SymPupil2d = SymPupil2d, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                 CtrBtwnPix=CtrBtwnPix,
                 CtrBtwnPix2 = CtrBtwnPix2, 
                 nlam=nlam, bw = bw, wv =wv,
                 rho0   = rho0, rho1 = rho1, cDarkHole = cDarkHole,
                 OPDmap2d = None, Ampmap2d = None)
corono00 = cd.APLC2d(**params)

#%%
label_lst  = ['all errors', 'phase errors only', 'ampl. errors only', 'no errors']

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
    pl.semilogy(rad_corono*Fmax2d/nImg2d, corono_poly_prf_std_t[icase]/direct_poly_img_t[icase].max(),
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
    pl.semilogy(rad_corono*Fmax2d/nImg2d, corono_mono_prf_std_t[icase0,i]/direct_mono_img_t[icase0, idx_lam_peak].max(),
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
pl.imshow(np.abs(direct_mono_lyot_t[icase0, ilam0])**2, cmap = 'inferno')
pl.title('Lyot plane intensity before stop (direct field)')

pl.figure(7)
pl.clf()
pl.imshow(np.abs(direct_mono_lyot_t[icase0, ilam0])**2*LyotStop2d, cmap = 'inferno')
pl.title('Lyot plane intensity after stop (direct field)')


pl.figure(8)
pl.clf()
pl.imshow(np.abs(corono_mono_lyot_t[icase0, ilam0])**2, cmap = 'inferno')
pl.title('Lyot plane intensity before stop (corono field)')

pl.figure(9)
pl.clf()
pl.imshow(np.abs(corono_mono_lyot_t[icase0, ilam0])**2*LyotStop2d, cmap = 'inferno')
pl.title('Lyot plane intensity after stop (corono field)')

pl.show()

#%%

f1 = pl.figure(10, figsize=(8,2))
pl.clf()
for i in range(ncase):
    exec('ax{0} = f1.add_subplot(14{0})'.format(i+1))
    exec('im = ax{0}.imshow(np.abs(direct_mono_lyot_t[{1}, ilam0])**2, cmap = "inferno", vmin=0, vmax=2)'.format(i+1,ncase-1-i))
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
    exec('ax{0} = f2.add_subplot(24{0})'.format(i+1))
    if i < ncase: 
        exec('im = ax{0}.imshow(np.log10(np.abs(corono_mono_lyot_t[{1}, ilam0])**2), cmap = "inferno", vmin=-3, vmax=0)'.format(i+1,ncase-1-i))
        exec('ax{0}.text(nPup/2, 0.1*nPup, "{1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,label_lst[ncase-1-i]))
    else:
        exec('im = ax{0}.imshow(np.log10(np.abs(corono_mono_lyot_t[{1}, ilam0])**2*LyotStop2d), cmap = "inferno", vmin=-3, vmax=0)'.format(i+1,(ncase - 1 -i) % ncase))        
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
    exec('ax{0} = f2.add_subplot(14{0})'.format(i+1))
    if i < ncase: 
        exec('im = ax{0}.imshow(np.log10(corono_poly_img_t[{1}]/direct_poly_img_t[{1}].max()), cmap = "inferno", vmin=-7.5, vmax=-3.5)'.format(i+1,ncase-1-i))
        exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "{1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,label_lst[ncase-1-i]))
    exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
    exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))



f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
#pl.suptitle('Lyot plane intensity')
pl.savefig(str(fpath_image_plane_disp))

pl.show()


print('ok')