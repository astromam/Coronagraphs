#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:27:14 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%% 
"""
### Initialization
"""
import numpy as np
import pylab as pl
ftsz = 16 
pl.rcParams.update({'font.size': ftsz})
from matplotlib.patches import Circle
from matplotlib import cm

import os
from pathlib import Path
from astropy.io import fits
from pyzelda.utils import aperture, imutils, zernike
import corono as coro


#%% APLC2d tests
"""
### Parameters
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


Fratio    = 40

# rMask_m   = 287e-6/2.
# rMask  = rMask_m/(wv*Fratio)
# rMask_mas = 1000.*rMask * (wv/dAper)/mas2rad
rMask_mas = 185/2. # ALC1 (145/2. mas), ALC2 (185/2. mas), ALC3 (240/2. mas) @ 1.593um
rMask = rMask_mas/(wv*180*3600*1000/np.pi/dAper)

print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))

rho0   = 5.
rho1   = 20.
nPup   = 384

bw     = 0.20#width/wv 
nFPM   = 50

nImg2dbis = 256
Fmax2dbis = nImg2dbis/(2*(wv/950e-9))
#nImg2dbis = 600
#Fmax2dbis = 60

nlambis   = 11

kw_aberr     = False
kw_2nddate   = True    
kw_skyobs    = True
kw_aftercorr = False
kw_saxo      = True

do_plot = True  

Dtel = 8
lam0D2mas = (wv/Dtel)*(180.*3600*1000/np.pi)

lam02um = wv*1e6 


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
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs').resolve()

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

fname_image_plane_plot_avg   = 'corono_poly_prf_avg_t_OPDmap={0}_plot.pdf'.format(imap0)
fname_image_plane_plot   = 'corono_poly_prf_std_t_OPDmap={0}_plot.pdf'.format(imap0)
fname_image_plane_mono_plot   = 'corono_poly_prf_std_t_OPDmap={0}_mono_plot.pdf'.format(imap0)
fname_image_allmaps_plot = 'corono_poly_prf_std_t_OPDmap=all_plot.pdf'
fname_image_allmaps_plot_comp = 'corono_poly_prf_std_t_OPDmap=all_plot_comp.pdf'
fname_image_plane_disp   = 'corono_poly_img_t_OPDmap={0}_disp.pdf'.format(imap0)
fname_pupil_plane_disp   = 'corono_poly_lyot_t_OPDmap={0}_disp.pdf'.format(imap0)
fname_image_plane_disp_all   = 'corono_poly_img_t_OPDmap={0}_disp_all0.pdf'.format(imap0)


fpath_image_plane_plot_avg   = fdir_plots / fname_image_plane_plot_avg
fpath_image_plane_plot   = fdir_plots / fname_image_plane_plot
fpath_image_plane_mono_plot   = fdir_plots / fname_image_plane_mono_plot
fpath_image_allmaps_plot = fdir_plots / fname_image_allmaps_plot
fpath_image_allmaps_plot_comp = fdir_plots / fname_image_allmaps_plot_comp
fpath_image_plane_disp   = fdir_plots / fname_image_plane_disp
fpath_pupil_plane_disp   = fdir_plots / fname_pupil_plane_disp
fpath_image_plane_disp_all   = fdir_plots / fname_image_plane_disp_all


#%% 
"""
### Display plots
"""

#% Entrance pupil

Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)

fpath = fdir_pupimages / 'Aperture.pdf'

pl.figure(1)
pl.clf()
pl.imshow(Pupil2d, cmap = 'inferno')
pl.title('Entrance pupil')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#%%  Apodization
"""
### Display of the apodizer
"""
Apod2d = fits.getdata(fpath_Apod2d)

fpath = fdir_pupimages / 'Apodizer.pdf'

extent_pup = [-0.5, 0.5, -0.5, 0.5]

f1 = pl.figure(2)
pl.clf()
ax0 = f1.add_subplot(111)
im = ax0.imshow(Apod2d*Pupil2d, cmap = cm.gray, 
          extent = extent_pup, origin = 'lower', vmin=0.0, vmax=1.0)
ax0.set_xlabel('Pupil diameter [D unit]')
#pl.imshow(Apod2d*Pupil2d, cmap = cm.Greys_r)
#ax0.title('Apodized entrance pupil')

f1.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

#f1.subplots_adjust(right=0.8)
cbar_ax = f1.add_axes([0.835, 0.20, 0.05, 0.70])
cbar    = f1.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Normalized amplitude', rotation=270, labelpad = 20)

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

#%%  Phase errors
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

#%%  Lyot Stop
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
Pupil = zernike.zernike1(1, npix=nPup)
idx_Pup = Pupil == 1.

OPDmap2d_init = np.zeros((nPup, nPup))

Tilt = zernike.zernike1(2, npix=nPup)

OPDmap2d_init[idx_Pup] = Tilt[idx_Pup]

pl.figure(50)
pl.clf()
pl.imshow(OPDmap2d_init)

aTilt = 100. #wv*(1*np.pi/180)/(2.*np.pi)
OPDmap2d_init *= aTilt



#%%
# list of parameters for each coronagraph configuration
#params00_ghost = coro.update_params(params, OPDmap2d=OPDmap2d_init, Ampmap2d = None)
#corono00_ghost = coro.design.APLC2d(**params00_ghost)

params2 = coro.update_params(params, OPDmap2d=OPDmap2d_init, Ampmap2d = None)
corono00 = coro.design.APLC2d(**params2)

#%%
"""
### Computation of the direct and coronagraphic images
"""

poly_direct_image1 = corono00.compute_direct_intensity_2d_bis(Apod2d)
poly_corono_image1 = corono00.compute_corono_intensity_2d_bis(Apod2d)

#%%
"""
### normalization of the direct and coronagraphic image in broadband light
"""
poly_direct_image1_peak = np.max(poly_direct_image1)

poly_direct_image1 /= poly_direct_image1_peak
poly_corono_image1 /= poly_direct_image1_peak

#%%
"""
### normalization of the direct and coronagraphic image in monochromatic light
"""
mono_direct_image1 = corono00.compute_direct_intensity_2d_bis(Apod2d, poly=False)
mono_corono_image1 = corono00.compute_corono_intensity_2d_bis(Apod2d, poly=False)

for ilam in range(corono00.nlam):
    mono_direct_image1_peak = np.max(mono_direct_image1[ilam])
       
    mono_direct_image1[ilam] /= mono_direct_image1_peak
    mono_corono_image1[ilam] /= mono_direct_image1_peak


#%%
"""
### Image plane intensity
"""
# radius coordinate in the image
r  = np.linspace(0,nImg2dbis//2-1,num=nImg2dbis//2)
r_lamD = (Fmax2dbis/nImg2dbis)*r 

# image center
xc = nImg2dbis//2
yc = nImg2dbis//2

# grid in x and y
xx,yy = np.meshgrid(np.arange(nImg2dbis),np.arange(nImg2dbis))

# grid in radius R
R = np.sqrt((xx-xc)**2+(yy-yc)**2)

# definition of the array profiles for the direct image
poly_direct_prf_avg = np.zeros((nImg2dbis//2))
poly_direct_prf_std = np.zeros((nImg2dbis//2))

# definition of the array profiles for the coronagraphic image
poly_corono_prf_avg = np.zeros((nImg2dbis//2))
poly_corono_prf_std = np.zeros((nImg2dbis//2))

# computation of the average and standard deviation intensity profiles for direct image 
f_poly_dir_avg = lambda r : poly_direct_image1[(R >= r -.5) & (R < r +.5)].mean()
f_poly_dir_std = lambda r : poly_direct_image1[(R >= r -.5) & (R < r +.5)].std()
poly_direct_prf_avg = np.vectorize(f_poly_dir_avg)(r)
poly_direct_prf_std = np.vectorize(f_poly_dir_std)(r)

# computation of the average and standard deviation intensity profiles for coronagraphic image 
f_poly_cor_avg = lambda r : poly_corono_image1[(R >= r -.5) & (R < r +.5)].mean()
f_poly_cor_std = lambda r : poly_corono_image1[(R >= r -.5) & (R < r +.5)].std()
poly_corono_prf_avg = np.vectorize(f_poly_cor_avg)(r)
poly_corono_prf_std = np.vectorize(f_poly_cor_std)(r)

#%%
"""
### computation of the averaged intensity profiles in monochromatic light
"""
# definition of the array profiles for the direct image
mono_direct_prf_avg = np.zeros((corono00.nlam, nImg2dbis//2))
mono_direct_prf_std = np.zeros((corono00.nlam, nImg2dbis//2))

# definition of the array profiles for the coronagraphic image
mono_corono_prf_avg = np.zeros((corono00.nlam, nImg2dbis//2))
mono_corono_prf_std = np.zeros((corono00.nlam, nImg2dbis//2))

# computation of the average and standard deviation intensity profiles for direct image 
for ilam in range(corono00.nlam):
    f_mono_dir_avg = lambda r : mono_direct_image1[ilam, (R >= r -.5) & (R < r +.5)].mean()
    f_mono_dir_std = lambda r : mono_direct_image1[ilam, (R >= r -.5) & (R < r +.5)].std()
    mono_direct_prf_avg[ilam] = np.vectorize(f_mono_dir_avg)(r)
    mono_direct_prf_std[ilam] = np.vectorize(f_mono_dir_std)(r)

# computation of the average and standard deviation intensity profiles for coronagraphic image
for ilam in range(corono00.nlam): 
    f_mono_cor_avg = lambda r : mono_corono_image1[ilam, (R >= r -.5) & (R < r +.5)].mean()
    f_mono_cor_std = lambda r : mono_corono_image1[ilam, (R >= r -.5) & (R < r +.5)].std()
    mono_corono_prf_avg[ilam] = np.vectorize(f_mono_cor_avg)(r)
    mono_corono_prf_std[ilam] = np.vectorize(f_mono_cor_std)(r)



#%% 
"""
###Intensity profiles of the direct and coronagraphic images
"""
rad_corono = np.arange(nImg2dbis//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,nmap))

i0 = 0

pl.figure(10, (8,4.5))
pl.clf()
pl.semilogy(r_lamD,poly_direct_prf_avg,
        label='current APLC', color = 'C0')

pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono00.xi2d.min(), xmax=corono00.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(f'Angular separation in $\lambda_0$/D ($\lambda_0={wv*1e6}\mu$m)')
pl.ylabel(r'Normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend(loc=1)
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100), fontsize=14)
pl.tight_layout()
pl.savefig(str(fpath_image_plane_plot_avg), transparent=True)


#%% 
"""
###Intensity profiles of the direct and coronagraphic images
"""
rad_corono = np.arange(nImg2dbis//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,nmap))

i0 = 0

pl.figure(11, (8,4.5))
pl.clf()
pl.semilogy(r_lamD, poly_corono_prf_std,
        label='current APLC', color = 'C0')

pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono00.xi2d.min(), xmax=corono00.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(f'Angular separation in $\lambda_0$/D ($\lambda_0={wv*1e6}\mu$m)')
pl.ylabel(r'1$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend(loc=1)
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100), fontsize=14)
pl.tight_layout()
pl.savefig(str(fpath_image_plane_plot), transparent=True)


#%% 
"""
### plot displays at multiple wavelengths
"""

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
    pl.semilogy(r_lamD, 5*corono_mono_prf_std_t[i0, i]/direct_mono_img_t[i0, idx_lam_peak].max(),
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

nSub = 150
nIni = (nImg2dbis-nSub)//2
nEnd = (nImg2dbis+nSub)//2

FhalfFOV = Fmax2dbis*nSub/nImg2dbis*lam0D2mas
extent_img = [-0.5*FhalfFOV, 0.5*FhalfFOV, -0.5*FhalfFOV, 0.5*FhalfFOV]

if nmap <= 10: 
    f2 = pl.figure(22)#, figsize=(7,5))
    pl.clf()
    for i in range(nmap):
        exec('ax{0} = f2.add_subplot(1,{1},{0})'.format(i+1,nmap))
        if i < nmap: 
            exec('im = ax{0}.imshow(np.log10(corono_poly_img_t[{1}][nIni:nEnd,nIni:nEnd]/direct_poly_img_t[{1}].max()), cmap = "inferno", vmin=-7.5, vmax=-3.5, extent=extent_img)'.format(i+1,i))
#            exec('ax{0}.text(nImg2dbis/2, 0.1*nImg2dbis, "map {1}", fontsize=8, horizontalalignment="center", color = "white")'.format(i+1,i))
        exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(i+1,))
        exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(i+1,))
        exec('ax{0}.set_xlabel("Angular separation [mas]")'.format(i+1,))
    
    f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    #f2.subplots_adjust(right=0.8)
    cbar_ax = f2.add_axes([0.835, 0.20, 0.05, 0.70])
    cbar    = f2.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Intensity in log scale', rotation=270, labelpad = 20)
    pl.tight_layout()
    pl.savefig(str(fpath_image_plane_disp), transparent=True)
    pl.show()

#%%
"""
Display of the coronagraphic image
"""
fname_image_plane_f_disp = 'corono_poly_img_f_nPup={0:04d}_disp.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_plots / fname_image_plane_f_disp

lw0 = 2.5

# parameters for the circle definition
cx0 = nImg2dbis/2
cy0 = nImg2dbis/2
crm = rMask*nImg2dbis/Fmax2dbis
cr0 = rho0*nImg2dbis/Fmax2dbis
cr1 = rho1*nImg2dbis/Fmax2dbis
# definition of a circle for the pupil
circle_mask = Circle((cx0, cy0), crm, color='red', fill = False, ls = '--',
                     linewidth = lw0)
circle_rho0 = Circle((cx0, cy0), cr0, color='blue', fill = False, ls = '--',
                     linewidth = lw0)
circle_rho1 = Circle((cx0, cy0), cr1, color='blue', fill = False, ls = '--',
                     linewidth = lw0)
# in the plot




f2 = pl.figure(23, figsize=(7,5))
pl.clf()
ax0 = f2.add_subplot(111)
im = ax0.imshow(np.log10(corono_poly_img_t[0]/direct_poly_img_t[0].max()), cmap = "inferno", vmin=-7.5, vmax=-3.5)
#exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "nmap={1:05d}", fontsize=16, horizontalalignment="center", color = "white")'.format(1,1))
ax0.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
ax0.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")
ax0.add_artist(circle_mask)
#ax0.add_artist(circle_rho0)
ax0.add_artist(circle_rho1)

f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.825, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Intensity in log scale', rotation=270, labelpad = 16)
if do_plot is True:
    pl.savefig(str(fpath_image_plane_f_disp), transparent=True)
pl.tight_layout()
pl.show()


#%%
lw0 = 2.5

lam00D_corono = (Fmax2dbis/nImg2dbis)*np.linspace(-nImg2dbis/2-0.5, nImg2dbis/2-0.5, nImg2dbis)

# bounds for the separations
lam00D_min = lam00D_corono.min()
lam00D_max = lam00D_corono.max()

# parameters for the circle definition
cx0 = 0#nImg2dbis/2
cy0 = 0#nImg2dbis/2
crm = rMask#*nImg2dbis/Fmax2dbis
cr0 = rho0#*nImg2dbis/Fmax2dbis
cr1 = rho1#*nImg2dbis/Fmax2dbis
# definition of a circle for the pupil
circle_mask = Circle((cx0, cy0), crm, color='red', fill = False, ls = '--',
                     linewidth = lw0)
circle_rho0 = Circle((cx0, cy0), cr0, color='blue', fill = False, ls = '--',
                     linewidth = lw0)
circle_rho1 = Circle((cx0, cy0), cr1, color='blue', fill = False, ls = '--',
                     linewidth = lw0)
# in the plot
extent_pup = [-0.5, 0.5, -0.5, 0.5]
extent_img = [lam00D_min, lam00D_max, lam00D_min, lam00D_max]


f2 = pl.figure(24, figsize=(8,4.5))
pl.clf()
ax0 = f2.add_subplot(121)
ax0.imshow(Apod2d*Pupil2d, cmap = 'inferno',
           extent = extent_pup,
           origin = 'lower')
ax0.set_title('Apodization')
ax0.set_xlabel(r'Pupil radius r in D')
#ax0.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
#ax0.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")


ax1 = f2.add_subplot(122)
im = ax1.imshow(np.log10(corono_poly_img_t[0]/direct_poly_img_t[0].max()), 
                cmap = "inferno", 
                vmin=-7.5, vmax=-3.5,
                extent = extent_img,
                origin = 'lower')
#exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "nmap={1:05d}", fontsize=16, horizontalalignment="center", color = "white")'.format(1,1))
#ax1.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
#ax1.tick_params(axis="y", which="both", left="off", right="on", labelleft="off", labelright="on")
ax1.set_title('Coronagraphic image')
ax1.set_xlim(lam00D_min, lam00D_max)
ax1.set_ylim(lam00D_min, lam00D_max)
ax1.add_artist(circle_mask)
ax1.add_artist(circle_rho0)
ax1.add_artist(circle_rho1)
ax1.set_xlabel(r'Angular separation in $\lambda_0$/D')

#ax1.autoscale(False)

f2.subplots_adjust(bottom=0.1, top=0.9, left=0.08, right=0.85,
                    wspace=0.2, hspace=0.02)

#f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.87, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Intensity in log scale', rotation=270, labelpad = 16)
if do_plot is True:
    pl.savefig(str(fpath_image_plane_disp_all), transparent=True)
pl.tight_layout()
pl.show()

print('plots: ok')

#%% 
"""
### Intensity profiles of the direct and coronagraphic images
"""
rad_corono = np.arange(nImg2dbis//2)
colors_map = pl.cm.rainbow(np.linspace(0,1,nmap))

pl.figure(25)
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
