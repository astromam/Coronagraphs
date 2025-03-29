#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 23:27:25 2023

@author: mndiaye
"""
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
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/').resolve()

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
Robustness to spectral bandwidth
"""

#%%
"""
### Star image in monochormatic light
"""   
nlam_ter = 11
bw_ter   = 0.20


#%%
"""
### Computation of the normalization factor
"""
# Parameters for the monochromatic PSF for a pupil without apodizer nor Lyot stop   
params_N = coro.update_params(params, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis, 
                                nlam = nlam_ter, bw = bw_ter, Pupil2d = Pupil2d, 
                                LyotStop2d = Pupil2d)

# Coronagraph object definition including previous parameters
if corono_name != 'APLC':
    raise NameError('{0}: Not an APLC!'.format(corono_name))
corono_N = coro.design.APLC2d(**params_N)

# Computation of the monochromatic psf of the telescope aperture
direct_poly_img_N = corono_N.compute_direct_intensity_2d(Pupil2d, poly=True)

# normalization term wrt to the PSF of the pupil without apodizer nor Lyot stop  
direct_poly_img_N_peak = direct_poly_img_N.max()

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
                                nlam = nlam_ter, bw = bw_ter, Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,)

# coronagraph object
if corono_name != 'APLC':
    raise NameError('{0}: Not an APLC!'.format(corono_name))
corono_S = coro.design.APLC2d(**params_S)



#%%
"""
### Planet image in monochormatic light
"""   
# vector of angular separation for the computation of planet transmission
sep_min  = 0.0
sep_max  = Fmax2dbis//2
sep_stp  = Fmax2dbis/nImg2dbis #0.5

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
corono_poly_img_P = np.zeros((nsep, nImg2dbis, nImg2dbis))
print('computation of the planet coronagraphic image')
t0 = time.time()
for l in range(nsep):
    params_P = coro.update_params(params_S, OPDmap2d = opd_arr[l]) 
    corono_P = coro.design.APLC2d(**params_P)
    corono_poly_img_P[l]  = corono_P.compute_corono_intensity_2d(Apod2d, poly = True)
t1 = time.time()
print('computation time: {0:.2f}s'.format(t1-t0))

# normalization of the planet coronagraphic image
corono_poly_img_P /= direct_poly_img_N_peak



#%%
"""
### Eta_S and Eta_P computation
"""
# photometric aperture circular or ring radius in lam/D
phot_rad = 0.7

# array of angular distances in the final image plane
xxi,yyi  = np.meshgrid(np.arange(nImg2dbis)-nImg2dbis//2+val, np.arange(nImg2dbis)-nImg2dbis//2+val)
mydist = (Fmax2dbis/nImg2dbis)*np.hypot(yyi,xxi)        

eta_P = np.zeros((nsep))

circ_aper = np.zeros((nImg2dbis, nImg2dbis))

sum_circ_aper = np.zeros((nsep))

print('eta_P computation')
t0 = time.time()
for i in range(nsep):    
    # compute averaged intensity of the planet at its location within a photmetric aperture of 0.7lam/D
    mydist_P = (Fmax2dbis/nImg2dbis)*np.hypot(yyi,xxi-sep_arr[i]*nImg2dbis/Fmax2dbis)        
    ind_P = (mydist_P <= phot_rad)
    circ_aper[ind_P] = 1.
    sum_circ_aper[i] = np.sum(circ_aper)
    eta_P[i] = np.sum(circ_aper*corono_poly_img_P[i])/np.sum(circ_aper)
    circ_aper[ind_P] = 0.
t1 = time.time()
print('computation time: {0}s'.format(t1-t0))

#%%
mydist_N = (Fmax2dbis/nImg2dbis)*np.hypot(yyi,xxi)        
ind_N = (mydist_N <= phot_rad)
circ_aper[ind_N] = 1.
sum_circ_aper[i] = np.sum(circ_aper)
circ_aper[ind_N] = 1.
eta_N = np.sum(circ_aper*direct_poly_img_N/direct_poly_img_N_peak)/np.sum(circ_aper)
    

#%%
fname_image_plane_f_disp = 'corono_poly_bw_throughput_nPup={0}_disp.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_plots / fname_image_plane_f_disp


pl.figure(40, (8, 4.5))
pl.clf()
pl.plot(sep_arr, eta_P, color='C0', label='current APLC')
pl.xlabel(f'Angular separation in $\lambda_0$/D ($\lambda_0={wv*1e6}\mu$m)')
pl.ylabel(r'Planet throughput $\eta_P$')
pl.xlim(-1, 31)
pl.ylim(-0.01, 0.41)
pl.axvline(x=rMask, ymin=0, ymax =1, linewidth=1, color='r', linestyle='--')
pl.grid()
pl.legend(loc=1)
pl.tight_layout()
if do_plot is True:
    pl.savefig(str(fpath_image_plane_f_disp), transparent=True)


#%%
"""
### Apodizer throughput
"""
TT_apod = np.sum((Apod2d*Pupil2d)**2)/np.sum(Pupil2d**2)
TT_coro = np.sum((Apod2d*Pupil2d*LyotStop2d)**2)/np.sum(Pupil2d**2)

print(f'TT_apod = {TT_apod*100:.1f}%')
print(f'TT_coro = {TT_coro*100:.1f}%')

