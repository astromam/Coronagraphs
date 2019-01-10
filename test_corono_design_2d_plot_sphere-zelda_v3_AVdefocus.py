#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:42:12 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%% Initialization
"""
### Initialization
"""
import numpy as np
import os
import pylab as pl
from pathlib import Path
from astropy.io import fits
from pyzelda.utils import aperture, imutils
import corono as coro

#%% APLC2d tests
"""
### Parameters
"""
# Coronagraph type
corono_name   = 'APLC' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = True
CtrBtwnPix2 = False
Pupil2dSym  = False

# Spectral bandwidth
wv        = 1.593e-6
width     = 52e-9

# Telescope characteristics
dAper     = 8
Fratio    = 40

# Focal plane mask
mas2rad   = np.pi/(180.*3600)
rMask_m   = 287e-6/2.
rMask  = rMask_m/(wv*Fratio)
rMask_mas = 1000.*rMask * (wv/dAper)/mas2rad
print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))

# sampling
nPup   = 384   # pupil
nFPM   = 200   # focal plane mask
nImg2d = 600   # final image plane 
Fmax2d = 60    # spatial frequencies in the final image plane

# wavelength sampling
nlam   = 5
bw     = width/wv 

# simulation configuration   
kw_aberr     = True
kw_2nddate   = True    
kw_skyobs    = False
kw_aftercorr = True
kw_saxo      = False
saxomap_i    = 0
saxomap_f    = 0

# test on the order of the min and max number of saxo phase screen
if saxomap_i <= saxomap_f:
    nsaxomap     = saxomap_f - saxomap_i + 1
else:
    raise NameError('initial saxo map (saxomap_i={0}) must be smaller than final saxo map (saxomap_f={1})!'.format(saxomap_i, saxomap_f))

# plot parameters for final image plane
rho0   = 5.
rho1   = 20.
cDarkHole   = 6

#%%
"""
### Directories
"""    
if kw_aberr is False:
    str_aberr = 'wo_aberr'
    str_date  = ''
    if kw_skyobs is True:
        str_obs   = 'sky'
    else:
        str_obs   = 'internal'
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
    imap0     = 0
    nmap      = 1
    beta_wfs  = 1./0.90
    str_saxo_tmp  = 'wo_saxo'
    if kw_2nddate is True:
        str_date = '2018-04-03'
    if kw_skyobs is True:
        str_obs  = 'sky'
    if kw_aftercorr is True:
        str_corr = 'after_correction'
        imap0    = 3
        beta_wfs = 1./0.95
    if kw_saxo is True and kw_2nddate is True:
        str_saxo = 'with_saxo'
        nmap     = nsaxomap*1
        beta_wfs = 1.0

#%%
fdir = Path('../../').resolve()

fdir_pupils  = fdir / 'data' / '2D' / 'pupils' / 'SPHERE' 
fdir_zelda   = fdir / 'data' / '2D' / 'ZELDA' / str_date / str_obs  
fdir_saxo    = fdir / 'data' / '2D' / 'ZELDA' / '2018-04-03'
fdir_plots   = fdir / 'results' / '2D' / 'plots' / 'SPHERE' / str_aberr / str_date / str_obs / str_corr  

if kw_aberr is True:
    fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_date / str_obs / str_saxo / str_corr  
else:
    fdir_results = fdir / 'results' / '2D' / 'data' / 'SPHERE' / str_aberr / str_obs / str_saxo / str_corr  

fdir_pupimages = fdir_results / 'pupimages'

if not os.path.exists(fdir_results):
    os.makedirs(fdir_results)
    
if not os.path.exists(fdir_plots):
    os.makedirs(fdir_plots)    

if not os.path.exists(fdir_pupimages):
    os.makedirs(fdir_pupimages)  

#%%
"""
### Filenames for the sources
"""
fname_Apod2d     = 'SPHERE_APO1_field_transmission_map.fits'
fname_Apod2d_OPDmapnm = 'apo_substrate_D1.fits'
fname_Ampmap2d   = 'sphere_pupil_clear_BH_field.fits'
fname_LyotStop2d = 'sphere_stop_ST_ALC2.fits'

if kw_aberr is True:
    if kw_skyobs is True:
        fname_ZELDAmapnm3d = '2018-04-01_night_ncpa_loop_700modes_5_ncpa_loop_opd.fits'
        if kw_2nddate is True:
            fname_ZELDAmapnm3d = '2018-04-03_night_ncpa_loop_sky_2_ncpa_loop_opd.fits'
    else:
        fname_ZELDAmapnm3d = '2018-04-01_ncpa_loop_700modes_2_ncpa_loop_opd.fits'        
        if kw_2nddate is True:        
            fname_ZELDAmapnm3d = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'
    
    if kw_saxo is True and kw_2nddate is True:    
        fname_SAXOmapnm3d = '2018-04-04T03_06_15-saxo_residual_turbulence.fits'
        if kw_aftercorr is True:
            fname_SAXOmapnm3d = '2018-04-04T03_12_50-saxo_residual_turbulence.fits'

#%% Filepaths for the file sources
fpath_Apod2d          = fdir_pupils / fname_Apod2d
fpath_Apod2d_OPDmapnm = fdir_pupils / fname_Apod2d_OPDmapnm
fpath_Ampmap2d        = fdir_pupils / fname_Ampmap2d

if kw_aberr is True:
    fpath_ZELDAmapnm3d = fdir_zelda  / fname_ZELDAmapnm3d   
    if kw_saxo is True:
        fpath_SAXOmapnm3d = fdir_saxo / fname_SAXOmapnm3d
    
fpath_LyotStop2d = fdir_pupils / fname_LyotStop2d

#%%

label_lst = ['no errors']
ncase = len(label_lst)

#%%
"""
### Filepaths for the file results
"""        
fname_direct_poly_img_t     = 'direct_poly_img_nmap={0:05d}_t.fits'.format(nmap)
fname_corono_poly_img_t     = 'corono_poly_img_nmap={0:05d}_t.fits'.format(nmap)
fpath_direct_poly_img_t     = fdir_results / fname_direct_poly_img_t
fpath_corono_poly_img_t     = fdir_results / fname_corono_poly_img_t

fname_direct_poly_img_f     = 'direct_poly_img_nmap={0:05d}_f.fits'.format(nmap)
fname_corono_poly_img_f     = 'corono_poly_img_nmap={0:05d}_f.fits'.format(nmap)
fpath_direct_poly_img_f     = fdir_results / fname_direct_poly_img_f
fpath_corono_poly_img_f     = fdir_results / fname_corono_poly_img_f

#%%
fname_direct_poly_prf_avg_f = 'direct_poly_prf_nmap={0:05d}_avg_f.fits'.format(nmap)
fname_corono_poly_prf_avg_f = 'corono_poly_prf_nmap={0:05d}_avg_f.fits'.format(nmap)
fname_direct_poly_prf_std_f = 'direct_poly_prf_nmap={0:05d}_std_f.fits'.format(nmap)
fname_corono_poly_prf_std_f = 'corono_poly_prf_nmap={0:05d}_std_f.fits'.format(nmap)
fpath_direct_poly_prf_avg_f = fdir_results / fname_direct_poly_prf_avg_f
fpath_corono_poly_prf_avg_f = fdir_results / fname_corono_poly_prf_avg_f
fpath_direct_poly_prf_std_f = fdir_results / fname_direct_poly_prf_std_f
fpath_corono_poly_prf_std_f = fdir_results / fname_corono_poly_prf_std_f

#%%
fname_image_plane_plot   = 'corono_poly_prf_std_t_nmap={0:05d}_OPDmap={0}_plot.pdf'.format(nmap, imap0)
fname_image_plane_disp   = 'corono_poly_img_t_nmap={0:05d}_OPDmap={1}_disp.pdf'.format(nmap, imap0)
fname_image_plane_f_disp = 'corono_poly_img_f_nmap={0:05d}_disp.pdf'.format(nmap)

fpath_image_plane_plot   = fdir_plots / fname_image_plane_plot
fpath_image_plane_disp   = fdir_plots / fname_image_plane_disp
fpath_image_plane_f_disp = fdir_plots / fname_image_plane_f_disp

#%% Entrance pupil
"""
### File reading
"""
# Pupil
if kw_skyobs is True:
    Pupil2d = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0)
else:
    Pupil2d = aperture.disc(nPup, nPup/2)

fpath = fdir_pupimages / 'Aperture.pdf'

#%%
pl.figure(1)
pl.clf()
pl.imshow(Pupil2d, cmap = 'inferno')
pl.title('Entrance pupil')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#%% Apodization
Apod2d = fits.getdata(fpath_Apod2d)

#%%
fpath = fdir_pupimages / 'Apodizer.pdf'

pl.figure(2)
pl.clf()
pl.imshow(Apod2d*Pupil2d, cmap = 'inferno')
pl.title('Apodized entrance pupil')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)


#%% Apodization OPD map
Apod2d_OPDmapnm = fits.getdata(fpath_Apod2d_OPDmapnm)
Apod2d_OPDmapnm[np.isnan(Apod2d_OPDmapnm)] = 0

#%%
fpath = fdir_pupimages / 'Apodizer_OPDmapnm.pdf'

pl.figure(20)
pl.clf()
pl.imshow(Apod2d_OPDmapnm, cmap = 'inferno')
pl.title('Apodizer OPD map nm')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#%% Amplitude errors
if kw_aberr is True:
    Ampmap2d = fits.getdata(fpath_Ampmap2d)

    fpath = fdir_pupimages / 'AmplMap.pdf'

    pl.figure(3)
    pl.clf()
    pl.imshow(Pupil2d*Ampmap2d, cmap = 'inferno')
    pl.title('Amplitude map')
    pl.tight_layout()
    pl.savefig(str(fpath), transparent=True)

#%% Phase errors
if kw_aberr is True:
    OPDmapnm3d = fits.getdata(fpath_ZELDAmapnm3d)
    OPDmapnm2d = OPDmapnm3d[imap0]
    OPDmap2d   = (beta_wfs*OPDmapnm2d+Apod2d_OPDmapnm)*1e-9
        
    if kw_saxo is True:
        SAXOmapnm3d_tmp = fits.getdata(fpath_SAXOmapnm3d)
        nsaxo_all = len(SAXOmapnm3d_tmp)        
        pupil_tmp = aperture.sphere_saxo_pupil()
        pupil = np.round(imutils.scale(pupil_tmp, 0, new_dim=(384,384), method='interp'))

        # rescale NCPA map
        SAXOmapnm3d = []
        for i in range(nmap):
            SAXOmapnm3d.append(imutils.scale(SAXOmapnm3d_tmp[i], 0, new_dim=(384,384), method='interp'))
        
        SAXOmapnm3d = np.asarray(SAXOmapnm3d)

        pl.figure(6)
        pl.clf()
        pl.imshow(SAXOmapnm3d[0], cmap = 'inferno')
        pl.title('Saxo phase map')

    pl.figure(4)
    pl.clf()
    pl.imshow(OPDmap2d, cmap = 'inferno')
    pl.title('Phase map')

#%% Lyot Stop
LyotStop2d = fits.getdata(fpath_LyotStop2d)

#%%
fpath = fdir_pupimages / 'LyotStop.pdf'

pl.figure(5)
pl.clf()
pl.imshow(LyotStop2d, cmap = 'inferno')
pl.title('Lyot stop')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#%%
"""
Image reading
"""
#%% array initialization
# read the images for each map
direct_poly_img_t = fits.getdata(fpath_direct_poly_img_t)
corono_poly_img_t = fits.getdata(fpath_corono_poly_img_t)

# read the averaged image
direct_poly_img_f = fits.getdata(fpath_direct_poly_img_f)
corono_poly_img_f = fits.getdata(fpath_corono_poly_img_f)

# read the averaged and standard deviation profiles of the images
direct_poly_prf_avg_f = fits.getdata(fpath_direct_poly_prf_avg_f)
corono_poly_prf_avg_f = fits.getdata(fpath_corono_poly_prf_avg_f)
direct_poly_prf_std_f = fits.getdata(fpath_direct_poly_prf_std_f)
corono_poly_prf_std_f = fits.getdata(fpath_corono_poly_prf_std_f)

#%%
if corono_name != 'APLC':
    raise NameError('Check the name of the coronagraph!')
    
params = coro.to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                 rMask = rMask,
                 Pupil2dSym = Pupil2dSym, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                 CtrBtwnPix=CtrBtwnPix,
                 CtrBtwnPix2 = CtrBtwnPix2, 
                 nlam=nlam, bw = bw, wv =wv,
                 rho0   = rho0, rho1 = rho1, cDarkHole = cDarkHole,
                 OPDmap2d = None, Ampmap2d = None)
corono00 = coro.design.APLC2d(**params)

#%% Intensity profiles of the direct and coronagraphic images
"""
### Plot parameters
"""
rad_corono = np.arange(nImg2d//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,nmap))

lam0D2mas = (wv/8.)*(360*60*60*1000/(2.*np.pi))

x_lam0D = rad_corono*Fmax2d/nImg2d
x_mas   = x_lam0D*lam0D2mas

kw_mas = True
if kw_mas is True:
    fac   = lam0D2mas*1
    unit  = 'mas'
else:
    fac   = 1.
    unit  = '$\lambda_0$/D'

x_abs = x_lam0D*fac

#%%
"""
### Display plot and images
"""
i0 = 0

pl.figure(12)
pl.clf()
pl.semilogy(x_abs, corono_poly_prf_std_f/direct_poly_img_f.max(),
        label='map {0}'.format(imap0), color = colors_cor[i0])

pl.axvline(x=rMask*fac, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0*fac, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1*fac, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono00.xi2d.min()*fac, xmax=corono00.xi2d.max()*fac, 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in {0}'.format(unit))
pl.ylabel(r'1$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
pl.tight_layout()
if kw_mas is False:
    pl.savefig(str(fpath_image_plane_plot), transparent=True)


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
        ax1.set_title('{0}'.format(str_corr))
        
    f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    f2.subplots_adjust(right=0.8)
    cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar    = f2.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
    pl.savefig(str(fpath_image_plane_disp), transparent=True)
    pl.tight_layout()
    pl.show()

#%%
f2 = pl.figure(24, figsize=(8,4.5))
pl.clf()
exec('ax{0} = f2.add_subplot(1,{1},{0})'.format(1,1))
exec('im = ax{0}.imshow(np.log10(np.fliplr(corono_poly_img_f)/direct_poly_img_f.max()), cmap = "inferno", vmin=-7.5, vmax=-3.5)'.format(1))
exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "nmap={1:05d} (flip lr)", fontsize=16, horizontalalignment="center", color = "white")'.format(1,nmap))
exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(1,))
exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(1,))

f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

f2.subplots_adjust(right=0.8)
ax1.set_title('{0}'.format(str_corr))
cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
pl.savefig(str(fpath_image_plane_f_disp), transparent=True)
pl.tight_layout()

pl.show()

#%%
print('ok')
