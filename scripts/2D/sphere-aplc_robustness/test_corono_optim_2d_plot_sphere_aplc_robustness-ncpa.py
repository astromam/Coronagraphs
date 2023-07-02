#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:43:27 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

import numpy as np
import pylab as pl
from pathlib import Path
from pyzelda.utils import aperture, imutils, zernike

import os
from matplotlib import cm
from astropy.io import fits
import corono as coro

#from scipy.misc import imresize
from skimage.transform import resize as imresize

#%% parameters
"""
Parameters
"""
pl.close('all')
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'vlt' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxContrastL1' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
    solver       = 'stdgrb' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    Binarity    = False
    FirstDerGlobalLim = 1.
    BinarityReg       = 0.1
    LSRobustness = False
    
    #nPup = corono0.params['nPup']
    nPup = 100
    nFPM = 50
    Fmax2d = 22.5
    nImg2d = 45
    
    # mask radius in lam0/D unit
    rMask = 2.252
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  2.0
    rho1 = 10.0
    
    # contrast in the dark region
    cDarkHole = 6.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.756
    
    # CtrBtwnPix2
    CtrBtwnPix  = True
    CtrBtwnPix2 = True
    Pupil2dSym  = False # set it True only for optimization
    ImPart      = True
    
    #nlam
    bw   = 0.2
    nlam = 5
   
    do_fits = True

nlambis = 11    
Fmax2dbis = 60
nImg2dbis = 600

do_plot = True    

kw_correction = True

#%%
"""
File reading for Pupil and Lyot stop
"""
if True:
#    fdir = Path('../../data/2D/pupils/').resolve()
    fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()
    if pupil_name == 'lvr':
        fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
        fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
    elif pupil_name == 'vlt':
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
        fname_lys = 'SPHERE/sphere_stop_ST_ALC2.fits' 
    else:
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
        fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    
    fpath_pup = fdir / fname_pup
    fpath_lys = fdir / fname_lys
    Pupil2d    = fits.getdata(fpath_pup)
    LyotStop2dtmp = fits.getdata(fpath_lys)
    LyotStop2d = imresize(LyotStop2dtmp, (nPup, nPup))
    
    if solver != 'gurobipy' and solver != 'stdgrb':
        solver = 'scipy'

params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg,
                 ImPart = ImPart, LSRobustness = LSRobustness)

#%%
"""
Working directories
"""
#fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name
fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name

fdir_pdf = Path('../../results/2D/plots/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

#%%
fname_direct_poly_img_f     = 'direct_poly_img_f.fits'
fname_corono_poly_img_f     = 'corono_poly_img_f.fits'
fpath_direct_poly_img_f     = fdir_pdf / fname_direct_poly_img_f
fpath_corono_poly_img_f     = fdir_pdf / fname_corono_poly_img_f

fname_direct_poly_prf_avg_f = 'direct_poly_prf_avg_f.fits'
fname_corono_poly_prf_avg_f = 'corono_poly_prf_avg_f.fits'
fname_direct_poly_prf_std_f = 'direct_poly_prf_std_f.fits'
fname_corono_poly_prf_std_f = 'corono_poly_prf_std_f.fits'
fpath_direct_poly_prf_avg_f = fdir_pdf / fname_direct_poly_prf_avg_f
fpath_corono_poly_prf_avg_f = fdir_pdf / fname_corono_poly_prf_avg_f
fpath_direct_poly_prf_std_f = fdir_pdf / fname_direct_poly_prf_std_f
fpath_corono_poly_prf_std_f = fdir_pdf / fname_corono_poly_prf_std_f

fname_direct_mono_img_t     = 'direct_mono_img_t.fits'
fname_corono_mono_img_t     = 'corono_mono_img_t.fits'
fpath_direct_mono_img_t     = fdir_pdf / fname_direct_mono_img_t
fpath_corono_mono_img_t     = fdir_pdf / fname_corono_mono_img_t

fname_direct_mono_prf_avg_t = 'direct_mono_prf_avg_t.fits'
fname_corono_mono_prf_avg_t = 'corono_mono_prf_avg_t.fits'
fname_direct_mono_prf_std_t = 'direct_mono_prf_std_t.fits'
fname_corono_mono_prf_std_t = 'corono_mono_prf_std_t.fits'
fpath_direct_mono_prf_avg_t = fdir_pdf / fname_direct_mono_prf_avg_t
fpath_corono_mono_prf_avg_t = fdir_pdf / fname_corono_mono_prf_avg_t
fpath_direct_mono_prf_std_t = fdir_pdf / fname_direct_mono_prf_std_t
fpath_corono_mono_prf_std_t = fdir_pdf / fname_corono_mono_prf_std_t


#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_2d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#%%
"""
Read files
"""
fname_gen = problem1.get_filename()
fname     = fname_gen + '.fits'
fpath     = fdir / fname

Apod_pyth = fits.getdata(fpath,)

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
pl.figure(4)
pl.clf()
pl.imshow(corono0.Pupil2d, cmap = 'inferno')
pl.title('Pupil transmission')

fname = fname_gen + '_apodisation_ampl_nPup={0}.pdf'.format(nPup)
fpath = fdir_pdf / fname

#pl.figure(5)
#pl.clf()
#pl.imshow(Apod_pyth*corono0.Pupil2d, cmap = cm.Greys_r)
#pl.title('Apod 1 transmission - MaxTau problem - '+ solver)
#pl.savefig(str(fpath))

pl.figure(2)
pl.clf()
pl.imshow(Apod_pyth*corono0.Pupil2d, cmap = 'inferno')
pl.title('Apodized entrance pupil')
pl.tight_layout()
if do_plot is True:
    pl.savefig(str(fpath), transparent=True)

#%% Amplitude errors
fpath_Ampmap2d = '/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/Coronagraphs/data/2D/pupils/SPHERE/sphere_pupil_clear_BH_field.fits'
Ampmap2d = fits.getdata(fpath_Ampmap2d)

fname = fname_gen + '_amplitude_map.pdf'
fpath = fdir_pdf / fname

pl.figure(3)
pl.clf()
pl.imshow(Ampmap2d, cmap = 'inferno')
pl.title('Amplitude map')
pl.tight_layout()
if do_plot is True:
    pl.savefig(str(fpath), transparent=True)


#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis) 

if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params2)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

if corono_name == 'APLC':
    direct_poly_img_f = corono0.compute_direct_intensity_2d(Apod_pyth)
    direct_mono_img_t = corono0.compute_direct_intensity_2d(Apod_pyth, poly=False)
else:
    direct_poly_img_f = corono0.compute_direct_intensity_2d(corono0.Pupil2d)
    direct_mono_img_t = corono0.compute_direct_intensity_2d(corono0.Pupil2d, poly=False)
corono_poly_img_f = corono0.compute_corono_intensity_2d(Apod_pyth)
corono_mono_img_t = corono0.compute_corono_intensity_2d(Apod_pyth, poly=False)

direct_poly_prf_avg_f, rad_direct = imutils.profile(direct_poly_img_f, type='mean')
corono_poly_prf_avg_f, rad_corono = imutils.profile(corono_poly_img_f, type='mean')
direct_poly_prf_std_f, rad_direct = imutils.profile(direct_poly_img_f, type='std')
corono_poly_prf_std_f, rad_corono = imutils.profile(corono_poly_img_f, type='std')

#%%
direct_mono_prf_avg_t = np.zeros((nlambis, nImg2dbis//2))
corono_mono_prf_avg_t = np.zeros((nlambis, nImg2dbis//2))

direct_mono_prf_std_t = np.zeros((nlambis, nImg2dbis//2))
corono_mono_prf_std_t = np.zeros((nlambis, nImg2dbis//2))

#%%

for i in range(corono0.nlam):
    direct_mono_prf_avg_t[i], rad_direct = imutils.profile(direct_mono_img_t[i], type='mean')
    corono_mono_prf_avg_t[i], rad_corono = imutils.profile(corono_mono_img_t[i], type='mean')
    direct_mono_prf_std_t[i], rad_direct = imutils.profile(direct_mono_img_t[i], type='std')
    corono_mono_prf_std_t[i], rad_corono = imutils.profile(corono_mono_img_t[i], type='std')    

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

#val = 0
#if CtrBtwnPix2 is True:
#    val = 1/2

#val = 0
#if nImg2d%2 == 0:
#    val = 1/2
#
#xx,yy  = np.meshgrid(np.arange(nImg2d)-nImg2d//2+val, np.arange(nImg2d)-nImg2d//2+val)
#mydist = (Fmax2d/nImg2d)*np.hypot(yy,xx)
#xi2d = mydist[nImg2d//2,nImg2d//2:]

xi2d = corono0.xi2d
if nImg2dbis%2 == 0:
    xi2d = corono0.xi2d_ctr


#%%

fname_image_plane_f_disp = 'corono_poly_img_f_nPup={0}_disp.pdf'.format(nPup)
fpath_image_plane_f_disp = fdir_pdf / fname_image_plane_f_disp

f2 = pl.figure(23, figsize=(8,4.5))
pl.clf()
exec('ax{0} = f2.add_subplot(1,{1},{0})'.format(1,1))
exec('im = ax{0}.imshow(np.log10(corono_poly_img_f/direct_poly_img_f.max()), cmap = "inferno", vmin=-7.5, vmax=-3.5)'.format(1))
#exec('ax{0}.text(nImg2d/2, 0.1*nImg2d, "nmap={1:05d}", fontsize=16, horizontalalignment="center", color = "white")'.format(1,1))
exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(1,))
exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(1,))

f2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.85, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('intensity in log scale', rotation=270, labelpad = 10)
if do_plot is True:
    pl.savefig(str(fpath_image_plane_f_disp), transparent=True)
pl.tight_layout()
pl.show()

#%% plot displays at multiple wavelengths

fname_image_plane_mono_plot = 'corono_poly_prf_std_t_mono_nPup={0}_plot.pdf'.format(nPup)
fpath_image_plane_mono_plot = fdir_pdf / fname_image_plane_mono_plot


values = range(nlambis)
colors_wv = pl.cm.rainbow(np.linspace(0,1,nlambis))

if corono0.nlam > 1:
    idx_lam_peak = (corono0.nlam+1)//2
else:
    idx_lam_peak = 0

# Intensity profiles of the direct and coronagraphic images
pl.figure(12)
pl.clf()
for i in range(nlambis):
    pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_mono_prf_std_t[i]/direct_mono_img_t[idx_lam_peak].max(),
                label=r'{0:.3f}$\lambda_0$'.format(corono0.lam_t[i]), color = colors_wv[i])
    
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi2d.min(), xmax=corono0.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'5$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title('Intensity profile in monochromatic light')
pl.tight_layout()
if do_plot is True:
    pl.savefig(str(fpath_image_plane_mono_plot), transparent=True)

pl.show()


#%% Intensity profiles of the direct and coronagraphic images

fname_image_plane_plot = 'corono_poly_prf_std_t_nPup={0}_plot.pdf'.format(nPup)
fpath_image_plane_plot = fdir_pdf / fname_image_plane_plot

rad_corono = np.arange(nImg2dbis//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,1))

i0 = 0

pl.figure(11)
pl.clf()
pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_prf_std_f/direct_poly_img_f.max(),
        label='map {0}'.format(0), color = colors_cor[i0])

pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi2d.min(), xmax=corono0.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'5$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
pl.tight_layout()
if do_plot is True:
    pl.savefig(str(fpath_image_plane_plot), transparent=True)

#%%
"""
Robustness to NPCA
"""

fdir_ncpa_int  = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/Coronagraphs/data/2D/ZELDA/2018-04-03/internal') 
fname_ncpa_int = '2018-04-03_ncpa_loop_700modes_ncpa_loop_opd.fits'
fpath_ncpa_int = fdir_ncpa_int / fname_ncpa_int

fdir_ncpa_sky  = Path('/Users/mndiaye/Library/CloudStorage/OneDrive-UniversitéNiceSophiaAntipolis/data/Coronagraphs/data/2D/ZELDA/2018-04-03/sky') 
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

OPDmap2d_int = imresize(OPDmap2d_int_t[imap_int], (nPup, nPup))*1e-9
OPDmap2d_sky = imresize(OPDmap2d_sky_t[imap_sky], (nPup, nPup))*1e-9
       
OPDmap2d_t = [OPDmap2d_int, OPDmap2d_sky]
 
#%%
# number of coronagraph configuration
ncorono      = len(OPDmap2d_t)
print('# of coronagraph configurations: {0}'.format(ncorono))

#%%
      
# list of parameters for each coronagraph configuration
params_t = []
for k in range(ncorono):
    Ampmap2d_tmp = imresize(Ampmap2d, (nPup, nPup))
    params_t.append(coro.update_params(params, OPDmap2d=OPDmap2d_t[k], Ampmap2d = Ampmap2d_tmp)) 

corono_t = []

params2_t  = []
for k in range(ncorono):
    params2_t.append(coro.update_params(params_t[k], nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis)) 

if corono_name == 'SP':
    corono_t.append(coro.design.SP2d(**params2_t[0]))
elif corono_name == 'APLC':
    for k in range(ncorono):
        corono_t.append(coro.design.APLC2d(**params2_t[k])) 
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
pl.figure(1)
pl.clf()
pl.imshow(Ampmap2d_tmp)
pl.show()


#%%
val = 0
if nImg2dbis%2 == 0:
    val = 1/2

sepbis=2.5
septer=5.0


# array of angular distances in the final image plane
xx,yy  = np.meshgrid(np.arange(nImg2dbis)-nImg2dbis//2+val, np.arange(nImg2dbis)-nImg2dbis//2+val)
mydist = (Fmax2dbis/nImg2dbis)*np.hypot(yy,xx)        
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
        direct_poly_img_aberr_t.append(corono_t[k].compute_direct_intensity_2d(Apod_pyth))
    else:
        direct_poly_img_aberr_t.append(corono_t[k].compute_direct_intensity_2d(corono_t[k].Pupil2d))
    corono_poly_img_aberr_t.append(corono_t[k].compute_corono_intensity_2d(Apod_pyth))

direct_poly_img_aberr_t = np.asarray(direct_poly_img_aberr_t)
corono_poly_img_aberr_t = np.asarray(corono_poly_img_aberr_t)


#%%

direct_poly_img_aberr_prf_avg_t = np.zeros((ncorono, nImg2dbis//2))
corono_poly_img_aberr_prf_avg_t = np.zeros((ncorono, nImg2dbis//2))
direct_poly_img_aberr_prf_std_t = np.zeros((ncorono, nImg2dbis//2))
corono_poly_img_aberr_prf_std_t = np.zeros((ncorono, nImg2dbis//2))

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
fpath_image_plane_disp = fdir_pdf / fname_image_plane_disp

if len(corono_t) <= 9:
    f2 = pl.figure(30, figsize=(10,4.5))
    pl.clf()
    for i in range(ncorono):
        exec('ax{2} = f2.add_subplot({0},{1},{2})'.format(1,ncorono,i+1))
        exec('im = ax{0}.imshow(np.log10(corono_poly_img_aberr_t[{1}]/direct_poly_img_aberr_t[{1}].max()), cmap = "inferno", vmin=-7.5, vmax=-3.5)'.format(i+1,i))
        exec('ax{0}.text(nImg2dbis/2, 0.1*nImg2dbis, "{1} NCPA", fontsize=8, horizontalalignment="center", color = "black")'.format(i+1,ncpa_name[i]))
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
fpath_ncpa_plot = fdir_pdf / fname_ncpa_plot


plot_lines = []

pl.figure(32)
pl.clf()
pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_prf_std_f/direct_poly_img_f.max(),
                    label='No NCPA'.format(0), color = colors_ncpa[0])
for i in range(ncorono):
    pl.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5.*corono_poly_img_aberr_prf_std_t[i]/direct_poly_img_aberr_t[i].max(),
                    label='{0} NCPA'.format(ncpa_name[i]), color = colors_ncpa[i+1])


pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi2d.min(), xmax=corono0.xi2d.max(), 
           linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'5$\sigma$ normalized intensity in log scale')
pl.ylim(3e-8, 3e-4)
pl.legend()
pl.title(r'Intensity profile in broadband light ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100))
pl.tight_layout()
pl.savefig(str(fpath_ncpa_plot), transparent=True)

pl.show()
