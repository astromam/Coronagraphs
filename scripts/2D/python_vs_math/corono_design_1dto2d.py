#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:48:23 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""
import numpy as np
import pylab as pl
import os

from pathlib import Path
import corono as coro

pl.rcParams.update({'font.size': 12})

from scipy.interpolate import interp1d, CubicSpline


#%% parameters
"""
### Parameters
"""
pl.close('all')
if True:
    corono_name  = 'APLC' # 'APLC' or 'SP'
    problem_name = 'MaxContrastLinf' # 'MaxContrastL1' #,'MaxContrastLinf' # 'MaxTau' #
    solver       = 'stdgrb' # 'stdgrb', 'gurobipy', 'scipy.linprog'
    
    FirstDer    = True
    SecondDer   = True
    MinIsland   = False
    FirstDerLim = 0.001/2
    SecondDerLim= 0.0001/2 
    FirstDerGlobalLim = 10.
    
    nPup = 1000
    nFPM = 50
    nImg = 180
    Fmax = 45
    R    = 1
    
    bw   = 0.1
    nlam = 5
    
    PupilID    = 0.10
    
    rMask       = 3.75
    
    rMask1      = 2.0
    rMask2      = 3.0
    rMask3      = 3.5
    OPDx2       = 0.5
    OPDx3       = 0.75
    
    LyotStopID = 0.20
    LyotStopOD = 1.0
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 = 5.0
    rho1 = 40.0
    
    # contrast in the dark region
    cDarkHole = 10.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.05
    
    r   = np.arange(nPup)*R/nPup + R/(2*nPup)
    Pupil1d      = (r>PupilID)*1.0
    LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0
    
    if solver != 'gurobipy' and solver != 'stdgrb':
        solver = 'scipy'
    
    params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                     nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                     bw = bw, nlam = nlam,
                     PupilID = PupilID, rMask = rMask, 
                     rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                     OPDx2 = OPDx2, OPDx3 = OPDx3, 
                     LyotStopID = LyotStopID,
                     LyotStopOD = LyotStopOD,
                     r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                     solver = solver, problem_name = problem_name,
                     corono_name = corono_name,
                     FirstDer = FirstDer, SecondDer = SecondDer,
                     FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                     MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                     )

nlambis = 11
nImgbis = 500
Fmaxbis = 50    

#%% 2D parameters
nPup2d = 1*nPup
Fmax2d = 2*Fmaxbis
nImg2d = 2*nImgbis

CtrBtwnPix = True
CtrBtwnPix2 = False

#%%
fdir = Path('../../results/1D/').resolve()

fdir_pyth = fdir / 'dat_pyth'
fdir_plot = fdir / 'plots'

if not os.path.exists(fdir_plot):
    os.makedirs(fdir_plot)
    
if not os.path.exists(fdir_pyth):
    os.makedirs(fdir_pyth)    

#%%  
""" 
### Coronagraph defintion
"""
if corono_name == 'APLC':
    corono0 = coro.design.APLC1d(**params)
elif corono_name == 'SP':
    corono0 = coro.design.SP1d(**params)
elif corono_name == 'HDZPM':
    corono0 = coro.design.HDZPM1d(**params)
elif corono_name == 'HTZPM':
    corono0 = coro.design.HTZPM1d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
### Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_1d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
     raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#%% Apodizer solution for the problems
"""
### Apodizer solutions
"""
fname_gen  = problem1.get_filename()
fname_pyth = fname_gen + '.dat'
fpath_pyth = fdir_pyth / fname_pyth
test0 = np.loadtxt(fpath_pyth)
Apod_pyth = test0[:, 1]

#%% Display of the apodizer
"""
### Plot display of the apodizers
"""
Apod1d = Apod_pyth/Apod_pyth.max()
T_apod = np.sum(Pupil1d**2*Apod1d**2)/np.sum(Pupil1d**2)
T_coro = np.sum(Pupil1d**2*LyotStop1d*Apod1d**2)/np.sum(Pupil1d**2)


fname_pl   = fname_gen + '_apodizers_tran.pdf'
fpath      = fdir_plot / fname_pl

fname_pl   = fname_gen + '_apodizers_tran.pdf'
fpath      = fdir_plot / fname_pl

pl.figure(1, (8, 4.5))
pl.clf()
pl.plot(corono0.r, Apod1d)
pl.xlabel(r'Pupil radius r')
pl.ylabel('Normalized amplitude')
pl.xlim(-0.02, 1.02)
pl.ylim(-0.02, 1.02)
pl.axvline(x=corono0.PupilID, ymin=-0.5, ymax=2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=corono0.LyotStopID, ymin=-0.5, ymax=2, linewidth=1, color='C2', linestyle='--')
pl.text(0.4, 0.3, '{0}% bandwidth'.format(int(bw*100)))
pl.text(0.4, 0.25, 'Apodizer, T={0}%'.format(int(T_apod*100)))
pl.text(0.4, 0.2, 'Coronagraph, T={0}%'.format(int(T_coro*100)))
pl.text(PupilID+0.01, 0.05, r'd={0}%'.format(int(PupilID*100)), color='C1')
pl.text(LyotStopID+0.01, 0.05, r'd$_S$={0}%'.format(int(LyotStopID*100)), color='C2')
#pl.legend()
pl.tight_layout()
pl.show()
pl.savefig(str(fpath), transparent=True, bbox_inches='tight')

#%% Signal in intensity
"""
### Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis, nImg=nImgbis, Fmax=Fmaxbis) 

if corono_name == 'APLC':
    corono0 = coro.design.APLC1d(**params2)
elif corono_name == 'SP':
    corono0 = coro.design.SP1d(**params2)
elif corono_name == 'HDZPM':
    corono0 = coro.design.HDZPM1d(**params2)
elif corono_name == 'HTZPM':
    corono0 = coro.design.HTZPM1d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

poly_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth)
poly_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth)

mono_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth, poly=False)
mono_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth, poly=False)

#%% Intensity profiles of the direct and coronagraphic images
"""
### Display of the intensity profiles of the coronagraphic images
"""

#fname_pl = fname_gen + '_intensity.pdf'
#fpath    = fdir_plot / fname_pl
#pl.figure(4, (8, 4.5))
#pl.clf()
#pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max())
#pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
#pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
#pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
#pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#pl.xlabel(r'Angular separation in $\lambda_0$/D')
#pl.ylabel('Normalized intensity in log scale')
#pl.xlim(-0.5, 50.5)
#pl.ylim(10**(-12.2), 10**(-3.8))
#pl.text(15, 10**(-4.5), '{0}% central obstruction'.format(int(PupilID*100)))
#pl.text(15, 10**(-5.), '{0}% bandwidth'.format(int(bw*100)))
#pl.text(corono0.rMask+0.25, 10**(-5), r'm/2={0:.2f}$\lambda_0$/D'.format(corono0.rMask), color='C1')
#pl.text(corono0.rho0+0.25, 10**(-12), r'$\rho_0$={0:.1f}$\lambda_0$/D'.format(corono0.rho0), color='C2')
#pl.text(corono0.rho1+0.25, 10**(-12), r'$\rho_1$={0:.1f}$\lambda_0$/D'.format(corono0.rho1), color='C2')
##pl.legend()
#pl.tight_layout()
#pl.savefig(str(fpath), transparent=True, bbox_inches='tight')

#%% Intensity profiles of the direct and coronagraphic images
"""
### Display of the monochromatic intensity profiles of the coronagraphic images
"""

values = range(nlambis)
colors = pl.cm.rainbow(np.linspace(0,1,nlambis))

#fname_pl = fname_gen + '_intensity_mono.pdf'
#fpath    = fdir_plot / fname_pl
#pl.figure(5, (8, 4.5))
#pl.clf()
#for i in range(corono0.nlam):
#    pl.semilogy(corono0.xi,mono_corono_image1[i]/mono_direct_image1[(corono0.nlam+1)//2].max(), 
#                label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[i]), color = colors[i])
#pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
#pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
#pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
#pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#pl.xlabel(r'Angular separation in $\lambda_0$/D')
#pl.ylabel('Normalized intensity in log scale')
#pl.xlim(-0.5, 50.5)
#pl.ylim(10**(-12.2), 10**(-3.8))
#pl.legend()
#pl.tight_layout()
#pl.savefig(str(fpath), transparent=True, bbox_inches='tight')


#%% 1d interpolation 
""" 
### 1d interpolation computation
"""

x1d = corono0.r/2
y1d = Apod1d
z1d = Pupil1d
t1d = LyotStop1d

idx = int(PupilID*nPup)
fit_A1d = interp1d(x1d[idx:], y1d[idx:], kind="cubic", bounds_error=False, fill_value="extrapolate")
#f1d = CubicSpline(x1d[idx:], y1d[idx:], extrapolate=True)

x1dbis = (1/(2*nPup2d))*(np.arange(2*nPup2d)-2*nPup2d/2+1/2)
Apod1dbis = fit_A1d(x1dbis)

#%%
"""
### Apod2d generation
"""
aa = (1/nPup2d)*(np.arange(nPup2d)-nPup2d/2+1/2)
bb = 2*aa[nPup2d//2:]

xx, yy = np.meshgrid(aa, aa)
mydist = np.hypot(yy,xx)

Apod2d = np.zeros((nPup2d, nPup2d))
Apod2d = fit_A1d(mydist)

Pupil2d = coro.utils.uniform_disk(nPup2d, nPup2d/2, CtrBtwnPix=CtrBtwnPix)\
-coro.utils.uniform_disk(nPup2d,PupilID*nPup2d/2, CtrBtwnPix=CtrBtwnPix)
#            
LyotStop2d = coro.utils.uniform_disk(nPup2d, LyotStopOD*nPup2d/2, CtrBtwnPix=CtrBtwnPix)\
-coro.utils.uniform_disk(nPup2d,LyotStopID*nPup2d/2, CtrBtwnPix=CtrBtwnPix)


#%%            
pl.figure(11)
pl.clf()
pl.imshow(Apod2d*Pupil2d, cmap='inferno')
pl.title('2D Apodizer')
pl.show()

pl.figure(12)
pl.clf()
pl.imshow(Pupil2d, cmap='inferno')
pl.title('2D Pupil')
pl.show()

pl.figure(13)
pl.clf()
pl.imshow(LyotStop2d, cmap='inferno')
pl.title('2D Lyot Stop')
pl.show()

pl.figure(14, (8, 4.5))
pl.clf()
#pl.subplot(211)
pl.plot(corono0.r, Apod1d, label='1d initial')
pl.plot(x1d*2, fit_A1d(x1d), label='1d interp')
pl.plot(x1dbis*2, fit_A1d(x1dbis), label='1d interp - zp', ls = '--')
#pl.plot(bb, Apod2d[nPup2d//2, nPup2d//2:]*Pupil2d[nPup2d//2, nPup2d//2:], label='2d')
pl.xlabel(r'Pupil radius r')
pl.ylabel('Normalized amplitude')
pl.xlim(-0.02, 1.02)
pl.ylim(-0.02, 1.02)
pl.axvline(x=corono0.PupilID, ymin=-0.5, ymax=2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=corono0.LyotStopID, ymin=-0.5, ymax=2, linewidth=1, color='C2', linestyle='--')
pl.text(0.4, 0.3, '{0}% bandwidth'.format(int(bw*100)))
pl.text(0.4, 0.25, 'Apodizer, T={0}%'.format(int(T_apod*100)))
pl.text(0.4, 0.2, 'Coronagraph, T={0}%'.format(int(T_coro*100)))
pl.text(PupilID+0.01, 0.05, r'd={0}%'.format(int(PupilID*100)), color='C1')
pl.text(LyotStopID+0.01, 0.05, r'd$_S$={0}%'.format(int(LyotStopID*100)), color='C2')
pl.legend()

#pl.subplot(212)
#pl.plot(corono0.r, Apod1d-f1d(x1d), color='C1')
#pl.plot(corono0.r, Apod1d-Apod2d[nPup2d//2, nPup2d//2:]*Pupil2d[nPup2d//2, nPup2d//2:], color='C2')
#pl.ylabel('Amplitude difference')
#pl.xlim(-0.02, 1.02)
#pl.ylim(-0.005, 0.005)
#pl.tight_layout()
#pl.show()
#pl.savefig(str(fpath), transparent=True, bbox_inches='tight')

##%%
#pl.figure(15, (8, 4.5))
#pl.clf()
#pl.subplot(211)
#pl.plot(corono0.r, Apod1d, label='1d initial')
#pl.plot(x1d*2, g1d(x1d), label='1d interp')
#pl.plot(x1d*2, Pupil2d[nPup2d//2, nPup2d//2:], label='2d')
#pl.xlabel(r'Pupil radius r')
#pl.ylabel('Normalized amplitude')
#pl.xlim(-0.02, 1.02)
#pl.ylim(-0.02, 1.02)
#pl.axvline(x=corono0.PupilID, ymin=-0.5, ymax=2, linewidth=1, color='C1', linestyle='--')
#pl.axvline(x=corono0.LyotStopID, ymin=-0.5, ymax=2, linewidth=1, color='C2', linestyle='--')
#pl.text(0.4, 0.3, '{0}% bandwidth'.format(int(bw*100)))
#pl.text(0.4, 0.25, 'Apodizer, T={0}%'.format(int(T_apod*100)))
#pl.text(0.4, 0.2, 'Coronagraph, T={0}%'.format(int(T_coro*100)))
#pl.text(PupilID+0.01, 0.05, r'd={0}%'.format(int(PupilID*100)), color='C1')
#pl.text(LyotStopID+0.01, 0.05, r'd$_S$={0}%'.format(int(LyotStopID*100)), color='C2')
#pl.legend()
#
#pl.subplot(212)
#pl.plot(corono0.r, Apod1d-f1d(x1d), color='C1')
#pl.plot(corono0.r, Apod1d-Apod2d[nPup2d//2, nPup2d//2:], color='C2')
#pl.ylabel('Amplitude difference')
#pl.xlim(-0.02, 1.02)
#pl.ylim(-0.005, 0.005)
#pl.tight_layout()
#pl.show()
#pl.savefig(str(fpath), transparent=True, bbox_inches='tight')


#%%  
""" 
Coronagraph defintion
"""
params2d = coro.to_dict(nPup=nPup2d, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlambis, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = False, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name)

corono2d = coro.design.APLC2d(**params2d)

poly_direct_image2d = corono2d.compute_direct_intensity_2d(Apod2d)
poly_corono_image2d = corono2d.compute_corono_intensity_2d(Apod2d)

mono_direct_image2d = corono2d.compute_direct_intensity_2d(Apod2d, poly=False)
mono_corono_image2d = corono2d.compute_corono_intensity_2d(Apod2d, poly=False)

#%% image plot
"""
Display direct and coronagraphic images
"""
#fname = fname_gen + '_direct_image.pdf'
#fpath = fdir_pdf / fname

#pl.figure(20)
#pl.clf()
#pl.imshow(np.log10(poly_direct_image2d), cmap = 'inferno',
#vmin=-12, vmax=0)
#pl.title('Apod1 - direct image')
#pl.savefig(str(fpath))

#fname = fname_gen + '_apodized_image.pdf'
#fpath = fdir_pdf / fname

pl.figure(21)
pl.clf()
pl.imshow(np.log10(poly_corono_image2d/poly_direct_image2d.max()), cmap = 'inferno',
          vmin=-12, vmax=0)
pl.title('Apod1 - apodized image')
#pl.savefig(str(fpath))

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the broadband intensity profiles of the coronagraphic images
"""
nImg2dbis = nImg2d*1

xi2d = corono2d.xi2d
if nImg2dbis%2 == 0:
    xi2d = corono2d.xi2d_ctr

#nImg2d = corono0.params['nImg2d']
#fname = fname_gen + '_intensity_profiles.pdf'
#fpath = fdir_pdf / fname

pl.figure(30, (8, 4.5))
pl.clf()
pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(), label='1d')
pl.semilogy(xi2d,poly_corono_image2d[nImg2dbis//2,nImg2dbis//2:]/poly_direct_image2d.max(),label='2d')
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.5, 50.5)
pl.ylim(10**(-12.2), 10**(-3.8))
pl.text(15, 10**(-4.5), '{0}% central obstruction'.format(int(PupilID*100)))
pl.text(15, 10**(-5.), '{0}% bandwidth'.format(int(bw*100)))
pl.text(corono0.rMask+0.25, 10**(-5), r'm/2={0:.2f}$\lambda_0$/D'.format(corono0.rMask), color='C1')
pl.text(corono0.rho0+0.25, 10**(-12), r'$\rho_0$={0:.1f}$\lambda_0$/D'.format(corono0.rho0), color='C2')
pl.text(corono0.rho1+0.25, 10**(-12), r'$\rho_1$={0:.1f}$\lambda_0$/D'.format(corono0.rho1), color='C2')
pl.legend()
pl.tight_layout()
#pl.savefig(str(fpath), transparent=True, bbox_inches='tight')

#%%
"""
Display of the monochromatic intensity profiles of the direct images
"""
pl.figure(31, (8, 4.5))
pl.clf()
pl.semilogy(corono0.xi,mono_corono_image1[nlam//2]/mono_direct_image1[nlam//2].max(), label='1d')
pl.semilogy(xi2d,mono_corono_image2d[nlam//2, nImg2dbis//2,nImg2dbis//2:]/mono_direct_image2d[nlam//2].max(),label='2d')
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='C1', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='C2', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.xlim(-0.5, 50.5)
pl.ylim(10**(-12.2), 10**(-3.8))
pl.text(15, 10**(-4.5), '{0}% central obstruction'.format(int(PupilID*100)))
pl.text(15, 10**(-5.), '{0}% bandwidth'.format(int(bw*100)))
pl.text(corono0.rMask+0.25, 10**(-5), r'm/2={0:.2f}$\lambda_0$/D'.format(corono0.rMask), color='C1')
pl.text(corono0.rho0+0.25, 10**(-12), r'$\rho_0$={0:.1f}$\lambda_0$/D'.format(corono0.rho0), color='C2')
pl.text(corono0.rho1+0.25, 10**(-12), r'$\rho_1$={0:.1f}$\lambda_0$/D'.format(corono0.rho1), color='C2')
pl.legend()
pl.tight_layout()

#%%
pl.show()

