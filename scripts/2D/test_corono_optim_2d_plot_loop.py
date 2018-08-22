#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

@author: mndiaye
"""
import numpy as np
import pylab as pl
from pathlib import Path

import os
from matplotlib import cm
from astropy.io import fits
import corono as coro

#%% parameters
"""
Parameters
"""
pl.close('all')

# Telescope name
pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxContrastLinf' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
solver       = 'stdgrb' # 'stdgrb' #  'gurobipy', 'scipy.linprog'

MinIsland   = True
FirstDerGlobalLim = 100.

#nPup = corono0.params['nPup']
nPup = 50
nFPM = 50
Fmax2d = 50
nImg2d = 500

# mask radius in lam0/D unit
rMask = 4.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  5.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 7.0

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
corono_name   = 'APLC' # 'SP' or 'APLC'
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True # set it True only for optimization

#nlam
bw   = 0.1
nlam = 5


do_fits = True

nlambis = 11    
Fmax2dbis = 50
nImg2dbis = 500    

#%%
"""
File reading for Pupil and Lyot stop
"""
if False:
    fdir = Path('../../pupils/2D/').resolve()
    if pupil_name == 'lvr':
        fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
        fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
    else:
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
        fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    
    fpath_pup = fdir / fname_pup
    fpath_lys = fdir / fname_lys
    Pupil2d    = fits.getdata(fpath_pup)
    LyotStop2d = fits.getdata(fpath_lys)
    
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
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)

#%%
"""
Working directories
"""
fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name

fdir_plot = Path('../../results/2D/plots/').resolve()
if not os.path.exists(fdir_plot):
    os.makedirs(fdir_plot)

#%%  
nFirstDerGlobalLim  = 36
stepFirstDerGlobalLim = 10.
FirstDerGlobalLim_t = stepFirstDerGlobalLim*np.arange(nFirstDerGlobalLim)
Apod_t = []
poly_direct_image_t = []
poly_corono_image_t = []

for istep, val in enumerate(FirstDerGlobalLim_t):
    print('{0}/{1}'.format(istep+1,nFirstDerGlobalLim))

    params = coro.update_params(params, FirstDerGlobalLim = val)
 
    
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
    
    poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod_pyth)
    poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod_pyth)

    peaknorm = 1./poly_direct_image1.max()
    
    poly_direct_image1 *= peaknorm
    poly_corono_image1 *= peaknorm
    
    poly_direct_image_t.append(poly_direct_image1)
    poly_corono_image_t.append(poly_corono_image1)
    
    mono_direct_image1 = corono0.compute_direct_intensity_2d(Apod_pyth, poly=False)
    mono_corono_image1 = corono0.compute_corono_intensity_2d(Apod_pyth, poly=False)

    
    #%% image plot
    """
    Display direct and coronagraphic images
    """
    xi2d = corono0.xi2d
    if nImg2dbis%2 == 0:
        xi2d = corono0.xi2d_ctr


    fname_pl = problem_name + '_apodizer{0:03d}'.format(istep) 
    fpath    = fdir_plot / fname_pl
    
    pl.figure(2, (8,3))
    pl.clf()

    pl.subplot(121)
    pl.imshow(Apod_pyth*corono0.Pupil2d/Apod_pyth.max(), cmap = cm.Greys_r)
    cbar = pl.colorbar()
    cbar.set_label('Apodizer amplitude')
#    pl.legend()
    
    pl.subplot(122)
    pl.semilogy(xi2d,poly_corono_image1[nImg2dbis//2,nImg2dbis//2:],label=solver)    
    pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
    pl.xlabel(r'Angular separation in $\lambda_0$/D')
    pl.ylabel('Normalized intensity in log scale')
    pl.ylim(1e-12, 1e-3)
#    pl.legend()
    pl.tight_layout()
    pl.savefig(str(fpath))

    
#%%
pl.show()
    
#%%
fname = problem1.get_filename(nlam = nlambis) + '_apod_vs_FirstDerGlobalLim.pdf'
fpath = fdir_plot / fname

yticks_ind = list(5*np.arange(1+nFirstDerGlobalLim//5))

aa = np.arange(nFirstDerGlobalLim)

# imshow for different apodizers
pl.figure(3,(8,4.5))
pl.clf()
pl.imshow(np.log10(np.asarray(poly_corono_image_t)[:, nImg2dbis//2, nImg2dbis//2:]),
          aspect='auto', vmin = -9., vmax = -3.,
          extent=[np.min(corono0.xi),np.max(corono0.xi),-0.5,nFirstDerGlobalLim-.5],
          origin='lower',cmap='inferno')
pl.yticks(aa[yticks_ind],FirstDerGlobalLim_t[yticks_ind])
pl.title('Broadband intensity profiles for different apodizers')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel(r'Global 1st derivative limit $\beta$')
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')

pl.tight_layout()
pl.savefig(str(fpath))
    

#%%
os.system('say "your program has finished"')