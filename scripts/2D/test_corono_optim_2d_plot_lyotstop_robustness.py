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
from astropy.io import fits
import corono as coro

#%% parameters
"""
Parameters
"""
pl.close('all')
if False:
    # Telescope name
    pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'

    # optimization parameters
    problem_name = 'MaxTau' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
    solver       = 'stdgrb' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland    = False
    FirstDerGlobalLim = 1.
    
    # sampling
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
    corono_name   = 'SP' # 'SP' or 'APLC'
    CtrBtwnPix  = True
    CtrBtwnPix2 = True
    Pupil2dSym  = False # set it True only for optimization
    
    #nlam
    bw   = 0.1
    nlam = 5

    # maximum pixel shift along a given axis for Lyot stop 
    pix_max   = 0
    
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
        
    circ = coro.utils.uniform_disk(nPup, 0.9*nPup/2, CtrBtwnPix=True)
    LyotStop2d *= circ
    
    # List of Lyot stops for the design optimization
    LyotStop2d_t = [LyotStop2d]
    
    # List construction for Lyot stop position shifts
    pix_t = []
    if pix_max >= 1:
        pix_pos_t = 1+np.arange(pix_max)
        pix_neg_t = - pix_pos_t
        pix_t = list(-pix_pos_t) + list(pix_pos_t)
        pix_t.sort()
    
    # List construstion for the Lyot stops 
    for j in range(2):
        for i in range(len(pix_t)):
            LyotStop2d_t.append(np.roll(LyotStop2d, pix_t[i], axis=j))

    # number of coronagraph configuration
    ncorono      = len(LyotStop2d_t)
    print('# of coronagraph configurations: {0}'.format(ncorono))

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
    
    # list of parameters for each coronagraph configuration
    params_t = []
    for k in range(ncorono):
        params_t.append(coro.update_params(params, LyotStop2d=LyotStop2d_t[k])) 

    corono_t = []

#%%
"""
Working directories
"""
fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name

fdir_pdf = Path('../../results/2D/plots/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

#%%  
""" 
Coronagraph defintion
"""
if False:
    if corono_name == 'SP':
        corono_t.append(coro.design.SP2d(**params))
    elif corono_name == 'APLC':
        for k in range(ncorono):
            corono_t.append(coro.design.APLC2d(**params_t[k])) 
    else:
        raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
if False:
    if problem_name == 'MaxTau':
        # Maximization of the integrated amplitude transmission of the apodizer
        problem1 = coro.optim_2d.MaxTau(corono=corono_t, **params)
    elif problem_name == 'MaxContrastL1':
        # Maximization of the contrast under L1-norm
        problem1 = coro.optim_2d.MaxContrast(corono=corono_t, Lnorm='L1',**params)
    elif problem_name == 'MaxContrastLinf':
        # Maximization of the contrast under L-infinite norm
        problem1 = coro.optim_2d.MaxContrast(corono=corono_t, Lnorm='Linf',**params)
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
for k in [0]:
    fname = fname_gen + '_pupil_ampl.pdf'
    fpath = fdir_pdf / fname

    pl.figure(4)
    pl.clf()
    pl.imshow(corono_t[k].Pupil2d, cmap = 'Greys_r')
    pl.title('Pupil transmission')
    pl.tight_layout()
    pl.savefig(str(fpath), transparent=True)
    
    fname = fname_gen + '_apodisation_ampl.pdf'
    fpath = fdir_pdf / fname
    
    pl.figure(5)
    pl.clf()
    pl.imshow(Apod_pyth*corono_t[k].Pupil2d, cmap = 'Greys_r')
    pl.title('Apod 1 transmission - ' + problem_name + ' problem - '+ solver)
    pl.tight_layout()
    pl.savefig(str(fpath), transparent=True)
    

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)

params2_t  = []
for k in range(ncorono):
    params2_t.append(coro.update_params(params_t[k], nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis)) 

corono_t = []
if corono_name == 'SP':
    corono_t.append(coro.design.SP2d(**params2_t[0]))
elif corono_name == 'APLC':
    for k in range(ncorono):
        corono_t.append(coro.design.APLC2d(**params2_t[k])) 
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

poly_direct_image_t = [] 
poly_corono_image_t = []

mono_direct_image_t = []
mono_corono_image_t = []

values = range(nlambis)
colors = pl.cm.rainbow(np.linspace(0,1,nlambis))

res_intensity_t = []

for k in range(ncorono):
    if corono_name == 'APLC':
        poly_direct_image_t.append(corono_t[k].compute_direct_intensity_2d(Apod_pyth))
    else:
        poly_direct_image_t.append(corono_t[k].compute_direct_intensity_2d(corono_t[k].Pupil2d))
    poly_corono_image_t.append(corono_t[k].compute_corono_intensity_2d(Apod_pyth))

#%% image plot
    """
    Display direct and coronagraphic images
    """
    fname = fname_gen + '_lyotstop_ampl_config={0}.pdf'.format(k)
    fpath = fdir_pdf / fname
    
    pl.figure(6)
    pl.clf()
    pl.imshow(corono_t[k].LyotStop2d, cmap = 'Greys_r')
    pl.title('Lyot stop - config={0}'.format(k))
    pl.tight_layout()
    pl.savefig(str(fpath), transparent=True)


    fname = fname_gen + '_direct_image_config={0}.pdf'.format(k)
    fpath = fdir_pdf / fname

    pl.figure(10*k)
    pl.clf()
    pl.imshow(np.log10(poly_direct_image_t[k]), cmap = 'inferno', vmin=-7, vmax=0.)
    pl.title('Apod1 - direct image - config={0}'.format(k))
    cbar = pl.colorbar()
    cbar.set_label('Normalized intensity in log scale')
    pl.tight_layout()
    pl.savefig(str(fpath), transparent=True)
    
    fname = fname_gen + '_apodized_image_config={0}.pdf'.format(k)
    fpath = fdir_pdf / fname
    
    pl.figure(10*k+1)
    pl.clf()
    pl.imshow(np.log10(poly_corono_image_t[k]), cmap = 'inferno', vmin=-7, vmax=0.)
    pl.title('Apod1 - apodized image - config={0}'.format(k))
    cbar = pl.colorbar()
    cbar.set_label('Normalized intensity in log scale')
    pl.tight_layout()
    pl.savefig(str(fpath), transparent=True)

#%% Intensity profiles of the direct and coronagraphic images
    """
    Display of the intensity profiles of the coronagraphic images
    """
    xi2d = corono_t[k].xi2d
    if nImg2dbis%2 == 0:
        xi2d = corono_t[k].xi2d_ctr        
    
    nImg2d = corono_t[k].params['nImg2d']
    fname = fname_gen + '_intensity_profiles_config={0}.pdf'.format(k)
    fpath = fdir_pdf / fname
    
    pl.figure(10*k+2)
    pl.clf()
    pl.title('Radial intensity profiles of the images - config={0}'.format(k))
    if corono_name == 'SP':
        pl.semilogy(xi2d,poly_corono_image_t[k][nImg2dbis//2,nImg2dbis//2:]/poly_corono_image_t[k].max(),label=solver)
    else:
        pl.semilogy(xi2d,poly_corono_image_t[k][nImg2dbis//2,nImg2dbis//2:]/poly_direct_image_t[k].max(),label=solver)    
    pl.axvline(x=corono_t[k].rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axvline(x=corono_t[k].rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axhline(10**(-cDarkHole), xmin=corono_t[k].xi.min(), xmax=corono_t[k].xi.max(), linewidth=1, color='k', linestyle='--')
    pl.xlabel(r'Angular separation in $\lambda_0$/D')
    pl.ylabel('Normalized intensity in log scale')
    pl.ylim(1e-9, 2e0)
    pl.legend()
    pl.tight_layout()
    pl.savefig(str(fpath), transparent=True)

#%%

    mono_direct_image_t.append(corono_t[k].compute_direct_intensity_2d(Apod_pyth, poly=False))
    mono_corono_image_t.append(corono_t[k].compute_corono_intensity_2d(Apod_pyth, poly=False))

    pl.figure(k*10+3)
    pl.clf()
    pl.title('Radial intensity profiles of the images - config={0}'.format(k))
    for i in range(corono_t[k].nlam):
        if corono_name == 'SP':
            pl.semilogy(xi2d,mono_corono_image_t[k][i, nImg2dbis//2,nImg2dbis//2:]/mono_corono_image_t[k].max(), '-',
                        label=r'{0:.2f}$\lambda_0$'.format(corono_t[k].lam_t[i]),
                        color = colors[i])
        else:
            pl.semilogy(xi2d,mono_corono_image_t[k][i, nImg2dbis//2,nImg2dbis//2:]/mono_direct_image_t[k][(corono_t[k].nlam+1)//2].max(), 
                        '-',
                        label=r'{0:.2f}$\lambda_0$'.format(corono_t[k].lam_t[i]),  color = colors[i])
    pl.axvline(x=corono_t[k].rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axvline(x=corono_t[k].rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axhline(10**(-cDarkHole), xmin=corono_t[k].xi.min(), xmax=corono_t[k].xi.max(), linewidth=1, color='k', linestyle='--')
    pl.xlabel(r'Angular separation in $\lambda_0$/D')
    pl.ylabel('Normalized intensity in log scale')
    pl.ylim(1e-9, 2e0)
    pl.legend()
    pl.tight_layout()

#%%
    dz2d, rad2d = corono_t[k].generate_area()
    dz = np.reshape(dz2d, (corono_t[k].nImg2d**2))

    aaa     = np.arange(corono_t[k].nImg2d**2)
    idx_dz  = list(aaa[dz])  
    ndz     = len(idx_dz)
   
    poly_corono_image_1d = np.reshape(poly_corono_image_t[k], nImg2d**2)
    res_intensity_t.append(np.mean(poly_corono_image_1d[idx_dz])/poly_direct_image_t[k].max())

#%%
#pix_t   = np.asarray([0, -1, 1, -1, 1])
#Dfrac_t = pix_t*(1/nPup)
#
#mylist0 = [1,0,2]
#mylist1 = [3,0,4]     
#
#fname = fname_gen + '_intensity_Lyot_stop_position.pdf'
#fpath = fdir_pdf / fname
#
#    
#pl.close('all')    
#pl.figure(0)
#pl.clf()
#pl.title('Mean DZ intensity vs Lyot stop position')
#pl.plot(Dfrac_t[mylist1], np.asarray(res_intensity_t)[mylist1]*10**(cDarkHole+1), 'x-',
#            label='x-axis')
#pl.plot(Dfrac_t[mylist0], np.asarray(res_intensity_t)[mylist0]*10**(cDarkHole+1), 'x-', 
#            label='y-axis')
#pl.xlabel(r'Lyot stop offset in aperture diameter D unit')
#pl.ylabel(r'Normalized mean intensity (x {0})'.format(10**(-cDarkHole-1)))
#pl.ylim(0, 5)
#pl.legend()
#pl.tight_layout()
#pl.savefig(str(fpath), transparent=True)
#
pl.show()



