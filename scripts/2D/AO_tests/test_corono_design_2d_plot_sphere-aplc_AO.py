#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:50:30 2019

@author: mndiaye
"""
import numpy as np
import pylab as pl
from pathlib import Path

import os
from astropy.io import fits
import corono as coro

from scipy.misc import imresize

#%% parameters
"""
### Parameters
"""
pl.close('all')
if True:
    # Telescope name
    corono_name  = 'APLC' # 'SP' or 'APLC'
    pupil_name   = 'vlt' # 'vlt' or 'sbr' or 'lvr'
    problem_name = 'MaxContrastL1' # 'MaxContrastL1' #'MaxTau' # , 'MaxContrastLinf' # #  
    solver       = 'stdgrb' # 'stdgrb' #  'gurobipy', 'scipy.linprog'
    
    MinIsland   = False
    FirstDerGlobalLim = 1.
    
    #nPup = corono0.params['nPup']
    nPup = 384
    nFPM = 50
    Fmax2d = 50
    nImg2d = 500
    
    # central wavelength (H2 filter)
    wv = 1.593e-6
    
    # mask radius in lam0/D unit
    dAper     = 8
    mas2rad   = np.pi/(180.*3600)
    rMask_m   = 287e-6/2.#372e-6/2.#
    Fratio    = 40
    rMask  = rMask_m/(wv*Fratio)
    print('Mask radius: {0:.3f} lambda_0/D at {1:.3f}um'.format(rMask, wv*1e6))
    rMask_mas = 1000.*rMask * (wv/dAper)/mas2rad
    print('Mask radius: {0:.2f} mas at {1:.3f}um'.format(rMask_mas, wv*1e6))
    
    # dark zone bounds (inner and outer edges) in lam0/D unit
    rho0 =  2.0
    rho1 = 20.0
    
    # contrast in the dark region
    cDarkHole = 6.0
    
    # tau (integrated Pupil transmission)
    tau   = 0.756
    
    # CtrBtwnPix2
    CtrBtwnPix  = True
    CtrBtwnPix2 = True
    Pupil2dSym  = False # set it True only for optimization
    
    #nlam
    bw   = 0.2
    nlam = 5

    
    do_fits = True

nlambis = 11  
nImg2dbis = 200  
Fmax2dbis = (nImg2dbis/2)*(950e-9/wv)
   

# total number of existing maps
qmap = 1000
# total number of used maps
nmap = 1000

# plot parameters
vmin0 = -8
vmax0 = 0

aplc_arr = ['aplc1', 'aplc2']
naplc = len(aplc_arr)

temp_freq = 1000

# PSD number
iPSD = 0

#%%  
""" 
### Coronagraph defintion
"""
params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
#                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)

if corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
### Problem defintion
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
### File reading path for Pupil, Apodizer, Lyot stop and AO residuals
"""
# Pupil
if True:
    fdir0 = Path('../../../data/2D/pupils/').resolve()
    if pupil_name == 'vlt':
        fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
        fname_lys = 'SPHERE/sphere_stop_ST_ALC2.fits' 
    else:
        raise ValueError('pupil_name should be vlt instead of {0}'.format(pupil_name))
    
    fpath_pup = fdir0 / fname_pup
    fpath_lys = fdir0 / fname_lys
    Pupil2d    = fits.getdata(fpath_pup)

# Lyot stop    
    LyotStop2dtmp = fits.getdata(fpath_lys)
    if nPup != 384:
        LyotStop2d = imresize(LyotStop2dtmp, (nPup, nPup))
    else:
        LyotStop2d = LyotStop2dtmp*1
    
    if solver != 'gurobipy' and solver != 'stdgrb':
        solver = 'scipy'

# Apodizer 
fdir_apod  = Path('../../../data/2D/pupils/SPHERE/').resolve()
fname      = 'SPHERE_APO1_field_transmission_map.fits'
fpath_apod = fdir_apod / fname

# AO residuals
fdir_opd = Path('../../../data/2D/AO_tests/').resolve()
fname = 'AOres_opd_nPup={0}_iPSD={1:04d}_nmap={2:04d}.fits'.format(nPup,0,qmap)
fpath_opd = fdir_opd / fname 

#%%
"""
### File reading path for the generated data
"""
fdir_pdf = Path('../../../results/2D/plots/AO_tests/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

direct_poly_img = np.empty((naplc, nImg2dbis, nImg2dbis))
corono_poly_img = np.empty((naplc, nImg2dbis, nImg2dbis))

direct_poly_img_AO = np.empty((naplc, nImg2dbis, nImg2dbis))
corono_poly_img_AO = np.empty((naplc, nImg2dbis, nImg2dbis))

direct_poly_avg = np.empty((naplc, nImg2dbis//2))
corono_poly_avg = np.empty((naplc, nImg2dbis//2))
direct_poly_std = np.empty((naplc, nImg2dbis//2))
corono_poly_std = np.empty((naplc, nImg2dbis//2))

direct_poly_avg_AO = np.empty((naplc, nImg2dbis//2))
corono_poly_avg_AO = np.empty((naplc, nImg2dbis//2))
direct_poly_std_AO = np.empty((naplc, nImg2dbis//2))
corono_poly_std_AO = np.empty((naplc, nImg2dbis//2))



for i, iaplc in enumerate(aplc_arr): 
    fdir_data = Path('../../../results/2D/data/AO_tests/{0}/'.format(iaplc)).resolve()
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    
    fname_direct = '{2}_direct_nPup={0}_nImg={1}_img.fits'.format(nPup, nImg2dbis, iaplc)
    fname_corono = '{2}_corono_nPup={0}_nImg={1}_img.fits'.format(nPup, nImg2dbis, iaplc)
    fpath_direct = fdir_data / fname_direct
    fpath_corono = fdir_data / fname_corono
    
    fname_direct_AO = '{4}_direct_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_img.fits'.format(nPup, nImg2dbis, iPSD, nmap, iaplc)
    fname_corono_AO = '{4}_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_img.fits'.format(nPup, nImg2dbis, iPSD, nmap, iaplc)
    fpath_direct_AO = fdir_data / fname_direct_AO
    fpath_corono_AO = fdir_data / fname_corono_AO
    
    # with no aberrations
    fname_direct_avg = '{2}_direct_nPup={0}_nImg={1}_avg.fits'.format(nPup, nImg2dbis, iaplc)
    fname_corono_avg = '{2}_corono_nPup={0}_nImg={1}_avg.fits'.format(nPup, nImg2dbis, iaplc)
    fname_direct_std = '{2}_direct_nPup={0}_nImg={1}_std.fits'.format(nPup, nImg2dbis, iaplc)
    fname_corono_std = '{2}_corono_nPup={0}_nImg={1}_std.fits'.format(nPup, nImg2dbis, iaplc)
    
    fpath_direct_avg = fdir_data / fname_direct_avg
    fpath_corono_avg = fdir_data / fname_corono_avg
    fpath_direct_std = fdir_data / fname_direct_std
    fpath_corono_std = fdir_data / fname_corono_std
    
    # with AO residuals
    fname_direct_avg_AO = '{4}_direct_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_avg.fits'.format(nPup, nImg2dbis, iPSD, nmap, iaplc)
    fname_corono_avg_AO = '{4}_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_avg.fits'.format(nPup, nImg2dbis, iPSD, nmap, iaplc)
    fname_direct_std_AO = '{4}_direct_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_std.fits'.format(nPup, nImg2dbis, iPSD, nmap, iaplc)
    fname_corono_std_AO = '{4}_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_std.fits'.format(nPup, nImg2dbis, iPSD, nmap, iaplc)
    
    fpath_direct_avg_AO = fdir_data / fname_direct_avg_AO
    fpath_corono_avg_AO = fdir_data / fname_corono_avg_AO
    fpath_direct_std_AO = fdir_data / fname_direct_std_AO
    fpath_corono_std_AO = fdir_data / fname_corono_std_AO

    #%%
    """
    ### Read image (no aberration)
    """
    direct_poly_img[i] = fits.getdata(fpath_direct)
    corono_poly_img[i] = fits.getdata(fpath_corono)
    
    #%%
    """
    ### Read image (with AO residuals)
    """
    direct_poly_img_AO[i] = fits.getdata(fpath_direct_AO)
    corono_poly_img_AO[i] = fits.getdata(fpath_corono_AO)
    
    #%%
    """
    ### Read profiles (with AO residuals)
    """
    # with no aberrations
    direct_poly_avg[i] = fits.getdata(fpath_direct_avg)
    corono_poly_avg[i] = fits.getdata(fpath_corono_avg)
    direct_poly_std[i] = fits.getdata(fpath_direct_std)
    corono_poly_std[i] = fits.getdata(fpath_corono_std)
    
    # with AO residuals
    direct_poly_avg_AO[i] = fits.getdata(fpath_direct_avg_AO)
    corono_poly_avg_AO[i] = fits.getdata(fpath_corono_avg_AO)
    direct_poly_std_AO[i] = fits.getdata(fpath_direct_std_AO)
    corono_poly_std_AO[i] = fits.getdata(fpath_corono_std_AO)

print('reading ok')
#%%
"""
### Plot comparison
"""
fname_image_plane_plot = 'aplcs_corono_nPup={0}_nImg={1}_iPSD={2:04d}_nmap={3:04d}_plt.pdf'.format(nPup, nImg2dbis, iPSD, nmap)
fpath_image_plane_plot = fdir_pdf / fname_image_plane_plot

rad_corono = np.arange(nImg2dbis//2)
colors_cor = pl.cm.rainbow(np.linspace(0,1,2))

lines_noturb = []
lines_siturb = []

fig = pl.figure(12)
pl.clf()
ax = fig.add_subplot(111)

for i, iaplc in enumerate(aplc_arr):
    ax.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_std_AO[i],
            color = colors_cor[i], ls='-')

    ax.semilogy(rad_corono*Fmax2dbis/nImg2dbis, 5*corono_poly_std[i]/direct_poly_img.max(),
            color = colors_cor[i], ls='--')

dummy_lines = []
dummy_lines += ax.semilogy([], [], ls='-', color='k')
dummy_lines += ax.semilogy([], [], ls='--', color='k')

dummy_lines2 = []
for i in range(naplc):
    dummy_lines2 += ax.semilogy([], [], ls='', color=colors_cor[i])

ax.axvspan(-1, rMask, alpha=0.25, color='b')
ax.set_xlabel(r'Angular separation in $\lambda_0$/D')
ax.set_ylabel(r'5$\sigma$ normalized intensity in log scale')
ax.set_xlim(-0.5, 30.5)
ax.set_ylim(3e-8, 3e-4)
ax.grid(True, which='both')
#ax.legend()

leg1 = ax.legend(dummy_lines, ['No turbulence', r'AO residuals, t={0:04d}ms'.format(int(nmap*temp_freq/1000))], 
             loc='lower right', frameon=False)
ax.add_artist(leg1);

leg2 = ax.legend(dummy_lines2, ['Current APLC', 'New APLC'],
             loc='upper right', frameon=False)
for i, text in enumerate(leg2.get_texts()):
    pl.setp(text, color = colors_cor[i])
ax.add_artist(leg2);

ax.set_title(r'Broadband intensity profile ($\Delta\lambda/\lambda_0$={0:.1f}%)'.format(bw*100,int(nmap*temp_freq/1000)))
pl.tight_layout()
pl.savefig(str(fpath_image_plane_plot), transparent=True)

pl.show()

