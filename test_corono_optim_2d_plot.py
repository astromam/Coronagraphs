#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

@author: mndiaye
"""

import pylab as pl
import numpy as np
from pathlib import Path

from corono import corono_design as cd
#from corono import corono_optim_2d as co2d

from corono.utils import to_dict

from matplotlib import cm

from astropy.io import fits

#%% parameters
"""
Parameters
"""
#nPup = corono0.params['nPup']
nPup = 50

nImg2d = 400
Fmax2d = 30

nFPM = 100 

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  4.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 4

# tau (integrated Pupil transmission)
tau   = 0.4

corono_name   = 'SP' # 'SP' or 'APLC'
ctr_btwn_pix  = True
ctr_btwn_pix2 = True
SymPupil2d = True

#nlam
nlam=1 

#%%
"""
File reading for Pupil and Lyot stop
"""
pupil_name = 'sbr' # 'vlt' or 'sbr'
fdir = Path('/Users/mndiaye/Dropbox/python/pupils/')
fname = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
fpath = fdir /  fname

Pupil2d = fits.getdata(fpath)
LyotStop2d = fits.getdata(fpath)


params = to_dict(nPup=nPup, rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 ctr_btwn_pix=ctr_btwn_pix, ctr_btwn_pix2=ctr_btwn_pix2, nlam=nlam, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 nImg2d = nImg2d, Fmax2d = Fmax2d, nFPM = nFPM,
                 SymPupil2d = SymPupil2d)



#params = to_dict(nPup=nPup, rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
#                 ctr_btwn_pix=ctr_btwn_pix, ctr_btwn_pix2=ctr_btwn_pix2, nlam=nlam)


#%%  
""" 
Coronagraph defintion
"""

if corono_name == 'SP':
    corono0 = cd.SP2d(**params)
else:
    corono0 = cd.APLC2d(**params)


#%%
#


fdir = Path('/Users/mndiaye/Dropbox/central storage/AMPL/PupilDataFiles/2D/General/dat/')

if corono_name == 'SP':
    fname_gen  = 'SP00_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}.fits'
else:
    fname_gen  = 'APLC_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}.fits'

fpath = fdir / pupil_name / fname_gen.format(**{key: corono0.params[key] for key in corono0.params})

Apod1_2d = fits.getdata(fpath,)

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
pl.figure(4)
pl.clf()
pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil transmission')

fpath = '/users/mndiaye/Desktop/Kernel_Apod/apodisation.pdf'
pl.figure(5)
pl.clf()
pl.imshow(Apod1_2d*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 transmission - MaxTau problem')
pl.savefig(fpath)

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""

if corono_name == 'APLC':
    poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod1_2d)
else:
    poly_direct_image1 = corono0.compute_direct_intensity_2d(corono0.Pupil2d)
poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod1_2d)

#%% image plot

fpath  = '/users/mndiaye/Desktop/Kernel_Apod/direct_image.pdf'
pl.figure(10)
pl.clf()
pl.imshow(poly_direct_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - direct image')
pl.savefig(fpath)

fpath  = '/users/mndiaye/Desktop/Kernel_Apod/apodized_image.pdf'
pl.figure(11)
pl.clf()
pl.imshow(poly_corono_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - apodized image')
pl.savefig(fpath)

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

nImg2d = corono0.params['nImg2d']


fpath = '/users/mndiaye/Desktop/Kernel_Apod/intensity_profiles.pdf'

pl.figure(8)
pl.clf()
pl.title('Radial intensity profiles of the images')
pl.semilogy(corono0.xi2d,poly_direct_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_corono_image1.max(),label='Apod - MaxTau')
#pl.semilogy(corono0.xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
#pl.semilogy(corono0.xi2d,poly_corono_image3[nImg2d//2,nImg2d//2:]/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
#pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()
pl.savefig(fpath)

#%%
pl.show()

#%%
#


#fpath = '/Users/mndiaye/Desktop/test.fits' 
#fits.writeto(fpath, Apod1_2d, clobber=True)

