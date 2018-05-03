#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:52:23 2018

@author: mndiaye
"""

import pylab as pl
import numpy as np
import time
from pathlib import Path

from corono import corono_design as cd
from corono import corono_optim_2d as co2d

from corono.utils import to_dict

from matplotlib import cm

from astropy.io import fits



#%% parameters
"""
Parameters
"""

#nPup = corono0.params['nPup']
nPup = 200

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  4.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 4

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
corono_name   = 'SP' # 'SP' or 'APLC'
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym = True

#nlam
nlam=5

do_fits = True

#%%
"""
File reading for Pupil and Lyot stop
"""
pupil_name = 'vlt' # 'vlt' or 'sbr'
fdir = '/Users/mndiaye/Dropbox/python/pupils/'
fname = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
fpath = fdir + fname

Pupil2d = fits.getdata(fpath)
LyotStop2d = fits.getdata(fpath)


params = to_dict(nPup=nPup, rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2=CtrBtwnPix2, nlam=nlam, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym)

#params = to_dict(nPup=nPup, rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
#                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2=CtrBtwnPix2, nlam=nlam)

#%%  
""" 
Coronagraph defintion
"""

if corono_name == 'SP':
    corono0 = cd.SP2d(**params)
else:
    corono0 = cd.APLC2d(**params)

#%%
"""
Problem defintion
"""
t0 = time.time()
# Maximization of the integrated amplitude transmission of the apodizer
problem1 = co2d.MaxTau(corono=corono0, **params)
# Maximization of the contrast under L1-norm
#problem2 = co2d.MaxContrast(corono=corono0, Lnorm='L1',**params)
# Maximization of the contrast under L-infinite norm
#problem3 = co2d.MaxContrast(corono=corono0, Lnorm='Linf',**params)

#%%
"""
Display of the matrices
"""
#A1, b1, c1 = problem1.compute_matrices()
#A2, b2, c2 = problem2.compute_matrices()
#A3, b3, c3 = problem3.compute_matrices()

#pl.figure(1)
#pl.clf()
#pl.title('Matrix for MaxTau problem')
#pl.imshow(abs(A1.T)**0.25)

#pl.figure(2)
#pl.clf()
#pl.title(r'Matrix for MaxContrast problem, L$_1$-norm')
#pl.imshow(abs(A2.T)**0.25)
###
#pl.figure(3)
#pl.clf()
#pl.title(r'Matrix for MaxContrast problem, L$_\infty$-norm')
#pl.imshow(abs(A3.T)**0.25)

#%% Gurobi model of the problems

"""
Gurobi models
"""
m1 = problem1.compute_gurobi_model()
#m2 = problem2.compute_gurobi_model()
#m3 = problem3.compute_gurobi_model()

#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
Apod1 = problem1.solve_model()
#Apod2 = problem2.solve_model()
#Apod3 = problem3.solve_model()
t1 = time.time()
print('Pupil2dSym:{0}, total computation time: {1:.2f}s'.format(Pupil2dSym, t1-t0))

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
Apod1_2d = np.reshape(Apod1, (corono0.nPup, corono0.nPup))
#Apod2_2d = np.reshape(Apod2, (corono0.nPup, corono0.nPup))
#Apod3_2d = np.reshape(Apod3, (corono0.nPup, corono0.nPup))

if Pupil2dSym == True:
    Apod1_2d += np.flip(Apod1_2d, axis=0)
    Apod1_2d += np.flip(Apod1_2d, axis=1)
    

pl.figure(4)
pl.clf()
pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil transmission')

ifig = 5
if Pupil2dSym == True:
    ifig = 25
    
pl.figure(ifig)
pl.clf()
pl.imshow(Apod1_2d*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 transmission - MaxTau problem')

#pl.figure(6)
#pl.clf()
#pl.imshow(Apod2_2d*Pupil2d, cmap = cm.inferno)
#pl.title(r'Apod 2 transmission - MaxContrast problem, L$_1$-norm')
#
#pl.figure(7)
#pl.clf()
#pl.imshow(Apod3_2d*Pupil2d, cmap = cm.inferno)
#pl.title(r'Apod 3 transmission - MaxContrast problem, L$_1$-norm')

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod1_2d)
poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod1_2d)
#poly_direct_image2 = corono0.compute_direct_intensity_2d(Apod2_2d)
#poly_corono_image2 = corono0.compute_corono_intensity_2d(Apod2_2d)
#poly_direct_image3 = corono0.compute_direct_intensity_2d(Apod3_2d)
#poly_corono_image3 = corono0.compute_corono_intensity_2d(Apod3_2d)

#%% image plot

pl.figure(10)
pl.clf()
pl.imshow(poly_direct_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - direct image')

pl.figure(11)
pl.clf()
pl.imshow(poly_corono_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - coronagraphic image')

#pl.figure(12)
#pl.clf()
#pl.imshow(poly_direct_image2**0.25, cmap = cm.inferno)
#pl.title('Apod2 - direct image')
#
#pl.figure(13)
#pl.clf()
#pl.imshow(poly_corono_image2**0.25, cmap = cm.inferno)
#pl.title('Apod2 - coronagraphic image')
#
#pl.figure(14)
#pl.clf()
#pl.imshow(poly_direct_image3**0.25, cmap = cm.inferno)
#pl.title('Apod3 - direct image')
#
#pl.figure(15)
#pl.clf()
#pl.imshow(poly_corono_image3**0.25, cmap = cm.inferno)
#pl.title('Apod3 - coronagraphic image')


#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

nImg2d = corono0.params['nImg2d']


pl.figure(8)
pl.clf()
pl.title('Intensity profiles of the coronagraphic images')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='MaxTau')
#pl.semilogy(corono0.xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
#pl.semilogy(corono0.xi2d,poly_corono_image3[nImg2d//2,nImg2d//2:]/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
#pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-problem1.params['cDarkHole']), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()

#%%
pl.show()

#%%
#
fdir = Path('/Users/mndiaye/Dropbox/central storage/AMPL/PupilDataFiles/2D/General/dat/')

if corono_name == 'SP':
    fname_gen  = 'SP00_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}.fits'
else:
    fname_gen  = 'APLC_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}.fits'

fpath = fdir / pupil_name / fname_gen.format(**{key: corono0.params[key] for key in corono0.params})

if do_fits is True:
    fits.writeto(fpath, Apod1_2d, clobber=True)