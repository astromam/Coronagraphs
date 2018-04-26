#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:52:23 2018

@author: mndiaye
"""

import pylab as pl
import numpy as np

from corono import corono_design as cd
from corono import corono_optim as co

from corono.utils import to_dict

#%% parameters
"""
Parameters
"""
# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  5.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 6.0

# tau (integrated Pupil transmission)
tau   = 0.5

# ctr2
ctr  = True
ctr2 = True

#nlam
nlam=5 

params = to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 ctr=ctr, ctr2=ctr2, nlam=nlam)

#%%  
""" 
Coronagraph defintion
"""
corono0 = cd.APLC2d(**params)

#%%
"""
Problem defintion
"""
# Maximization of the integrated amplitude transmission of the apodizer
problem1 = co.MaxTau(corono=corono0, **params)
# Maximization of the contrast under L1-norm
problem2 = co.MaxContrast(corono=corono0, Lnorm='L1',**params)
# Maximization of the contrast under L-infinite norm
problem3 = co.MaxContrast(corono=corono0, Lnorm='Linf',**params)

#%%
"""
Display of the matrices
"""
A1, b1, c1 = problem1.compute_matrices()
A2, b2, c2 = problem2.compute_matrices()
A3, b3, c3 = problem3.compute_matrices()

pl.figure(1)
pl.clf()
pl.title('Matrix for MaxTau problem')
pl.imshow(abs(A1.T)**0.25)

pl.figure(2)
pl.clf()
pl.title(r'Matrix for MaxContrast problem, L$_1$-norm')
pl.imshow(abs(A2.T)**0.25)
##
pl.figure(3)
pl.clf()
pl.title(r'Matrix for MaxContrast problem, L$_\infty$-norm')
pl.imshow(abs(A3.T)**0.25)

#%% Gurobi model of the problems
"""
Gurobi models
"""
m1 = problem1.compute_gurobi_model()
m2 = problem2.compute_gurobi_model()
m3 = problem3.compute_gurobi_model()

#%% Apodizer solution for the problems
"""
Apodizer solutions
"""
Apod1 = problem1.solve_model()
Apod2 = problem2.solve_model()
Apod3 = problem3.solve_model()

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
Apod1_2d = np.reshape(Apod1, (corono0.nPup, corono0.nPup))
Apod2_2d = np.reshape(Apod2, (corono0.nPup, corono0.nPup))
Apod3_2d = np.reshape(Apod3, (corono0.nPup, corono0.nPup))


pl.figure(4)
pl.clf()
pl.imshow(corono0.Pupil2d)
pl.title('Pupil transmission')

pl.figure(5)
pl.clf()
pl.imshow(Apod1_2d)
pl.title('Apodizer transmission - MaxTau problem')

pl.figure(6)
pl.clf()
pl.imshow(Apod2_2d)
pl.title(r'Apodizer transmission - MaxContrast problem, L$_1$-norm')

pl.figure(7)
pl.clf()
pl.imshow(Apod3_2d)
pl.title(r'Apodizer transmission - MaxContrast problem, L$_1$-norm')

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod1_2d)
poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod1_2d)
poly_direct_image2 = corono0.compute_direct_intensity_2d(Apod2_2d)
poly_corono_image2 = corono0.compute_corono_intensity_2d(Apod2_2d)
poly_direct_image3 = corono0.compute_direct_intensity_2d(Apod3_2d)
poly_corono_image3 = corono0.compute_corono_intensity_2d(Apod3_2d)

#%% image plot

pl.figure(10)
pl.clf()
pl.imshow(poly_direct_image1**0.25)
pl.title('Apod1 - direct image')

pl.figure(11)
pl.clf()
pl.imshow(poly_corono_image1**0.25)
pl.title('Apod1 - coronagraphic image')

pl.figure(12)
pl.clf()
pl.imshow(poly_direct_image2**0.25)
pl.title('Apod2 - direct image')

pl.figure(13)
pl.clf()
pl.imshow(poly_corono_image2**0.25)
pl.title('Apod2 - coronagraphic image')

pl.figure(14)
pl.clf()
pl.imshow(poly_direct_image3**0.25)
pl.title('Apod3 - direct image')

pl.figure(15)
pl.clf()
pl.imshow(poly_corono_image3**0.25)
pl.title('Apod3 - coronagraphic image')


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
pl.semilogy(corono0.xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
pl.semilogy(corono0.xi2d,poly_corono_image3[nImg2d//2,nImg2d//2:]/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
#pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-problem1.params['cDarkHole']), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()

#%%
pl.show()
