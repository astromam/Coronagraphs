#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:40:02 2018

@author: mndiaye
"""
import pylab as pl

from corono import corono_design as cd
from corono import corono_optim as co

#%%  
""" 
Coronagraph defintion
"""
corono0 = cd.APLC1d()
#corono0 = cd.SP1d()

#%%
"""
Problem defintion
"""
# Maximization of the integrated amplitude transmission of the apodizer
problem1 = co.MaxTau(corono=corono0,)
# Maximization of the contrast under L1-norm
problem2 = co.MaxContrast(corono=corono0, Lnorm='L1')
# Maximization of the contrast under L-infinite norm
problem3 = co.MaxContrast(corono=corono0, Lnorm='Linf')

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
pl.figure(4)
pl.clf()
pl.title('Transmission profiles of the apodizers')
pl.plot(corono0.r, Apod1, label='MaxTau')
pl.plot(corono0.r, Apod2, label=r'MaxContrast, L$_1$-norm')
pl.plot(corono0.r, Apod3, label=r'MaxContrast, L$_\infty$-norm')
pl.xlabel(r'Pupil radius r')
pl.ylabel('Apodizer amplitude transmission')
pl.legend()

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
poly_direct_image1 = corono0.compute_direct_intensity_1d(Apod1)
poly_corono_image1 = corono0.compute_corono_intensity_1d(Apod1)
poly_direct_image2 = corono0.compute_direct_intensity_1d(Apod2)
poly_corono_image2 = corono0.compute_corono_intensity_1d(Apod2)
poly_direct_image3 = corono0.compute_direct_intensity_1d(Apod3)
poly_corono_image3 = corono0.compute_corono_intensity_1d(Apod3)

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""
pl.figure(5)
pl.clf()
pl.title('Intensity profiles of the coronagraphic images')
#pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label='MaxTau')
pl.semilogy(corono0.xi,poly_corono_image2/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
pl.semilogy(corono0.xi,poly_corono_image3/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-problem1.params['cDarkHole']), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()

#%%
pl.show()