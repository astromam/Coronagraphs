#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:40:02 2018

@author: mndiaye
"""
import pylab as pl

from corono import coronagraph_cl as cg
from corono import optim_cl as op

#%%  
''' 
coronagraph defintion
'''
corono0 = cg.APLC1d()

'''
problem defintion
'''
problem1 = op.MaxTau(corono=corono0,)
problem2 = op.MaxContrast(corono=corono0, Lnorm='L1')
problem3 = op.MaxContrast(corono=corono0, Lnorm='Linf')
#A, b, c = problem.compute_matrices()

#pl.figure(1)
#pl.clf()
#pl.imshow(abs(A)**0.25)

#%% Model of the problem
m1 = problem1.compute_gurobi_model()
m2 = problem2.compute_gurobi_model()
m3 = problem3.compute_gurobi_model()

#%% Apodizer solution for the problem
Apod1 = problem1.solve_model()
Apod2 = problem2.solve_model()
Apod3 = problem3.solve_model()

#%% Display of the apodizer
pl.figure(6)
pl.clf()
pl.plot(Apod1)

#%% Signal in intensity
poly_direct_image1 = corono0.compute_direct_intensity_1d(Apod1)
poly_corono_image1 = corono0.compute_corono_intensity_1d(Apod1)
poly_direct_image2 = corono0.compute_direct_intensity_1d(Apod2)
poly_corono_image2 = corono0.compute_corono_intensity_1d(Apod2)
poly_direct_image3 = corono0.compute_direct_intensity_1d(Apod3)
poly_corono_image3 = corono0.compute_corono_intensity_1d(Apod3)

#%% Intensity profiles of the direct and coronagraphic images
pl.figure(8)
pl.clf()
pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label='Corono')
pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image2/poly_direct_image2.max(),label='Corono')
pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
pl.semilogy(corono0.xi,poly_corono_image3/poly_direct_image3.max(),label='Corono')
pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()

#%%
pl.show()