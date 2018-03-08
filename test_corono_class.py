#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 23:07:48 2018

@author: mndiaye
"""

#%% Initialization
import numpy as np
import pylab as pl
from corono import coronagraph_cl as cg

##%% APLC tests   
#'''
#tests on the APLC 1d class
#'''
#a = cg.APLC1d()
#Apod = np.ones((a.params['nPup']))
#direct_matrix_re, direct_matrix_im = a.prop_direct_matrix()
#corono_matrix_re, corono_matrix_im = a.prop_corono_matrix()
#
#direct_int_a = a.compute_direct_intensity_1d(Apod)
#corono_int_a = a.compute_corono_intensity_1d(Apod)
#
##%% plot displays
#pl.figure(1)
#pl.clf()
#pl.imshow(direct_matrix_re[:,2,:])
#
#pl.figure(2)
#pl.clf()
#pl.imshow(direct_matrix_im[:,2,:])
#
#pl.figure(3)
#pl.clf()
#pl.imshow(corono_matrix_re[:,2,:])
#
#pl.figure(4)
#pl.clf()
#pl.imshow(corono_matrix_im[:,2,:])
#
#pl.figure(5)
#pl.clf()
#pl.semilogy(direct_int_a/direct_int_a.max())
#pl.semilogy(corono_int_a/direct_int_a.max())
#
##%% DZPM tests
#'''
#tests on the DZPM 1d class
#'''
#b = cg.DZPM1d(PupilObs=0.,LyotStopObs=0,nImg=256,nPup=300,Fmax=50)
#Apod_b = np.ones((b.params['nPup']))
#direct_matrix_re, direct_matrix_im = b.prop_direct_matrix()
#corono_matrix_re, corono_matrix_im = b.prop_corono_matrix()
#
#ome1 = -2.340
#ome2 = 2.051
#Apod_b = 1 + ome1*(b.r/2)**2 + ome2*(b.r/2)**4
#
#direct_int_b = b.compute_direct_intensity_1d(Apod_b)
#corono_int_b = b.compute_corono_intensity_1d(Apod_b)
#
##%% plot displays
#pl.figure(11)
#pl.clf()
#pl.imshow(direct_matrix_re[:,2,:])
#
#pl.figure(12)
#pl.clf()
#pl.imshow(direct_matrix_im[:,2,:])
#
#pl.figure(13)
#pl.clf()
#pl.imshow(corono_matrix_re[:,2,:])
#
#pl.figure(14)
#pl.clf()
#pl.imshow(corono_matrix_im[:,2,:])
#
#pl.figure(15)
#pl.clf()
#pl.semilogy(b.xi, direct_int_b/direct_int_b.max())
#pl.semilogy(b.xi, corono_int_b/direct_int_b.max())
#
#pl.show()

#%%
'''
tests on APLC 2d class
'''
a = cg.APLC2d()

Apod2d = a.Pupil2d

poly_direct_intensity_2d = a.compute_direct_intensity_2d(Apod2d)
poly_corono_intensity_2d = a.compute_corono_intensity_2d(Apod2d)

direct_field_t_re, direct_field_t_im = a.prop_direct_matrix_2d()
corono_field_t_re, corono_field_t_im = a.prop_corono_matrix_2d()


#%% plot displays
pl.figure(1)
pl.clf()
pl.imshow(poly_direct_intensity_2d**0.25, cmap = 'inferno')

pl.figure(2)
pl.clf()
pl.imshow(poly_corono_intensity_2d**0.25, cmap = 'inferno')


nImg2d    = a.params['nImg2d']
rMask     = a.params['rMask']
rho0      = a.params['rho0']
rho1      = a.params['rho1']
cDarkHole = a.params['cDarkHole']
# Intensity profiles of the direct and coronagraphic images
pl.figure(3)
pl.clf()
pl.semilogy(a.xi2d,poly_direct_intensity_2d[nImg2d//2,nImg2d//2:]/poly_direct_intensity_2d.max(),label='Direct')
pl.semilogy(a.xi2d,poly_corono_intensity_2d[nImg2d//2,nImg2d//2:]/poly_direct_intensity_2d.max(),label='Corono')
pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=a.xi2d.min(), xmax=a.xi2d.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()

idx = (a.params['nlam']-1)//2

pl.figure(4)
pl.clf()
pl.imshow(direct_field_t_re[:,idx,:], cmap = 'inferno')

pl.figure(5)
pl.clf()
pl.imshow(direct_field_t_im[:,idx,:], cmap = 'inferno')

pl.figure(6)
pl.clf()
pl.imshow(corono_field_t_re[:,idx,:], cmap = 'inferno')

pl.figure(7)
pl.clf()
pl.imshow(corono_field_t_im[:,idx,:], cmap = 'inferno')

pl.show()