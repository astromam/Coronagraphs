#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 23:07:48 2018

@author: mndiaye
"""

#%% Initialization
import numpy as np
import pylab as pl
from corono import corono_design as cd
from corono.utils import to_dict, uniform_disk, radius_disk

#%% APLC2d tests
"""
tests on APLC 2d class
"""
corono_name   = 'DZPM' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = False
CtrBtwnPix2 = False
SymPupil2d  = False
cDarkHole   = 6

rho0   = 2.
nPup   = 300
nImg2d = 512
Fmax2d = 100
nlam   = 5
nFPM   = 200


Pupil2d = uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
pl.figure(1)
pl.imshow(Pupil2d)


LyotStop2d = uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
pl.figure(1)
pl.imshow(Pupil2d)

#%%

params = to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM, 
                 SymPupil2d = SymPupil2d, 
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                 CtrBtwnPix=CtrBtwnPix,
                 CtrBtwnPix2 = CtrBtwnPix2, nlam=nlam,
                 rho0   = rho0, cDarkHole = cDarkHole)

if corono_name == 'SP':
    a = cd.SP2d(**params)
elif corono_name == 'APLC':
    a = cd.APLC2d(**params)
elif corono_name == 'DZPM':
    a = cd.DZPM2d(**params)
else:
    print('Check the name of the coronagraph')
    stop

if corono_name == 'DZPM':
    rr = a.rr
    pl.figure(20)
    pl.imshow(rr)
    pl.show()
    Apod2d = 1 + a.params['ome1']*(rr)**2 + a.params['ome2']*(rr)**4 
else:
    Apod2d = a.Pupil2d

poly_direct_intensity_2d = a.compute_direct_intensity_2d(Apod2d)
poly_corono_intensity_2d = a.compute_corono_intensity_2d(Apod2d)

#poly_direct_intensity_2d[nImg2d//2:,nImg2d//2:]=0
#poly_direct_intensity_2d[nImg2d//2:,0:nImg2d//2:]=0
#poly_direct_intensity_2d[0:nImg2d//2:,nImg2d//2:,]=0

#direct_field_t_re, direct_field_t_im = a.prop_direct_matrix_2d()
#corono_field_t_re, corono_field_t_im = a.prop_corono_matrix_2d()



#%% plot displays
pl.figure(11)
pl.clf()
pl.imshow(poly_direct_intensity_2d**0.25, cmap = 'inferno')

pl.figure(12)
pl.clf()
pl.imshow(poly_corono_intensity_2d**0.25, cmap = 'inferno')


nImg2d    = a.params['nImg2d']
if corono_name == 'DZPM':
    rMask1     = a.params['rMask1']
    rMask2     = a.params['rMask2']
else:
    rMask     = a.params['rMask']
rho0      = a.params['rho0']
rho1      = a.params['rho1']

# Intensity profiles of the direct and coronagraphic images
pl.figure(3)
pl.clf()
pl.semilogy(a.xi2d,poly_direct_intensity_2d[nImg2d//2,nImg2d//2:]/poly_direct_intensity_2d.max(),label='Direct')
pl.semilogy(a.xi2d,poly_corono_intensity_2d[nImg2d//2,nImg2d//2:]/poly_direct_intensity_2d.max(),label='Corono')
if corono_name == 'DZPM':
    pl.axvline(x=rMask1, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=rMask2, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
else:    
    pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=a.xi2d.min(), xmax=a.xi2d.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()
pl.title('Image radial intensity profile')

pl.show()


print('\n coronagraphic intensity peak: {0}'.format(np.max(poly_corono_intensity_2d/poly_direct_intensity_2d.max())))
#%%
#idx = (a.params['nlam']-1)//2

#pl.figure(13)
#pl.clf()
#pl.imshow(direct_field_t_re[:,idx,:], cmap = 'inferno')
#pl.title('direct response matrix for APLC 2D - real part')
#
#pl.figure(14)
#pl.clf()
#pl.imshow(direct_field_t_im[:,idx,:], cmap = 'inferno')
#pl.title('direct response matrix for APLC 2D - imaginary part')
#
#pl.figure(15)
#pl.clf()
#pl.imshow(corono_field_t_re[:,idx,:], cmap = 'inferno')
#pl.title('coronagraphic response matrix for APLC 2D - real part')
#
#pl.figure(16)
#pl.clf()
#pl.imshow(corono_field_t_im[:,idx,:], cmap = 'inferno')
#pl.title('coronagraphic response matrix for APLC 2D - imaginary part')
