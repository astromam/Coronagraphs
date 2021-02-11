#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 23:04:54 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""


#%% Initialization
import numpy as np
import pylab as pl
import corono as coro

#%% APLC2d tests
"""
tests on APLC 2d class
"""
pl.close('all')
corono_name   = 'APLC' # 'SP' or 'APLC' or DZPM
CtrBtwnPix  = False
CtrBtwnPix2 = False
Pupil2dSym  = False
cDarkHole   = 6

# samplings
nPup   = 200
nImg2d = 200
Fmax2d = 50
nFPM   = 100

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0   = 2.
rho1   = 10.

# spectral sampling
nlam   = 1

#%%
"""
File reading for Pupil and Lyot stop
"""
Pupil2d    = coro.utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
LyotStop2d = coro.utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)

params = coro.to_dict(nPup=nPup, nImg2d=nImg2d, Fmax2d = Fmax2d, nFPM = nFPM, 
                      Pupil2dSym = Pupil2dSym, 
                      Pupil2d = Pupil2d, LyotStop2d = LyotStop2d, 
                      CtrBtwnPix=CtrBtwnPix,
                      CtrBtwnPix2 = CtrBtwnPix2, nlam=nlam,
                      rho0   = rho0, rho1 = rho1,
                      cDarkHole = cDarkHole)

#%%
"""
Coronagraph definition
"""
if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
elif corono_name == 'DZPM':
    corono0 = coro.design.DZPM2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
if corono_name == 'DZPM':
    rr = corono0.rr
#    pl.figure(0)
#    pl.clf()
#    pl.imshow(rr)
#    pl.title('test on radial coordinate')
#    pl.show()
    Apod2d = 1 + corono0.params['ome1']*(rr)**2 + corono0.params['ome2']*(rr)**4
    Apod2d *= Pupil2d
else:
    Apod2d = corono0.Pupil2d

poly_direct_intensity_2d = corono0.compute_direct_intensity_2d(Apod2d)
poly_corono_intensity_2d = corono0.compute_corono_intensity_2d(Apod2d)

peaknorm = 1./poly_direct_intensity_2d.max()

poly_direct_intensity_2d *= peaknorm
poly_corono_intensity_2d *= peaknorm

#%%
params2 = coro.update_params(params, Einsum=True)
corono2 = coro.design.APLC2d(**params2)

poly_direct_intensity_2d_bis = corono2.compute_direct_intensity_2d(Apod2d)
poly_corono_intensity_2d_bis = corono2.compute_corono_intensity_2d(Apod2d)

peaknorm_bis = 1./poly_direct_intensity_2d_bis.max()

poly_direct_intensity_2d_bis *= peaknorm_bis
poly_corono_intensity_2d_bis *= peaknorm_bis


#%% pupil displays
pl.figure(1)
pl.clf()
pl.imshow(Pupil2d, cmap = 'Greys_r')
pl.title('Entrance pupil')

pl.figure(2)
pl.clf()
pl.imshow(Apod2d, cmap = 'Greys_r')
pl.title('Apodizer')

pl.figure(3)
pl.clf()
pl.imshow(LyotStop2d, cmap = 'Greys_r')
pl.title('Lyot stop')

#%% image displays
pl.figure(11)
pl.clf()
pl.imshow(np.log10(poly_direct_intensity_2d), cmap = 'inferno',
          vmin=-10, vmax=0)
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')
pl.title('Broadband direct image')

pl.figure(12)
pl.clf()
pl.imshow(np.log10(poly_corono_intensity_2d), cmap = 'inferno',
          vmin=-10, vmax=0)
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')
pl.title('Broadband coronagraphic image')

#%% image displays
pl.figure(13)
pl.clf()
pl.imshow(np.log10(poly_direct_intensity_2d_bis), cmap = 'inferno',
          vmin=-10, vmax=0)
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')
pl.title('Broadband direct image (einsum)')

pl.figure(14)
pl.clf()
pl.imshow(np.log10(poly_corono_intensity_2d_bis), cmap = 'inferno',
          vmin=-10, vmax=0)
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')
pl.title('Broadband coronagraphic image (einsum)')

print('direct image diff: {0}'.format(np.max(np.abs(poly_direct_intensity_2d_bis-poly_direct_intensity_2d))))
print('corono image diff: {0}'.format(np.max(np.abs(poly_corono_intensity_2d_bis-poly_corono_intensity_2d))))

#%% Intensity profiles of the direct and coronagraphic images
nImg2d    = corono0.params['nImg2d']
if corono_name == 'DZPM':
    rMask1     = corono0.params['rMask1']
    rMask2     = corono0.params['rMask2']
else:
    rMask     = corono0.params['rMask']
rho0      = corono0.params['rho0']
rho1      = corono0.params['rho1']

xi2d = corono0.xi2d
if nImg2d%2 == 0:
    xi2d = corono0.xi2d_ctr


pl.figure(3)
pl.clf()
pl.semilogy(xi2d,poly_direct_intensity_2d[nImg2d//2,nImg2d//2:],
            label='Direct')
pl.semilogy(xi2d,poly_corono_intensity_2d[nImg2d//2,nImg2d//2:],
            label='Corono')
if corono_name == 'DZPM':
    pl.axvline(x=rMask1, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=rMask2, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
else:    
    pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi2d.min(), xmax=corono0.xi2d.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()
pl.title('Image radial intensity profile')

pl.show()


print('coronagraphic intensity peak: {0}'.format(np.max(poly_corono_intensity_2d/poly_direct_intensity_2d.max())))
print('coronagraphic intensity peak: {0} (einsum)'.format(np.max(poly_corono_intensity_2d_bis/poly_direct_intensity_2d_bis.max())))
