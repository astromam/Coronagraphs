#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:57:03 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

#%% Initialization
import numpy as np
import pylab as pl
import corono as coro

pl.close('all')

#%%
"""
Parameters
"""
PupilID   = 0.
LyotStopOD = 1.
LyotStopID = 0

nImg        = 256
nPup        = 300
Fmax        = 50

corono_name = 'APLC'

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0        = 2
rho1        = 10


cDarkHole   = 6

# Pupil radial coordinate        
r   = np.arange(nPup)/nPup +1./(2*nPup)

# Telescope aperture
Pupil1d      = (r>PupilID)*1.0

# Lyot stop 
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

params = coro.to_dict(PupilID = PupilID, LyotStopID = LyotStopID,
                      nPup = nPup, r = r,
                      Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                      nImg = nImg, Fmax = Fmax,
                      rho0 = rho0, rho1 = rho1)

#%%
"""
Coronagraph definition
"""
Apod = np.ones(nPup)
if corono_name == 'SP':
    corono0 = coro.design.SP1d(**params)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC1d(**params)
elif corono_name == 'DZPM':
    ome1 = -2.340
    ome2 = 2.051
    Apod = 1 + ome1*(r/2)**2 + ome2*(r/2)**4    
    corono0 = coro.design.DZPM1d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%% Computation of the image intensity
poly_direct_intensity_1d = corono0.compute_direct_intensity_1d(Apod)
poly_corono_intensity_1d = corono0.compute_corono_intensity_1d(Apod)

normpeak = 1./poly_direct_intensity_1d.max()

poly_direct_intensity_1d *= normpeak
poly_corono_intensity_1d *= normpeak

#%% plot displays
pl.figure(10)
pl.clf()
pl.semilogy(corono0.xi, poly_direct_intensity_1d, label= 'Direct')
pl.semilogy(corono0.xi, poly_corono_intensity_1d, label= 'Corono')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#pl.axvline(x=corono0.params['rMask1'], ymin=-12, ymax =2, linewidth=1, 
#           color='r', linestyle='--')
#pl.axvline(x=corono0.params['rMask2'], ymin=-12, ymax =2, linewidth=1, 
#           color='r', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), 
           linewidth=1, color='k', linestyle='--')
pl.title('Image radial intensity profiles for the DZPM in 1D')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()

pl.show()

print('\n coronagraphic intensity peak: {0}'.format(poly_corono_intensity_1d.max()))
