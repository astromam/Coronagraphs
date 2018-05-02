#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:57:03 2018

@author: mndiaye
"""

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

#%% APLC1d tests   
"""
#tests on the APLC 1d class
"""
#a = cd.APLC1d()
#a = cd.SP1d()
#Apod = np.ones((a.params['nPup']))
#direct_matrix_re, direct_matrix_im = a.prop_direct_matrix_1d()
#corono_matrix_re, corono_matrix_im = a.prop_corono_matrix_1d()
#
#direct_int_a = a.compute_direct_intensity_1d(Apod)
#corono_int_a = a.compute_corono_intensity_1d(Apod)
#
#%% plot displays
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
#pl.semilogy(a.xi, direct_int_a/direct_int_a.max())
#pl.semilogy(a.xi, corono_int_a/direct_int_a.max())
#pl.title('Intensity profiles for the APLC in 1D')
#pl.xlabel(r'Angular separation in $\lambda_0$/D')
#pl.ylabel('Normalized intensity in log scale')
#


#%% DZPM1d tests
"""
#tests on the DZPM 1d class
"""
PupilObs    = 0.
LyotStopObs = 0
nImg        = 256
nPup        = 300
Fmax        = 50
rho0        = 2
rho1        = 10
cDarkHole   = 6

params = to_dict(PupilObs = PupilObs, LyotStopObs = LyotStopObs,
                 nImg = nImg, nPup= nPup, Fmax = Fmax,
                 rho0 = rho0, rho1 = rho1)

b = cd.DZPM1d(**params)
Apod_b = np.ones((nPup))
direct_matrix_re, direct_matrix_im = b.prop_direct_matrix_1d()
corono_matrix_re, corono_matrix_im = b.prop_corono_matrix_1d()

ome1 = -2.340
ome2 = 2.051
Apod_b = 1 + ome1*(b.r/2)**2 + ome2*(b.r/2)**4

direct_int_b = b.compute_direct_intensity_1d(Apod_b)
corono_int_b = b.compute_corono_intensity_1d(Apod_b)

#%% plot displays
pl.figure(6)
pl.clf()
pl.imshow(direct_matrix_re[:,2,:])
pl.title(r'direct response matrix for DZPM at $\lambda_0$ - real part')

pl.figure(7)
pl.clf()
pl.imshow(direct_matrix_im[:,2,:])
pl.title(r'direct response matrix for DZPM at $\lambda_0$ - imaginary part')

pl.figure(8)
pl.clf()
pl.imshow(corono_matrix_re[:,2,:])
pl.title(r'coronagraphic response matrix for DZPM at $\lambda_0$ - real part')

pl.figure(9)
pl.clf()
pl.imshow(corono_matrix_im[:,2,:])
pl.title(r'coronagraphic response matrix for DZPM at $\lambda_0$ - imaginary part')

pl.figure(10)
pl.clf()
pl.semilogy(b.xi, direct_int_b/direct_int_b.max(), label= 'Direct')
pl.semilogy(b.xi, corono_int_b/direct_int_b.max(), label= 'Corono')
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=b.params['rMask1'], ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=b.params['rMask2'], ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=b.xi.min(), xmax=b.xi.max(), linewidth=1, color='k', linestyle='--')
pl.title('Intensity profiles for the DZPM in 1D')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.legend()

pl.show()

print('\n coronagraphic intensity peak: {0}'.format(np.max(corono_int_b/direct_int_b.max())))
