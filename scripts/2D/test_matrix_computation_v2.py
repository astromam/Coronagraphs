#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:46:39 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%%
"""
Initialisation
"""
import numpy as np
import pylab as pl
import time

import corono as coro
from corono import utils

#%%
"""
Parameters
"""
nPup = 100

rho0 = 5.0
rho1 = 10.0

nFPM  = 10
rMask = 2.5
dMask = 2*rMask

Fmax2d = 22.5
nImg2d = 45

CtrBtwnPix = True

#%%
# Pupil
Pupil2d = utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
Pupil1d = Pupil2d.ravel()

pup     = (Pupil1d > 0.)
bbb     = np.arange(nPup**2)
idx_pup = list(bbb[pup])
npp     = len(idx_pup)

#%%
Mask2d = utils.uniform_disk(nFPM, nFPM/2., CtrBtwnPix=CtrBtwnPix)
Mask1d = Mask2d.ravel()
M = Mask1d*1

msk     = (Mask1d > 0.)
ccc     = np.arange(nFPM**2)
idx_msk = list(ccc[msk])
nmm     = len(idx_msk)

#%%
LyotStop2d = utils.uniform_disk(nPup, nPup/4., CtrBtwnPix=CtrBtwnPix)
LyotStop1d = LyotStop2d.ravel()
L = LyotStop1d[idx_pup]*1

#%%
"""
Computation
"""

if CtrBtwnPix is True:
    val = 1/2 
x2d,y2d  = np.meshgrid(np.arange(nPup)-nPup/2+val, np.arange(nPup)-nPup/2+val)
x2d /= nPup
y2d /= nPup 


#%%
if CtrBtwnPix is True:
    val = 1/2   
p2d,q2d  = np.meshgrid(np.arange(nFPM)-nFPM/2+val, np.arange(nFPM)-nFPM/2+val)
p2d *= dMask/nFPM
q2d *= dMask/nFPM 

#%%
if CtrBtwnPix is True:
    val = 1/2   
u2d,v2d  = np.meshgrid(np.arange(nImg2d)-nImg2d/2+val, np.arange(nImg2d)-nImg2d/2+val)
u2d *= Fmax2d/nImg2d
v2d *= Fmax2d/nImg2d 

mydist = np.hypot(v2d, u2d)
res    = (mydist <= rho1)*(mydist >= rho0)
dz     = res.ravel()
aaa    = np.arange(nImg2d**2)
idx_dz = list(aaa[dz])  
ndz    = len(idx_dz)

Annulus1d = np.zeros((nImg2d**2))
Annulus1d[dz] = 1.
Annulus2d = np.reshape(Annulus1d, (nImg2d, nImg2d))


#%%
x1d_tmp = x2d.ravel()
y1d_tmp = y2d.ravel()

p1d_tmp = p2d.ravel()
q1d_tmp = q2d.ravel()

u1d_tmp = u2d.ravel()
v1d_tmp = v2d.ravel()

x1d = x1d_tmp[idx_pup]
y1d = y1d_tmp[idx_pup]

p1d = p1d_tmp[idx_msk]
q1d = q1d_tmp[idx_msk]

u1d = u1d_tmp[idx_dz]
v1d = v1d_tmp[idx_dz]


#%%
FAB = np.exp(-2j*np.pi*(x1d[:, None]*p1d[None, :] + y1d[:, None]*q1d[None, :]))
FAB *= dMask/(nPup*nFPM)

print('FAB: {0}'.format(FAB.shape))

#%%
FBC = np.exp(2j*np.pi*(p1d[:, None]*x1d[None, :] + q1d[:, None]*y1d[None, :]))
FBC *= dMask/(nPup*nFPM)

print('FBC: {0}'.format(FBC.shape))

#%%
FCD = np.exp(-2j*np.pi*(x1d[:, None]*u1d[None, :] + y1d[:, None]*v1d[None, :]))
#FCD = np.cos(-2*np.pi*(x1d[:, None]*u1d[None, :] + y1d[:, None]*v1d[None, :]))+1j*np.sin(-2*np.pi*(x1d[:, None]*u1d[None, :] + y1d[:, None]*v1d[None, :]))

FCD *= Fmax2d/(nPup*nImg2d)
print('FCD: {0}'.format(FCD.shape))

#%%
#LFCD = FCD*L[:,None]
#MFBC = FBC*M[:,None]
LFCD = FCD*L[:,None]
MFBC = FBC*M[idx_msk,None]
print('LFCD: {0}'.format(LFCD.shape))
print('MFBC: {0}'.format(MFBC.shape))

#%%
Q1 = LFCD
print('Q1: {0}'.format(Q1.shape))

#Q2 = np.einsum('ik,kl,lj-> ij', FAB, MFBC, LFCD)
t0 = time.time()
Q2 = (FAB.dot(MFBC)).dot(LFCD)
t1 = time.time()
print('Q2 computation time: {0:.2f}s'.format(t1-t0))
print('Q2: {0}'.format(Q2.shape))

#%%

Q = Q1 - Q2
print('Q: {0}'.format(Q.shape))

#%%
"""
test 1 
"""
#Field1d = Pupil1d.dot(Q1)
#Field2d = np.reshape(Field1d, (nImg2d, nImg2d))
#Image2d = np.abs(Field2d)**2

Field1d_tmp = Pupil1d[idx_pup].dot(Q1)

Field1d = np.zeros((nImg2d**2), dtype='complex128')
Field1d[idx_dz] = Field1d_tmp
Field2d = np.reshape(Field1d, (nImg2d, nImg2d))

Image2d = np.abs(Field2d)**2

Field2d0 = utils.sft(Pupil2d*LyotStop2d, nImg2d, Fmax2d, CtrBtwnPix=CtrBtwnPix)
Image2d0 = np.abs(Field2d0)**2*Annulus2d

#%%
"""
Plot display
"""
pl.figure(0)
pl.imshow(Pupil2d)
pl.title('Pupil 2d')
pl.show()

#%%
pl.figure(1)
pl.imshow(Image2d**0.25)
pl.title('Image 2d')
pl.show()

#%%
pl.figure(2)
pl.imshow(Image2d0**0.25)
pl.title('Image 2d 0')
pl.show()

print(np.max(abs(Image2d - Image2d0)))

#%%
"""
Test 2
"""
#Field1dbis = Pupil1d.dot(Q)
#Field2dbis = np.reshape(Field1dbis, (nImg2d, nImg2d))
#Image2dbis = np.abs(Field2dbis)**2

Field1dbis_tmp = Pupil1d[idx_pup].dot(Q)

Field1dbis = np.zeros((nImg2d**2), dtype='complex128')
Field1dbis[idx_dz] = Field1dbis_tmp
Field2dbis = np.reshape(Field1dbis, (nImg2d, nImg2d))
Image2dbis = np.abs(Field2dbis)**2

field_B       = Mask2d*utils.sft(Pupil2d, nFPM, dMask, 
                                CtrBtwnPix=CtrBtwnPix)
field_C       = Pupil2d - utils.isft(field_B, nPup, dMask, 
                               CtrBtwnPix=CtrBtwnPix)
field_L       = field_C*LyotStop2d
field_D = utils.sft(field_L, nImg2d, Fmax2d, 
          CtrBtwnPix=CtrBtwnPix)

Image2dbis0 = np.abs(field_D)**2*Annulus2d

#%%
"""
Plot display
"""
pl.figure(3)
pl.imshow(Image2dbis)
pl.title('Image corono 2d')
pl.show()

pl.figure(4)
pl.imshow(Image2dbis0)
pl.title('Image corono 2d')
pl.show()

print(np.max(abs(Image2dbis - Image2dbis0)))