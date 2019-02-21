#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:28:09 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%%
"""
Initialisation
"""
from memory_profiler import profile
import numpy as np
import pylab as pl
import time
from pathlib import Path

from astropy.io import fits

from corono import utils

#%%
"""
Parameters
"""
pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'

nPup  = 20
nFPM  = 20
Fmax2d = 22.5
nImg2d = 45

rMask = 2.8
dMask = 2*rMask

rho0 = 5.0
rho1 = 10.0

bw   = 0.1
lam0 = 1
nlam = 5

CtrBtwnPix = True

ImPart = False

#%%
fdir = Path('../../data/2D/pupils/').resolve()
fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
fpath_pup = fdir / fname_pup
fpath_lys = fdir / fname_lys

#%%
# Pupil
#Pupil2d = utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)
Pupil2d    = fits.getdata(fpath_pup)
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
#LyotStop2d = utils.uniform_disk(nPup, nPup/4., CtrBtwnPix=CtrBtwnPix)
LyotStop2d = fits.getdata(fpath_lys)
LyotStop1d = LyotStop2d.ravel()
L = LyotStop1d[idx_pup]*1

#%%
"""
Pre-Computation
"""
dlam       = bw*lam0
lam_t      = np.linspace(lam0-dlam/2*(nlam>1), lam0+dlam/2,nlam)


#%%
if CtrBtwnPix is True:
    val = 1/2 
x2d,y2d = np.meshgrid(np.arange(nPup)-nPup/2+val, np.arange(nPup)-nPup/2+val)
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
"""
Matrix computation function
"""

#%%

def Q_direct(lam):   
    if ImPart is True:
        LFCD = np.exp(-2j*np.pi*(lam0/lam)*(x1d[:, None]*u1d[None, :] + y1d[:, None]*v1d[None, :]))
    else:
        LFCD = np.cos(-2*np.pi*(lam0/lam)*(x1d[:, None]*u1d[None, :] + y1d[:, None]*v1d[None, :]))
    
    LFCD *= (lam0/lam)*Fmax2d/(nPup*nImg2d)
    LFCD *= L[:,None]
    return LFCD

#%%
#@profile
def Q_corono(lam):
    if ImPart is True:
        FAB = np.exp(-2j*np.pi*(lam0/lam)*(x1d[:, None]*p1d[None, :] + y1d[:, None]*q1d[None, :]))
    else:
        FAB = np.cos(-2*np.pi*(lam0/lam)*(x1d[:, None]*p1d[None, :] + y1d[:, None]*q1d[None, :]))
    
    FAB *= (lam0/lam)*dMask/(nPup*nFPM)
    
    if ImPart is True:
        MFBC = FAB.conjugate().T    
    else:
        MFBC = FAB.T
    MFBC *= M[idx_msk,None]
    
    LFCD = Q_direct(lam)
    
    Q  = - (FAB.dot(MFBC)).dot(LFCD)
    Q += LFCD
    
    return Q
#%%
"""
Matrix computation
"""
t00 = time.time()
Q0 = np.empty((npp, nlam, ndz))
for i, lam in enumerate(lam_t):
    Q0[:, i] = Q_direct(lam)
    
print('Q0: {0}'.format(Q0.shape))
t11 = time.time()
print('direct computation time: {0:.5f}s'.format(t11-t00))


t00 = time.time()
Q = np.empty((npp, nlam, ndz))
for i, lam in enumerate(lam_t):
    Q[:, i] = Q_corono(lam)
    
print('Q: {0}'.format(Q.shape))
t11 = time.time()
print('corono computation time: {0:.5f}s'.format(t11-t00))


#%%
"""
Direct image - test 
"""
t0 = time.time()
Field1d_tmp = np.tensordot(Pupil1d[idx_pup], Q0, (0, 0))

if ImPart is True:
    Field1d = np.zeros((nlam, nImg2d**2), dtype='complex128')
else:
    Field1d = np.zeros((nlam, nImg2d**2))

Field1d[:, idx_dz] = Field1d_tmp
Field2d = np.reshape(Field1d, (nlam, nImg2d, nImg2d))
Image2d = np.abs(Field2d)**2
Image2d = np.sum(Image2d, axis=0)

t1 = time.time() 
print('direct, new    : {0:.6f}s'.format(t1-t0))

#%%
t0 = time.time()

Image2d0 = np.zeros((nImg2d, nImg2d))
for i, lam in enumerate(lam_t):
    if ImPart is True:
        Field2d0 = utils.sft(Pupil2d*LyotStop2d, nImg2d, (lam0/lam)*Fmax2d, CtrBtwnPix=CtrBtwnPix)
    else:
        Field2d0 = utils.sft_even(Pupil2d*LyotStop2d, nImg2d, (lam0/lam)*Fmax2d, CtrBtwnPix=CtrBtwnPix)    
    Image2d0 += np.abs(Field2d0)**2*Annulus2d

t1 = time.time() 
print('direct, classic: {0:.6f}s'.format(t1-t0))


#%%
"""
Coronagraphic image - test
"""
t0 = time.time()
Field1dbis_tmp = np.tensordot(Pupil1d[idx_pup], Q, (0, 0))

if ImPart is True:
    Field1dbis = np.zeros((nlam, nImg2d**2), dtype='complex128')
else:
    Field1dbis = np.zeros((nlam, nImg2d**2))
    
Field1dbis[:, idx_dz] = Field1dbis_tmp
Field2dbis = np.reshape(Field1dbis, (nlam, nImg2d, nImg2d))
Image2dbis = np.abs(Field2dbis)**2
Image2dbis = np.sum(Image2dbis, axis=0)

t1 = time.time() 
print('corono, new    : {0:.6f}s'.format(t1-t0))

#%%
t0 = time.time()

Image2dbis0 = np.zeros((nImg2d, nImg2d))
for i, lam in enumerate(lam_t):
    if ImPart is True: 
        field_B       = Mask2d*utils.sft(Pupil2d, nFPM, (lam0/lam)*dMask, 
                                        CtrBtwnPix=CtrBtwnPix)
        field_C       = Pupil2d - utils.isft(field_B, nPup, (lam0/lam)*dMask, 
                                       CtrBtwnPix=CtrBtwnPix)
        field_L       = field_C*LyotStop2d
        field_D       = utils.sft(field_L, nImg2d, Fmax2d, 
                  CtrBtwnPix=CtrBtwnPix)
    else:
        field_B       = Mask2d*utils.sft_even(Pupil2d, nFPM, (lam0/lam)*dMask, 
                                        CtrBtwnPix=CtrBtwnPix)
        field_C       = Pupil2d - utils.isft_even(field_B, nPup, (lam0/lam)*dMask, 
                                       CtrBtwnPix=CtrBtwnPix)
        field_L       = field_C*LyotStop2d
        field_D       = utils.sft_even(field_L, nImg2d, (lam0/lam)*Fmax2d, 
                  CtrBtwnPix=CtrBtwnPix)    

    Image2dbis0 += np.abs(field_D)**2*Annulus2d
    
t1 = time.time() 
print('corono, classic: {0:.6f}s'.format(t1-t0))

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
pl.title('Image 2d (new)')
pl.show()

#%%
pl.figure(2)
pl.imshow(Image2d0**0.25)
pl.title('Image 2d (classic)')
pl.show()

print('direct image max diff.: {0}'.format(np.max(abs(Image2d - Image2d0))))

#%%
"""
Plot display
"""
pl.figure(3)
pl.imshow(Image2dbis)
pl.title('Image corono 2d (new)')
pl.show()

#%%
pl.figure(4)
pl.imshow(Image2dbis0)
pl.title('Image corono 2d (classic)')
pl.show()

print('corona image max diff.: {0}'.format(np.max(abs(Image2dbis - Image2dbis0))))