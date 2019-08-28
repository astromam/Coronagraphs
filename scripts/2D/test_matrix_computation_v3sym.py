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
#from memory_profiler import profile
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

nPup  = 50
nFPM  = 50
Fmax2d = 23.#22.5
nImg2d = 46 #45

rMask = 2.8
dMask = 2*rMask

rho0 = 5.0
rho1 = 10.0

bw   = 0.1
lam0 = 1
nlam = 1

Pupil2dSym = True # If Pupil2dSym is True, set ImPart to False
ImPart     = False

CtrBtwnPix = True

#%%
if ImPart is True:
    dtype0 = 'complex128'
else:
    dtype0 = 'float64'

#%%
fdir = Path('../../data/2D/pupils/').resolve()
fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
fpath_pup = fdir / fname_pup
fpath_lys = fdir / fname_lys


#%%
def build_from_quarter(array_quarter):
    n = 2*array_quarter.shape[0]
    array = np.empty((n,n))
    
    array[n//2:, n//2:] = array_quarter
    array[:n//2, n//2:] = np.flip(array_quarter, axis=0)
    array[:, :n//2]     = np.flip(array[:, n//2:], axis=1)
    return array

#%%

NA    = nPup*1
NAbis = nPup/2 
if Pupil2dSym is True:
   NA    = nPup//2
   NAbis = 0

NB    = nFPM*1
NBbis = nFPM/2
if Pupil2dSym is True:
    NB    = nFPM//2
    NBbis = 0

ND    = nImg2d*1
NDbis = nImg2d/2
if Pupil2dSym is True:
    ND    = nImg2d//2
    NDbis = 0    

#%%
# Pupil
Pupil2d = fits.getdata(fpath_pup)

if Pupil2dSym is True:
    Pupil2dquarter = Pupil2d[nPup//2:, nPup//2:]
    Pupil1d = Pupil2dquarter.ravel()
else:
    Pupil1d = Pupil2d.ravel()

pup     = (Pupil1d > 0.)
bbb     = np.arange(NA**2)
idx_pup = list(bbb[pup])
npp     = len(idx_pup)

#%%
Mask2d = utils.uniform_disk(nFPM, nFPM/2., CtrBtwnPix=CtrBtwnPix)

if Pupil2dSym is True:
    Mask2dquarter = Mask2d[nFPM//2:, nFPM//2:]
    Mask1d = Mask2dquarter.ravel()    
else:    
    Mask1d = Mask2d.ravel()

msk     = (Mask1d > 0.)
ccc     = np.arange(NB**2)
idx_msk = list(ccc[msk])
nmm     = len(idx_msk)

M       = Mask1d[idx_msk]


#%%
#LyotStop2d = utils.uniform_disk(nPup, nPup/4., CtrBtwnPix=CtrBtwnPix)
LyotStop2d = fits.getdata(fpath_lys)

if Pupil2dSym is True:
    LyotStop2dquarter = LyotStop2d[nPup//2:, nPup//2:]
    LyotStop1d = LyotStop2dquarter.ravel()
else:
    LyotStop1d = LyotStop2d.ravel()

L       = LyotStop1d[idx_pup]


#%%
"""
Pre-Computation
"""
dlam       = bw*lam0
lam_t      = np.linspace(lam0-dlam/2*(nlam>1), lam0+dlam/2,nlam)

if CtrBtwnPix is True:
    val = 1/2

#%%   
x2d,y2d = np.meshgrid(np.arange(NA)-NAbis+val, np.arange(NA)-NAbis+val)
x2d /= nPup
y2d /= nPup 

#%%
p2d,q2d = np.meshgrid(np.arange(NB)-NBbis+val, np.arange(NB)-NBbis+val)
p2d *= dMask/nFPM
q2d *= dMask/nFPM 

#%%
u2d,v2d  = np.meshgrid(np.arange(ND)-NDbis+val, np.arange(ND)-NDbis+val)
u2d *= Fmax2d/nImg2d
v2d *= Fmax2d/nImg2d 

#%%
mydist = np.hypot(v2d, u2d)
res    = (mydist <= rho1)*(mydist >= rho0)
dz     = res.ravel()
aaa    = np.arange(ND**2)
idx_dz = list(aaa[dz])  
ndz    = len(idx_dz)

Annulus1d = np.zeros((ND**2))
Annulus1d[dz] = 1.

if Pupil2dSym is True: 
    Annulus2dtmp = np.reshape(Annulus1d, (ND, ND))
    Annulus2d = build_from_quarter(Annulus2dtmp)
else:
    Annulus2d = np.reshape(Annulus1d, (ND, ND))

#Annulus2d = np.reshape(Annulus1d, (ND, ND))

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
        MFBC = FAB.T
    else:
        MFBC = FAB.T
        
    MFBC *= M[:,None]
    
    LFCD = Q_direct(lam)
    
    Q  = - (FAB.dot(MFBC)).dot(LFCD)
    Q += LFCD
    
    return Q

#%%
"""
Matrix computation
"""
t00 = time.time()
Q0 = np.empty((npp, nlam, ndz), dtype=dtype0)
for i, lam in enumerate(lam_t):
    Q0[:, i] = Q_direct(lam)
    
print('Q0: {0}'.format(Q0.shape))
t11 = time.time()
print('direct computation time: {0:.5f}s'.format(t11-t00))


#t00 = time.time()
#Q = np.empty((npp, nlam, ndz), dtype=dtype0)
#for i, lam in enumerate(lam_t):
#    Q[:, i] = Q_corono(lam)
#    
#print('Q: {0}'.format(Q.shape))
#t11 = time.time()
#print('corono computation time: {0:.5f}s'.format(t11-t00))


#%%
"""
Direct image - test 
"""
t0 = time.time()
Field1d0_tmp = np.tensordot(Pupil1d[idx_pup], Q0, (0, 0))

Field1d0 = np.zeros((nlam, ND**2), dtype=dtype0)

Field1d0[:, idx_dz] = Field1d0_tmp
Field2d0 = np.reshape(Field1d0, (nlam, ND, ND))
Image2d0tmp = np.abs(Field2d0)**2
Image2d0tmp = np.sum(Image2d0tmp, axis=0)

if Pupil2dSym is True:
    Image2d0 = build_from_quarter(Image2d0tmp)
else:
    Image2d0 = Image2d0tmp

t1 = time.time() 
print('direct, new    : {0:.6f}s'.format(t1-t0))

#%%
t0 = time.time()

Image2d0_bis = np.zeros((nImg2d, nImg2d))
for i, lam in enumerate(lam_t):
    if ImPart is True:
        Field2d0_bis = utils.sft(Pupil2d*LyotStop2d, nImg2d, (lam0/lam)*Fmax2d, CtrBtwnPix=CtrBtwnPix)
    else:
        Field2d0_bis = utils.sft_even(Pupil2d*LyotStop2d, nImg2d, (lam0/lam)*Fmax2d, CtrBtwnPix=CtrBtwnPix)    
    Image2d0_bis += np.abs(Field2d0_bis)**2*Annulus2d

t1 = time.time() 
print('direct, classic: {0:.6f}s'.format(t1-t0))

#%%
#"""
#Coronagraphic image - test
#"""
#t0 = time.time()
#Field1d_tmp = np.tensordot(Pupil1d[idx_pup], Q, (0, 0))
#
#Field1d = np.zeros((nlam, ND**2), dtype=dtype0)
#    
#Field1d[:, idx_dz] = Field1d_tmp
#Field2d = np.reshape(Field1d, (nlam, ND, ND))
#Image2dtmp = np.abs(Field2d)**2
#Image2dtmp = np.sum(Image2dtmp, axis=0)
#
#if Pupil2dSym is True:
#    Image2d = build_from_quarter(Image2dtmp)
#else:
#    Image2d = Image2dtmp
#
#t1 = time.time() 
#print('corono, new    : {0:.6f}s'.format(t1-t0))
#
##%%
#t0 = time.time()
#
#Image2d_bis = np.zeros((nImg2d, nImg2d))
#for i, lam in enumerate(lam_t):
#    if ImPart is True: 
#        field_B       = Mask2d*utils.sft(Pupil2d, nFPM, (lam0/lam)*dMask, 
#                                        CtrBtwnPix=CtrBtwnPix)
#        field_C       = Pupil2d - utils.isft(field_B, nPup, (lam0/lam)*dMask, 
#                                       CtrBtwnPix=CtrBtwnPix)
#        field_L       = field_C*LyotStop2d
#        field_D       = utils.sft(field_L, nImg2d, (lam0/lam)*Fmax2d, 
#                  CtrBtwnPix=CtrBtwnPix)
#    else:
#        field_B       = Mask2d*utils.sft_even(Pupil2d, nFPM, (lam0/lam)*dMask, 
#                                        CtrBtwnPix=CtrBtwnPix)
#        field_C       = Pupil2d - utils.isft_even(field_B, nPup, (lam0/lam)*dMask, 
#                                       CtrBtwnPix=CtrBtwnPix)
#        field_L       = field_C*LyotStop2d
#        field_D       = utils.sft_even(field_L, nImg2d, (lam0/lam)*Fmax2d, 
#                  CtrBtwnPix=CtrBtwnPix)    
#
#    Image2d_bis += np.abs(field_D)**2*Annulus2d
#    
#t1 = time.time() 
#print('corono, classic: {0:.6f}s'.format(t1-t0))

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
pl.imshow(Image2d0**0.25)
pl.title('Image 2d (new)')
pl.show()

#%%
pl.figure(2)
pl.imshow(Image2d0_bis**0.25)
pl.title('Image 2d (classic)')
pl.show()

#%%
pl.figure(3)
pl.imshow(abs(Image2d0-Image2d0_bis))
pl.title('Image 2d diff')
pl.show()

print('direct image max diff.: {0}'.format(np.max(abs(Image2d0 - Image2d0_bis))))

#%%
#"""
#Plot display
#"""
#pl.figure(11)
#pl.imshow(Image2d)
#pl.title('Image corono 2d (new)')
#pl.show()
#
##%%
#pl.figure(12)
#pl.imshow(Image2d_bis)
#pl.title('Image corono 2d (classic)')
#pl.show()
#
##%%
#pl.figure(13)
#pl.imshow(abs(Image2d-Image2d_bis))
#pl.title('Image corono 2d diff')
#pl.show()
#
#
#print('corona image max diff.: {0}'.format(np.max(abs(Image2d - Image2d_bis))))