#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:20:41 2019

@author: mndiaye
"""

import numpy as np
import time
import os

from pathlib import Path
import corono as coro

from corono.utils import besselJ0

import pylab as pl

#%% parameters
"""
Parameters
"""
corono_name  = 'APLC' # 'APLC' or 'SP'
problem_name = 'MaxContrastL1' # ,'MaxTau' # 'MaxContrastLinf' #'MaxContrastL1'
solver       = 'gurobipy' # 'stdgrb', 'gurobipy', 'scipy.linprog'
slvLogToConsole = 0
slvCrossover    = 0
slvMethod       = 2
allLogToConsole = 0

FirstDer    = False
SecondDer   = False
MinIsland   = False
FirstDerLim = 0.01
SecondDerLim= 0.001 
FirstDerGlobalLim = 10.

CtrBtwnPix  = False
CtrBtwnPix2 = False

nPup = 500
nFPM = 50
nImg = 220
Fmax = 11
R    = 1
nImg2d = int(2*nImg)
Fmax2d = int(2*Fmax)

bw   = 0.1
nlam = 11

PupilID    = 0.0
rMask       = 4.2

nFPM2 = int(rMask*nFPM*(1+ bw/2))+2

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75

LyotStopID = 0.0
LyotStopOD = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 3.5
rho1 = 10.0

# contrast in the dark region
cDarkHole = 8.0

# tau (integrated Pupil transmission)
tau   = 0.5

r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilID)*1.0
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

Pupil2d    = coro.utils.uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix) -\
coro.utils.uniform_disk(nPup, PupilID*nPup/2., CtrBtwnPix=CtrBtwnPix)

LyotStop2d = coro.utils.uniform_disk(nPup, LyotStopOD*nPup/2., CtrBtwnPix=CtrBtwnPix) -\
coro.utils.uniform_disk(nPup, LyotStopID*nPup/2., CtrBtwnPix=CtrBtwnPix) 


params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM2, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, problem_name = problem_name,
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 nImg2d = nImg2d, Fmax2d = Fmax2d,
                 CtrBtwnPix = CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d)

#%%
"""
Working directory
"""
fdir = Path('../../results/1D/dat_pyth').resolve()
if not os.path.exists(fdir):
    os.makedirs(fdir)      

#%%
"""
Sampling
"""    
lam0 = 1
# wavelengths        
dlam       = bw*lam0
lam_t      = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam)

# mask size at Apod given wavelength
rMask_t    = (lam0/lam_t)*rMask

# mask sampling at Apod given wavelength and max nFPM_max 
nFPM_t     = rMask_t*nFPM
nFPM_max   = int(np.max(nFPM_t))
mask_lam   = (np.arange(nFPM_max+1)[None,:] < nFPM_t[:,None])
xi_FPM_lam = np.arange(nFPM_max+1)[None,:]*mask_lam/nFPM
                
dr = 1/nPup        
        
# revisit of the Hankel kernel for the focal plane mask (FPM)
dmi = rMask/nFPM2
# FPM image plane coordinate
mi  = np.arange(nFPM2 +1)*dmi
# FPM image plane coordinate weighted with wavelength
mii = mi[None,:]*lam0/lam_t[:,None]

dxi = Fmax/nImg
# Final image plane coordinate
xi  = np.arange(nImg+1)*dxi
# final image plane coordinate weighted with wavelength
xii = xi[None,:]*lam0/lam_t[:,None]


#%%

# Hankel kernel for the focal plane mask (FPM) 
hankel_kernel_FPM_all  = besselJ0(np.pi/R*xi_FPM_lam[:,:,None]*r[None,None,:])
hankel_kernel_iFPM_all = besselJ0(np.pi/R*xi_FPM_lam[:,None,:]*r[None,:,None])

# Hankel kernel (no wavelength variation)
hankel_kernel     = besselJ0(np.pi/R*xi[:,None]*r[None,:])
# Hankel kernel (including wavelength variation)
hankel_kernel_all = besselJ0(np.pi/R*xii[:,:,None]*r[None,None,:])


#%%

# Hankel kernel (including wavelength variation)
FPM_block = np.pi*(lam0/lam_t[:,None, None])*mi[None,:,None]*r[None,None,:]

HK_FPM_poly = np.pi*(lam0/lam_t[:,None, None])*besselJ0(FPM_block)*r[None, None, :]*dr
# Hankel kernel (inverse transform)
iHK_FPM_poly = np.pi*(lam0/lam_t[:,None, None])*besselJ0(FPM_block.transpose(0,2,1))*mi[None, None, :]*dmi


#%%

block = np.pi*(lam0/lam_t[:,None,None])*xi[None,:,None]*r[None,None,:]

HK_poly = np.pi*(lam0/lam_t[:,None, None])*besselJ0(block)*r[None, None, :]*dr


#%% # direct propagation (no focal plane mask)
def compute_direct_field_1d_uno(Apod):
    """
    Computes the electric field of the direct image with APLC
    for the 1D problem.
    
    Parameters
    ---------- 
    Apod : array_like
        Entrance pupil apodization :math:`\Phi`
            
    Returns    
    ----------
    res : array_like
        Direct electric field :math:`\Psi_0` at all the wavelengths
        
    """
    return lam0/lam_t[:,None]*np.pi*hankel_kernel_all.dot(\
        Pupil1d*Apod*LyotStop1d*r)*dr

#%% # propagation through coronagraph (with focal plane mask)
def compute_corono_field_1d_uno(Apod):
    """
    Computes the electric field of the coronagraphic image with APLC
    for the 1D problem.
    
    Parameters
    ---------- 
    Apod : array_like
        Entrance pupil apodization :math:`\Phi`
            
    Returns    
    ----------
    res : array_like
        Coronagraphic electric field :math:`\Psi_D` at all the wavelengths
        
    """
    FPM_field  = np.pi*hankel_kernel_FPM_all.dot(
            Apod*Pupil1d*r/R)\
            *(R/nPup)*xi_FPM_lam
    
    iFPM_field = np.zeros((nlam,nPup))
    for i in range(nlam):
        iFPM_field[i,:] = np.pi*hankel_kernel_iFPM_all[i,:,:].dot(
                FPM_field[i,:])*(1/nFPM)
    
    nolyot_field = (Apod[None,:]*Pupil1d[None,:]-iFPM_field)\
            *r[None,:]/R
    
    lyot_field   = nolyot_field*LyotStop1d[None,:]
    
    corono_field_tmp = np.zeros((nlam,nImg+1))
    for i in range(nlam):
        corono_field_tmp[i,:] = hankel_kernel_all[i,:,:].dot(
                lyot_field[i,:])
    
    return lam0/lam_t[:,None]*np.pi*corono_field_tmp*R/nPup
#    return FPM_field

#%%
def compute_corono_field_1d_bis(Apod):
    """
    Computes the electric field of the coronagraphic image with APLC
    for the 1D problem.
    
    Parameters
    ---------- 
    Apod : array_like
        Entrance pupil apodization :math:`\Phi`
            
    Returns    
    ----------
    corono_field : array_like
        Coronagraphic electric field :math:`\Psi_D` at all the wavelengths
        
    """
    E_field = Apod*Pupil1d
#    FPM_field = np.zeros((nlam,nFPM+1))
#    for i in range(nlam):
#        FPM_field[i] = HK_FPM_poly[i].dot(E_field)
    FPM_field = np.einsum("ijk,k->ij", HK_FPM_poly, E_field)
               
#    iFPM_field = np.zeros((nlam,nPup))
#    for i in range(nlam):
#        iFPM_field[i] = iHK_FPM_poly[i].dot(FPM_field[i])
    iFPM_field = np.einsum("ijk,ik->ij", iHK_FPM_poly, FPM_field)
                            
    lyot_field = (Apod[None,:]*Pupil1d[None,:]-iFPM_field)*LyotStop1d[None,:]
    
#    corono_field = np.zeros((nlam,nImg+1))
#    for i in range(nlam):
#        corono_field[i] = HK_poly[i].dot(lyot_field[i])
    corono_field = np.einsum("ijk,ik-> ij", HK_poly, lyot_field)
    
    return corono_field
#    return FPM_field

#%% # propagation through coronagraph (with focal plane mask)
def compute_fpm_field_1d_uno(Apod):
    """
    Computes the electric field of the coronagraphic image with APLC
    for the 1D problem.
    
    Parameters
    ---------- 
    Apod : array_like
        Entrance pupil apodization :math:`\Phi`
            
    Returns    
    ----------
    res : array_like
        Coronagraphic electric field :math:`\Psi_D` at all the wavelengths
        
    """
    FPM_field  = np.pi*hankel_kernel_FPM_all.dot(
            Apod*Pupil1d*r/R)\
            *(R/nPup)*xi_FPM_lam
    
    return FPM_field


#%%
def compute_fpm_field_1d_bis(Apod):
    """
    Computes the electric field of the coronagraphic image with APLC
    for the 1D problem.
    
    Parameters
    ---------- 
    Apod : array_like
        Entrance pupil apodization :math:`\Phi`
            
    Returns    
    ----------
    corono_field : array_like
        Coronagraphic electric field :math:`\Psi_D` at all the wavelengths
        
    """
    E_field = Apod*Pupil1d
#    FPM_field = np.zeros((nlam,nFPM+1))
#    for i in range(nlam):
#        FPM_field[i] = HK_FPM_poly[i].dot(E_field)
    FPM_field = np.einsum("ijk,k->ij", HK_FPM_poly, E_field)
               
    return FPM_field

#%%
Psi1D_D0_mono =  compute_direct_field_1d_uno(Pupil1d)        
Psi1D_D1_mono =  compute_corono_field_1d_uno(Pupil1d)
Psi1D_D2_mono =  compute_corono_field_1d_bis(Pupil1d)  

Int1D_D0_mono = np.abs(Psi1D_D0_mono)**2
Int1D_D1_mono = np.abs(Psi1D_D1_mono)**2
Int1D_D2_mono = np.abs(Psi1D_D2_mono)**2

nInt1D_D0_mono = Int1D_D0_mono/np.max(Int1D_D0_mono[nlam//2])
nInt1D_D1_mono = Int1D_D1_mono/np.max(Int1D_D0_mono[nlam//2])
nInt1D_D2_mono = Int1D_D2_mono/np.max(Int1D_D0_mono[nlam//2])

Int1D_D0_poly = np.sum(Int1D_D0_mono, 0)
Int1D_D1_poly = np.sum(Int1D_D1_mono, 0)
Int1D_D2_poly = np.sum(Int1D_D2_mono, 0)

nInt1D_D0_poly = Int1D_D0_poly/np.max(Int1D_D0_poly)
nInt1D_D1_poly = Int1D_D1_poly/np.max(Int1D_D0_poly)
nInt1D_D2_poly = Int1D_D2_poly/np.max(Int1D_D0_poly)


#%%
"""
2D corono
"""


corono0 = coro.design.APLC2d(**params)
Int2D_D0_poly = corono0.compute_direct_intensity_2d(Pupil2d)
Int2D_D1_poly = corono0.compute_corono_intensity_2d(Pupil2d)
nInt2D_D0_poly = Int2D_D0_poly/np.max(Int2D_D0_poly)
nInt2D_D1_poly = Int2D_D1_poly/np.max(Int2D_D0_poly)

nInt2D_D0_poly_vec = nInt2D_D0_poly[nImg2d//2,nImg2d//2:]
nInt2D_D1_poly_vec = nInt2D_D1_poly[nImg2d//2,nImg2d//2:]

xi2d = corono0.xi2d[:-1]


#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

#fname_pl = fname_gen + '_intensity.pdf'
#fpath    = fdir_plot / fname_pl
pl.figure(0, figsize=(6, 4.5))
pl.clf()
pl.title('1D broadband coronagraphic images')

pl.semilogy(xi,nInt1D_D0_poly,label='1D direct v1')

pl.semilogy(xi,nInt1D_D1_poly,label='1D corono v1')
pl.semilogy(xi,nInt1D_D2_poly,label='1D corono v2', linestyle='--')

pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-6, 2e0)
pl.legend()
pl.tight_layout()
#pl.savefig(str(fpath))

#%%
pl.figure(1, figsize=(6, 4.5))
pl.clf()
pl.title('1D vs 2D broadband coronagraphic images')

pl.semilogy(xi,nInt1D_D0_poly,label='1D direct v1')
pl.semilogy(xi2d,nInt2D_D0_poly_vec,label='2D direct', linestyle = '--')

pl.semilogy(xi,nInt1D_D1_poly,label='1D corono v1')
pl.semilogy(xi,nInt1D_D2_poly,label='1D corono v2', linestyle='--')
pl.semilogy(xi2d,nInt2D_D1_poly_vec,label='2D corono', linestyle='--')

pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-6, 2e0)
pl.legend()
pl.tight_layout()

#%%
#fname_pl = fname_gen + '_intensity.pdf'
#fpath    = fdir_plot / fname_pl
pl.figure(2, figsize=(6, 4.5))
pl.clf()
pl.title('1d difference of direct image profiles')
#pl.semilogy(xi,nInt_D0_poly,)
pl.semilogy(xi[:-1],np.abs(nInt1D_D2_poly[:-1]-nInt1D_D1_poly[:-1]),label='|1Dv2-1Dv1|')

pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-12, 2e0)
pl.legend()
pl.tight_layout()
#pl.savefig(str(fpath))


#%%
#fname_pl = fname_gen + '_intensity.pdf'
#fpath    = fdir_plot / fname_pl
pl.figure(3, figsize=(6, 4.5))
pl.clf()
pl.title('1d vs 2D difference')
#pl.semilogy(xi,nInt_D0_poly,)
pl.semilogy(xi[:-1],np.abs(nInt1D_D0_poly[:-1]-nInt2D_D0_poly_vec), label='direct')
pl.semilogy(xi[:-1],np.abs(nInt1D_D1_poly[:-1]-nInt2D_D1_poly_vec), label='corono (v1)')
pl.semilogy(xi[:-1],np.abs(nInt1D_D2_poly[:-1]-nInt2D_D1_poly_vec), label='corono (v2)')

pl.axvline(x=rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=xi.min(), xmax=xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-12, 2e0)
pl.legend()
pl.tight_layout()
#pl.savefig(str(fpath))

#%%
#print(np.max(np.abs(nInt_D2_poly-nInt_D1_poly)))

print('max diff 1d   v 2d   direct: {0}'.format(np.max(np.abs(nInt1D_D0_poly[:-1]-nInt2D_D0_poly_vec))))

print('max diff 1dv1 v 2d   corono: {0}'.format(np.max(np.abs(nInt1D_D1_poly[:-1]-nInt2D_D1_poly_vec))))
print('max diff 1dv2 v 2d   corono: {0}'.format(np.max(np.abs(nInt1D_D2_poly[:-1]-nInt2D_D1_poly_vec))))

print('max diff 1dv1 v 1dv2 corono: {0}'.format(np.max(np.abs(nInt1D_D2_poly[:-1]-nInt1D_D1_poly[:-1]))))