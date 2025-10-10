#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:52:33 2024

@author: mndiaye
"""
import numpy as np
import time
import os
from pathlib import Path

import corono as coro

from astropy.io import fits

import sys
import pwd

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

try:
    import gurobipy as gb
except ImportError:
    gb = False
    print('gb is False')   
    
import matplotlib.pyplot as plt

#import cv2

#%% parameters
"""
Parameters
"""
# Telescope name
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'vlt_btw_compass' # 'vlt' or 'sbr' or 'lvr' or 'vlt_btw'
problem_name = 'MaxContrastL1' # 'MaxContrastLinf' #  ,'MaxTau' #  'MaxContrastL1' #
solver       = 'gurobipy' #,'stdgrb' #  'gurobipy', 'scipy.linprog'
slvLogToConsole = 1
slvCrossover    = 0
slvMethod       = 2
slvSparse       = 0
allLogToConsole = 1

MinIsland   = False
Binarity    = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1

#nPup = corono0.params['nPup']
red_factor = 0.99
nPup0 = 400 # 506 #512
nExt0 = 0
nDim0 = nPup0 + nExt0
nFPM = 50
Fmax2d = 45 #75 # 45#22.5
nImg2d = 90 #150 # 90#45

LS_OD = 0.96
nPupLS = int(nPup0*LS_OD)

# number of progressive refinement
nProgRef = 1

# mask radius in lam0/D units
#rMask = 1.766 # ALC1 at 1.593um (145mas) 
rMask = 2.252 # ALC2 at 1.593um (185mas)


# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  0.0
rho1 = 30.0

# contrast in the dark region
cDarkHole = 10.0

# tau (integrated Pupil transmission)
tau   = 0.756 #/0.954
# CtrBtwnPix2
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True
ImPart      = True 
LSRobustness = True # (new robustness approach using derivative of the field with respect to Lyot stop displacement)
# kwd_qrt = True


#nlam
bw   = 0.2
nlam = 3
shift_tot0 = (np.sqrt(2.)/0.5)*int(0.005*nPup0) #np.sqrt(shift_x**2+shift_y**2)
alpha = 0 # np.pi/3 # np.arctan2(shift_x, shift_y)
shift_y = shift_tot0*np.cos(alpha)
shift_x = shift_tot0*np.sin(alpha)

LSRobustness_coeff_v9 = shift_tot0*(2**(nProgRef-1))

str_LSRcoeff_v9 = ''
if LSRobustness:
    str_LSRcoeff_v9 = f'LSRcoeff_v9={int(np.round(LSRobustness_coeff_v9*1e3)):05d}'

print(str_LSRcoeff_v9)

# Lyot stop with dead actuators
str_dead_act = ''
do_dead_act = True
if do_dead_act:
    Pupil2dSym = False
    ImPart = True
    str_dead_act = '_deadact'

str_red_factor = ''    
if red_factor != 1.:
    str_red_factor = f'_nPup={int(np.round(red_factor*nPup0))}'

do_fits = True


#%%
"""
### Functions
"""
dtype0 = 'float64'
if ImPart is True:
    dtype0 = 'complex128'

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

ncorono = 1
neps = 1 

# wavelengths
lam0 = 1.        
dlam = bw*lam0
lam_t = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam)
    
# Focal plane mask
mask2d = coro.utils.uniform_disk(nFPM, nFPM/2., CtrBtwnPix=CtrBtwnPix)
mask2d_qrt = mask2d[nFPM//2:,nFPM//2:]

# mask size at a given wavelength for SFT
mB_t  = 2.*rMask*(lam0/lam_t)*(nDim0/nPup0)
mD_t  = Fmax2d*(lam0/lam_t)*(nDim0/nPupLS)


#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').resolve()
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').expanduser()
        fdir_sav = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
        sim_case = 'test' # 'test' or 'server'
    elif syst == 'linux':
        fdir = Path('/scratch/mndiaye/data/Coronagraphs/data/2D/pupils/').resolve()
        fdir_sav = Path('/scratch/mndiaye/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
        sim_case = 'server' # 'test' or 'server'            
    else:
        raise ValueError('Unknown operating system {0}'.format(user))
elif user == 'ndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils/').expanduser()
        fdir_sav = Path('/Users/mndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
        sim_case = 'test' # 'test' or 'server'
    elif syst == 'linux':
        fdir = Path('/home/ndiaye/scratch/data/Coronagraphs/data/2D/pupils/').resolve()
        fdir_sav = Path('/home/ndiaye/scratch/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
        sim_case = 'server' # 'test' or 'server'            
    else:
        raise ValueError('Unknown operating system {0}'.format(user))    
else:
    raise ValueError('Unknown user {0}'.format(user))


#%%       
#    @profile
def compute_response_matrices(idx_pup, idx_dz, npp, ndz, corono=None):
    r"""
    Computes the response matrix for the coronagraph with and without 
    the focal plane mask.
    
    Notes
    -----        
    corono_field_re_t_tmp, corono_field_im_t_tmp : array_like, array_like
        Real and imaginary parts of the coronagraphic response matrix
        for all the points in the pupil :math:`P_0` and at all the wavelengths

    corono_field_re_t, corono_field_im_t : array_like, array_like
        Real and imaginary parts of the coronagraphic response matrix
        for all the points in the pupil :math:`P_0` and at all the wavelengths.
        These arrays are sliced from corono_field_re_t_tmp, corono_field_im_t_tmp 
        for the points inside the region of interest in the final image plane.

    corono_field_t : array_like
        Concatenation of the real and imaginary parts of the coronagraphic 
        response matrix for all the points in the pupil :math:`P_0` and 
        at all the wavelengths
            
    """        
    if corono is None:
        pass
    else:
        corono_field_re_t_tmp = np.empty((npp, corono.nlam, 
                                               corono.nImg2d**2))

        Apod2d = np.zeros((corono.nPup, corono.nPup))


        if ImPart is True:             
            corono_field_im_t_tmp = np.empty((npp, corono.nlam, 
                                               corono.nImg2d**2))

            for i, val in enumerate(idx_pup):
                (i0,j0) = np.unravel_index(val, (corono.nPup, corono.nPup))
                Apod2d[i0,j0] = 1            
                corono_field_re_t_tmp[i], corono_field_im_t_tmp[i] = \
                corono.compute_corono_field_2d_vec(Apod2d)
                Apod2d[i0,j0] = 0 
                
            corono_field_re_t = np.reshape(
                    corono_field_re_t_tmp[:,:, idx_dz], 
                    (npp, corono.nlam*ndz))
            
            corono_field_re_t_tmp = None
            del corono_field_re_t_tmp
        
            corono_field_im_t = np.reshape(
                corono_field_im_t_tmp[:,:, idx_dz], 
                (npp, corono.nlam*ndz))
            corono_field_re_t = np.concatenate((corono_field_re_t, 
                                                corono_field_im_t), axis=1)
            
            corono_field_im_t = None
            del corono_field_im_t
        
            corono_field_im_t_tmp = None
            del corono_field_im_t_tmp

        else:    
            for i, val in enumerate(idx_pup):
                (i0,j0) = np.unravel_index(val, (corono.nPup, corono.nPup))
                Apod2d[i0,j0] = 1            
                corono_field_re_t_tmp[i] = \
                corono.compute_corono_field_2d_real_vec(Apod2d)
                Apod2d[i0,j0] = 0 
                
            corono_field_re_t = np.reshape(
                    corono_field_re_t_tmp[:,:, idx_dz], 
                    (npp, corono.nlam*ndz))
            
            corono_field_re_t_tmp = None
            del corono_field_re_t_tmp

        return corono_field_re_t

#%%
def sft_qrt(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square. 

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    NA    = 2*np.shape(A2)[0]
    coeff = (m)/(NA*NB)
    
    U = np.zeros((1,NB//2))
    X = np.zeros((1,NA//2))
    
    X[0,:] = (1./NA)*(np.arange(NA//2)+val)
    U[0,:] = (m/NB)*(np.arange(NB//2)+val)
       
    XU = 2.*np.pi* X.T.dot(U)
    A3 = np.cos(XU)
    A1 = A3.T
    
    B  = A1.dot(A2.dot(A3))

    return 4*coeff*B

#%%
def isft_qrt(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].

    See Also
    --------
    sft() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft_qrt(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

#%%
def sft_hlf(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square. 

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    
    sign = -1.0
    if inv:
        sign = 1.0

    U = np.zeros((1,NB))
    X = np.zeros((1,NA))
    
    X[0,:] = (1./NA)*(np.arange(NA)-NA/2.+val)    
    U[0,:] =  (m/NB)*(np.arange(NB)-NB/2.+val)
       
    XU = 2.*np.pi* X.T.dot(U)
    YV = 2.*np.pi* X.T.dot(U[:,:NB//2])
    
    A3 = sign*1j*np.sin(XU)  +np.cos(XU)
    A1 = (sign*1j*np.sin(YV) +np.cos(YV)).T
    
    B  = A1.dot(A2.dot(A3))

    return coeff*B

#%%
def isft_hlf(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].

    See Also
    --------
    sft() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft_hlf(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

#%%

def compute_corono_field_2d_qrt(Apod2d_qrt, Pupil2d_qrt, LyotStop2d_qrt, corono=None):
    """
    Computes the coronagraph electric field for a classical Lyot coronagraph
    with four planes (A: entrance pupil, B: intermediate focal plane, 
    C: relayed pupil before stop, L: relayed pupil after stop, 
    D: final image plane).
    Resolution element are given in :math:`\lambda_0/D` where 
    :math:`\lambda_0` and :math:`D` denote the central and the telescope 
    diameter.

    Parameters
    ---------- 
    Apod2d : array_like 
        Entrance pupil apodization :math:`\Phi`
        
    Returns    
    ----------    
    field_Dtmp : array_like
        Coronagraphic electric field :math:`\Psi_D` in the final image plane
        at all the wavelengths
        
    """        
    if corono is None:
        pass
    else:
        field_A_qrt    = Apod2d_qrt*Pupil2d_qrt
                    
        field_Dtmp_qrt = np.zeros((corono.nlam,corono.nImg2d//2,corono.nImg2d//2), dtype=dtype0)
    
    
        for i in range(corono.nlam):                       
            field_B_qrt       = mask2d_qrt*sft_qrt(field_A_qrt, corono.nFPM, corono.mB_t[i], 
                                            CtrBtwnPix=True)
            field_C_qrt       = field_A_qrt - isft_qrt(field_B_qrt, corono.nPup, corono.mB_t[i], 
                                           CtrBtwnPix=True)
            field_L_qrt       = field_C_qrt*LyotStop2d_qrt
            field_Dtmp_qrt[i] = sft_qrt(field_L_qrt, corono.nImg2d, corono.mD_t[i], 
                      CtrBtwnPix=True)
                         
        return field_Dtmp_qrt


#%%    
def compute_corono_field_2d_LSasym(Apod2d_qrt, Pupil2d_qrt, LyotStop2d, corono=None):
    """
    Computes the coronagraph electric field for a classical Lyot coronagraph
    with four planes (A: entrance pupil, B: intermediate focal plane, 
    C: relayed pupil before stop, L: relayed pupil after stop, 
    D: final image plane).
    Resolution element are given in :math:`\lambda_0/D` where 
    :math:`\lambda_0` and :math:`D` denote the central and the telescope 
    diameter.

    Parameters
    ---------- 
    Apod2d : array_like 
        Entrance pupil apodization :math:`\Phi`
        
    Returns    
    ----------    
    field_Dtmp : array_like
        Coronagraphic electric field :math:`\Psi_D` in the final image plane
        at all the wavelengths
        
    """        
    if corono is None:
        pass
    else:
        field_A_qrt    = Apod2d_qrt*Pupil2d_qrt
                    
        field_Dtmp = np.zeros((corono.nlam,corono.nImg2d,corono.nImg2d), dtype=dtype0)
    
        for i in range(corono.nlam):                       
            field_B_qrt       = mask2d_qrt*sft_qrt(field_A_qrt, corono.nFPM, corono.mB_t[i], 
                                            CtrBtwnPix=True)
            field_C_qrt       = field_A_qrt - isft_qrt(field_B_qrt, corono.nPup, corono.mB_t[i], 
                                           CtrBtwnPix=True)
            
            field_C           = np.zeros((corono.nPup, corono.nPup))
            field_C[corono.nPup//2:, corono.nPup//2:] = field_C_qrt
            field_C[:corono.nPup//2, corono.nPup//2:] = np.flip(field_C_qrt, axis=0)
            field_C[:, :corono.nPup//2]               = np.flip(field_C[:, corono.nPup//2:], axis=1)
            
            field_L       = field_C*LyotStop2d
            field_Dtmp[i] = coro.utils.sft(field_L, corono.nImg2d, corono.mD_t[i], 
                      CtrBtwnPix=True)
        return field_Dtmp


#%%
def compute_response_matrices_qrt(idx_pup, idx_dz, npp, ndz,corono=None):
    r"""
    Computes the response matrix for the coronagraph with and without 
    the focal plane mask.
    
    Notes
    -----        
    corono_field_re_t_tmp, corono_field_im_t_tmp : array_like, array_like
        Real and imaginary parts of the coronagraphic response matrix
        for all the points in the pupil :math:`P_0` and at all the wavelengths

    corono_field_re_t, corono_field_im_t : array_like, array_like
        Real and imaginary parts of the coronagraphic response matrix
        for all the points in the pupil :math:`P_0` and at all the wavelengths.
        These arrays are sliced from corono_field_re_t_tmp, corono_field_im_t_tmp 
        for the points inside the region of interest in the final image plane.

    corono_field_t : array_like
        Concatenation of the real and imaginary parts of the coronagraphic 
        response matrix for all the points in the pupil :math:`P_0` and 
        at all the wavelengths
            
    """        
    if corono is None:
        pass
    else:

        Pupil2d_qrt = corono.Pupil2d[corono.nPup//2:,corono.nPup//2:]
        Apod2d_qrt = np.zeros((corono.nPup//2, corono.nPup//2))         

        if ImPart is True:
            corono_field_re_t_tmp = np.empty((npp, corono.nlam, 
                                                   corono.nImg2d**2))             
            corono_field_im_t_tmp = np.empty((npp, corono.nlam, 
                                               corono.nImg2d**2))

            for i, val in enumerate(idx_pup):
                (i0,j0) = np.unravel_index(val, (corono.nPup//2, corono.nPup//2))
                Apod2d_qrt[i0,j0] = 1            
                test = (1./4)*compute_corono_field_2d_LSasym(Apod2d_qrt, Pupil2d_qrt, corono.LyotStop2d, corono)
                corono_field_re_t_tmp[i] = np.reshape(test.real, (nlam, nImg2d**2))
                corono_field_im_t_tmp[i] = np.reshape(test.imag, (nlam, nImg2d**2))
                Apod2d_qrt[i0,j0] = 0 
                
            corono_field_re_t = np.reshape(
                    corono_field_re_t_tmp[:,:, idx_dz], 
                    (npp, corono.nlam*ndz))
            
            corono_field_re_t_tmp = None
            del corono_field_re_t_tmp
        
            corono_field_im_t = np.reshape(
                corono_field_im_t_tmp[:,:, idx_dz], 
                (npp, corono.nlam*ndz))

            corono_field_im_t_tmp = None
            del corono_field_im_t_tmp
            
            return corono_field_re_t, corono_field_im_t

        else:
            corono_field_re_t_tmp = np.empty((npp, corono.nlam, 
                                                   (corono.nImg2d//2)**2))
            LyotStop2d_qrt = corono.LyotStop2d[corono.nPup//2:,corono.nPup//2:]

            
            for i, val in enumerate(idx_pup):
                (i0,j0) = np.unravel_index(val, (corono.nPup//2, corono.nPup//2))
                Apod2d_qrt[i0,j0] = 1
                test_qrt = (1./4)*compute_corono_field_2d_qrt(Apod2d_qrt, Pupil2d_qrt, LyotStop2d_qrt, corono)
                corono_field_re_t_tmp[i] = \
                np.reshape(test_qrt, (corono.nlam, (corono.nImg2d//2)**2))
                # corono.compute_corono_field_2d_real_vec(Apod2d)
                Apod2d_qrt[i0,j0] = 0 
                
            corono_field_re_t = np.reshape(
                    corono_field_re_t_tmp[:,:, idx_dz], 
                    (npp, corono.nlam*ndz))
            
            corono_field_re_t_tmp = None
            del corono_field_re_t_tmp

            return corono_field_re_t

#%%

def compute_corono_field_2d_qrt_LSrobustness_bis(Apod2d_qrt, Pupil2d_qrt, LyotStop2d_qrt, dist2d, corono=None):
    """
    Computes the coronagraph electric field for a classical Lyot coronagraph
    with four planes (A: entrance pupil, B: intermediate focal plane, 
    C: relayed pupil before stop, L: relayed pupil after stop, 
    D: final image plane).
    Resolution element are given in :math:`\lambda_0/D` where 
    :math:`\lambda_0` and :math:`D` denote the central and the telescope 
    diameter.

    Parameters
    ---------- 
    Apod2d : array_like 
        Entrance pupil apodization :math:`\Phi`
        
    Returns    
    ----------    
    field_Dtmp : array_like
        Coronagraphic electric field :math:`\Psi_D` in the final image plane
        at all the wavelengths
        
    """        
    if corono is None:
        pass
    else:
        field_A_qrt    = Apod2d_qrt*Pupil2d_qrt
                    
        field_Dtmp_qrt = np.zeros((corono.nlam,corono.nImg2d//2,corono.nImg2d//2), dtype=dtype0)
        field_Ftmp_qrt = np.zeros((corono.nlam,corono.nImg2d//2,corono.nImg2d//2), dtype=dtype0)
    
    
        for i in range(corono.nlam):                       
            field_B_qrt       = mask2d_qrt*sft_qrt(field_A_qrt, corono.nFPM, corono.mB_t[i], 
                                            CtrBtwnPix=True)
            field_C_qrt       = field_A_qrt - isft_qrt(field_B_qrt, corono.nPup, corono.mB_t[i], 
                                           CtrBtwnPix=True)
            field_L_qrt       = field_C_qrt*LyotStop2d_qrt
            field_Dtmp_qrt[i] = sft_qrt(field_L_qrt, corono.nImg2d, corono.mD_t[i], 
                      CtrBtwnPix=True)
            
            """
            ### Expression below is not valid 
            """
            
            field_Ftmp_qrt[i] = field_Dtmp_qrt[i]*(np.exp((1j*2.*np.pi/corono.lam_t[i])*dist2d))
            
                         
        return field_Ftmp_qrt


#%%    
def compute_corono_field_2d_LSasym_LSrobustness_bis(Apod2d_qrt, Pupil2d_qrt, LyotStop2d, dist2d, str_axis, corono=None):
    """
    Computes the coronagraph electric field for a classical Lyot coronagraph
    with four planes (A: entrance pupil, B: intermediate focal plane, 
    C: relayed pupil before stop, L: relayed pupil after stop, 
    D: final image plane).
    Resolution element are given in :math:`\lambda_0/D` where 
    :math:`\lambda_0` and :math:`D` denote the central and the telescope 
    diameter.

    Parameters
    ---------- 
    Apod2d : array_like 
        Entrance pupil apodization :math:`\Phi`
        
    Returns    
    ----------    
    field_Dtmp : array_like
        Coronagraphic electric field :math:`\Psi_D` in the final image plane
        at all the wavelengths
        
    """        
    if corono is None:
        pass
    else:
        field_A_qrt    = Apod2d_qrt*Pupil2d_qrt
                    
        # field_Dtmp = np.zeros((corono.nlam,corono.nImg2d,corono.nImg2d), dtype=dtype0)
        field_Dtmp_shift_grad_xy = np.zeros((corono.nlam,corono.nImg2d,corono.nImg2d), dtype=dtype0)
        
         # dot product term insde the complex exponential to represent the shift in spatial domain
        if str_axis == 'x':
            xixy = (2.*np.pi)*xxp
        elif str_axis == 'y':
            xixy = (2.*np.pi)*yyp
        else:
            raise NameError(f'str_axis is {str_axis}, should be x or y')


        # direct Fourier transform of the Lyot stop 
        FT_LyotStop2d = coro.utils.sft(LyotStop2d, 2*nDim, nDim, 
                  CtrBtwnPix=True)
        
        # FT of the Lyot stop multiplied by the exponential term to represent the shift in spatial domain
        weighted_FT_LyotStop2d_xy = FT_LyotStop2d*(-1j*xixy)
        
        # shifted Lyot stop using FTs
        LyotStop2d_shift_ana_xy = coro.utils.isft(weighted_FT_LyotStop2d_xy, nDim, nDim, 
                  CtrBtwnPix=True)

    
        for i in range(corono.nlam):                       
            field_B_qrt       = mask2d_qrt*sft_qrt(field_A_qrt, corono.nFPM, corono.mB_t[i], 
                                            CtrBtwnPix=True)
            field_C_qrt       = field_A_qrt - isft_qrt(field_B_qrt, corono.nPup, corono.mB_t[i], 
                                           CtrBtwnPix=True)
            
            field_C           = np.zeros((corono.nPup, corono.nPup))
            field_C[corono.nPup//2:, corono.nPup//2:] = field_C_qrt
            field_C[:corono.nPup//2, corono.nPup//2:] = np.flip(field_C_qrt, axis=0)
            field_C[:, :corono.nPup//2]               = np.flip(field_C[:, corono.nPup//2:], axis=1)
            
            # field_L       = field_C*LyotStop2d
            # field_Dtmp[i] = coro.utils.sft(field_L, corono.nImg2d, corono.mD_t[i], 
            #           CtrBtwnPix=True)
       
            # electric field in the re-imaged pupil plane C after the shifted Lyot stop
            field_L_shift_ana_xy = field_C*LyotStop2d_shift_ana_xy
            
            # electric field in the final image plane D after the shifted Lyot stop
            field_Dtmp_shift_grad_xy[i] = coro.utils.sft(field_L_shift_ana_xy, corono.nImg2d, corono.mD_t[i], 
                      CtrBtwnPix=True) 
                            
        return field_Dtmp_shift_grad_xy


#%%
def compute_response_matrices_qrt_LSrobustness_bis(idx_pup, idx_dz, npp, ndz, str_axis, corono=None):
    r"""
    Computes the response matrix for the coronagraph with and without 
    the focal plane mask.
    
    Notes
    -----        
    corono_field_re_t_tmp, corono_field_im_t_tmp : array_like, array_like
        Real and imaginary parts of the coronagraphic response matrix
        for all the points in the pupil :math:`P_0` and at all the wavelengths

    corono_field_re_t, corono_field_im_t : array_like, array_like
        Real and imaginary parts of the coronagraphic response matrix
        for all the points in the pupil :math:`P_0` and at all the wavelengths.
        These arrays are sliced from corono_field_re_t_tmp, corono_field_im_t_tmp 
        for the points inside the region of interest in the final image plane.

    corono_field_t : array_like
        Concatenation of the real and imaginary parts of the coronagraphic 
        response matrix for all the points in the pupil :math:`P_0` and 
        at all the wavelengths
            
    """        
    if corono is None:
        pass
    else:

        Pupil2d_qrt = corono.Pupil2d[corono.nPup//2:,corono.nPup//2:]
        Apod2d_qrt = np.zeros((corono.nPup//2, corono.nPup//2))         

        if ImPart is True:
            corono_field_re_t_tmp = np.empty((npp, corono.nlam, 
                                                   corono.nImg2d**2))             
            corono_field_im_t_tmp = np.empty((npp, corono.nlam, 
                                               corono.nImg2d**2))

            for i, val in enumerate(idx_pup):
                (i0,j0) = np.unravel_index(val, (corono.nPup//2, corono.nPup//2))
                Apod2d_qrt[i0,j0] = 1
                test = 0
                test = (1./4)*compute_corono_field_2d_LSasym_LSrobustness_bis(Apod2d_qrt, Pupil2d_qrt, corono.LyotStop2d, dist2d, str_axis, corono)    
                corono_field_re_t_tmp[i] = np.reshape(test.real, (nlam, nImg2d**2))
                corono_field_im_t_tmp[i] = np.reshape(test.imag, (nlam, nImg2d**2))
                Apod2d_qrt[i0,j0] = 0 
                
            corono_field_re_t = np.reshape(
                    corono_field_re_t_tmp[:,:, idx_dz], 
                    (npp, corono.nlam*ndz))
            
            corono_field_re_t_tmp = None
            del corono_field_re_t_tmp
        
            corono_field_im_t = np.reshape(
                corono_field_im_t_tmp[:,:, idx_dz], 
                (npp, corono.nlam*ndz))

            corono_field_im_t_tmp = None
            del corono_field_im_t_tmp
            
            return corono_field_re_t, corono_field_im_t

        else:
            corono_field_re_t_tmp = np.empty((npp, corono.nlam, 
                                                   (corono.nImg2d//2)**2))
            LyotStop2d_qrt = corono.LyotStop2d[corono.nPup//2:,corono.nPup//2:]

            
            for i, val in enumerate(idx_pup):
                (i0,j0) = np.unravel_index(val, (corono.nPup//2, corono.nPup//2))
                Apod2d_qrt[i0,j0] = 1
                test_qrt = (1./4)*compute_corono_field_2d_qrt_LSrobustness_bis(Apod2d_qrt, Pupil2d_qrt, LyotStop2d_qrt, corono)
                corono_field_re_t_tmp[i] = \
                np.reshape(test_qrt, (corono.nlam, (corono.nImg2d//2)**2))
                # corono.compute_corono_field_2d_real_vec(Apod2d)
                Apod2d_qrt[i0,j0] = 0 
                
            corono_field_re_t = np.reshape(
                    corono_field_re_t_tmp[:,:, idx_dz], 
                    (npp, corono.nlam*ndz))
            
            corono_field_re_t_tmp = None
            del corono_field_re_t_tmp

            return corono_field_re_t



#%%
def get_filename(corono=None):
    """
    Generate a string of characters to define a filename with all the 
    parameters
    
    Parameters
    --------
    kwargs : dict
        parameters given by the user for the keys with the values to update
    
    Returns
    --------
    fname_gen : str
        generic string of characters for a filename
        
    
    """
    if corono is None:
        pass
    else:
        
        params = corono.params.copy()
        params = coro.utils.update_params(params)
                
        if problem_name == 'MaxTau':
            str_opt = '_C={cDarkHole:.1f}'
        elif problem_name == 'MaxContrastL1' or problem_name == 'MaxContrastLinf':
            str_opt = '_tau={tau:.3f}'
        else:
            raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))
    
        # str_FirstDerGlobal = ''
        # if MinIsland is True:
        #     str_FirstDerGlobal = '_1stderglo={FirstDerGlobalLim}'
            
        str_LSRobustness = ''
        if LSRobustness is True:
            str_LSRobustness = '_LSRobustness=1'
    
        fname_corono = corono.get_filename()             
            
        fname_gen_optim  = '{problem_name}' \
        + str_opt \
        + str_LSRobustness \
        + '_{solver}'
        
        
        return '{pupil_name}_'.format(**params) + fname_corono + fname_gen_optim.format(**params)

#%%
"""
### scale arrays
"""
Gray_2d = 0
Ones_2d = 0
ApoCst = 0

for k in range(nProgRef):
      
    
    nDim = nDim0*2**k
    nPup = nPup0*2**k
    shift_tot = shift_tot0*(2**k)
    if pupil_name == 'lvr':
        fname_pup = f'ATLAST_Aperture_nPup={nPup}.fits'
        fname_lys = f'ATLAST_LyotStop_nPup={nPup}.fits'
    else:
        fname_pup = f'pupil={pupil_name}_nArr={nPup0}_nPup={int(np.round(red_factor*nPup0))}.fits' #'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup0,)
        fname_lys = f'pupil={pupil_name}_nArr={nPup0}_nPup={int(np.round(red_factor*nPup0))}.fits' #'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup0,)
        if do_dead_act:
            fname_lys = f'sphere_stop_ST_ALC2_nPup{nPup:04d}.fits'
    
    fpath_pup = fdir / fname_pup
    fpath_lys = fdir / fname_lys
    
    Pupil2d = np.zeros((nDim, nDim))
    LyotStop2d = np.zeros((nDim, nDim))
    nIni = (nDim-nPup)//2
    nEnd = (nDim+nPup)//2
    Pupil2d[nIni:nEnd, nIni:nEnd] = fits.getdata(fpath_pup)
    LyotStop2d[nIni:nEnd, nIni:nEnd] = fits.getdata(fpath_lys)
    
    Pupil2d_qrt = Pupil2d[nDim//2:,nDim//2:]
    LyotStop2d_qrt = LyotStop2d[nDim//2:,nDim//2:]
    
    
    params0 = coro.to_dict(nPup=nDim, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                     rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                     CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                     nlam=nlam, bw=bw,
                     Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                     Pupil2dSym = Pupil2dSym, rMask=rMask,
                     problem_name = problem_name, 
                     solver = solver, 
                     corono_name = corono_name, pupil_name = pupil_name,
                     slvLogToConsole = slvLogToConsole,
                     slvCrossover = slvCrossover, slvMethod = slvMethod,
                     slvSparse = slvSparse,
                     allLogToConsole = allLogToConsole,
                     MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                     Binarity = Binarity, BinarityReg = BinarityReg,
                     ImPart = ImPart, LSRobustness = LSRobustness)
    
    #%%  
    """ 
    Coronagraph defintion
    """
    if corono_name == 'SP':
        corono0 = coro.design.SP2d(**params0)
    elif corono_name == 'APLC':
        corono0 = coro.design.APLC2d(**params0)
    else:
        raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
        
    if LSRobustness:
        print('Lyot Stop robustness')
        
        # pix_x_t = [1,0,1] #[-1, -1, -1,  0, 0,  1, 1, 1] # [-1, 1, 0, 0] #
        # pix_y_t = [0,1,1] #[-1,  0,  1, -1, 1, -1, 0, 1] # [0, 0, -1, 1] #
        # ncorono = len(pix_x_t)
        # # List construstion for the Lyot stops 
        # LyotStop2d_t = np.zeros((ncorono, nDim, nDim))
        # for l in range(ncorono):
        #     LyotStop2d_t[l] = np.roll(np.roll(LyotStop2d, (pix_x_t[l])*(2**k), axis=0), (pix_y_t[l])*(2**k), axis=1)
        
        # # list of parameters for each coronagraph configuration
        # params_t = []
        # # initialization of coronagraph list
        # corono_t = []
        # for l in range(ncorono):
        #     params_t.append(coro.update_params(params0, LyotStop2d=LyotStop2d_t[l])) 
        #     corono_t.append(coro.design.APLC2d(**params_t[l])) 

        
        
    #%%
    """
    ### identification of the gray points
    """
    if k != 0:
        Gray_2d2 = np.kron(Gray_2d,np.ones((2,2)))    
        Ones_2d2 = np.kron(Ones_2d,np.ones((2,2))) 
        
        #%%
        Ones_2d2_qrt = Ones_2d2[nDim//2:,nDim//2:]    
        Gray_2d2_qrt = Gray_2d2[nDim//2:,nDim//2:]
        
        Pupil2d2_qrt = Pupil2d[nDim//2:,nDim//2:]
        LyotStop2d2_qrt = LyotStop2d[nDim//2:,nDim//2:]
    
        ApoCst = Ones_2d2_qrt.sum()    
    
    
    #%%
    """
    Problem definition without class
    """
    print('definition of the variables for the problem')
    t0 = time.time()
    
    dz2d, rad2d = corono0.generate_area()
    
    xx,yy  = np.meshgrid(np.arange(nImg2d)-nImg2d//2+1/2, 
                         np.arange(nImg2d)-nImg2d//2+1/2)
    dist2d = (Fmax2d/nImg2d)*np.hypot(yy,xx)
    
    xyp = (nDim/(2*nDim))*(np.arange(2*nDim)-2*nDim//2+1/2)
    xxp, yyp  = np.meshgrid(xyp, xyp)
    
    #%%
    """
    ### Point selection in the image plane
    """
    if ImPart == True:
        dist2d_hlf = dist2d[:nImg2d//2, :] 
        dz2d_hlf  = dz2d[:nImg2d//2, :] 
        dz        = np.reshape(dz2d_hlf, (corono0.nImg2d*corono0.nImg2d//2))
        aaa       = np.arange(corono0.nImg2d*corono0.nImg2d//2)
        dist      = np.reshape(dist2d_hlf, (corono0.nImg2d*corono0.nImg2d//2))
    else:
        dist2d_qrt = dist2d[nImg2d//2:,nImg2d//2:] 
        dz2d_qrt = dz2d[nImg2d//2:,nImg2d//2:] 
        dz       = np.reshape(dz2d_qrt, ((corono0.nImg2d//2)**2))
        aaa      = np.arange((corono0.nImg2d//2)**2)
        dist     = np.reshape(dist2d_qrt, (corono0.nImg2d//2)**2)
    
        
    idx_dz  = list(aaa[dz == 1])  
    ndz     = len(idx_dz)
    
    if LSRobustness:
        idx_dist_tmp = dist[dz == 1]
        idx_dist = np.zeros((corono0.nlam, ndz))
        for i in range(corono0.nlam):
            idx_dist[i] = idx_dist_tmp*(2.*np.pi*lam0/lam_t[i])
        
        idx_dist= np.reshape(idx_dist, (corono0.nlam*ndz))


#%%    
    """
    ### Point selection in the pupil plane
    """
    if k == 0:
        Pupil_vec = np.reshape(Pupil2d_qrt, ((corono0.nPup//2)**2))
    else:
        Pupil_vec = np.reshape(Gray_2d2_qrt, ((corono0.nPup//2)**2))
    pup     = (Pupil_vec > 0.)
    bbb     = np.arange((corono0.nPup//2)**2)
    
    idx_pup = list(bbb[pup])
    npp     = len(idx_pup)
    
    if problem_name == 'MaxTau':
        if corono0.Pupil2dSym == False:
            PupilLyotStop_vec = np.reshape(Pupil2d*LyotStop2d, ((corono0.nPup)**2))
        else:
            PupilLyotStop_vec = np.reshape(Pupil2d_qrt*LyotStop2d_qrt, ((corono0.nPup//2)**2))

#%%
    neps = 0
    if problem_name == 'MaxContrastLinf':
        neps = 1
    elif problem_name == 'MaxContrastL1':
        neps = ndz*1   
        
    #%%
    TR = np.sum(Pupil_vec)
    
    nI1 = 1 
    # if ImPart is True:
    #     nI1 = 2
    nPsiD = nI1*corono0.nlam*ndz
    
    t1 = time.time()
    print(f'variable definition time: {t1-t0:.3f}s\n')
    
    #%%
    """
    ### Computation of the matrix A
    """
    print('computing matrix A')
    t00=time.time() 
    PsiD_re = np.zeros((npp, nPsiD))
    PsiD_im = 0
               
    # Compute coronagraph response matrix
    t0 = time.time()
#    if kwd_qrt:
    if ImPart:
        PsiD_im = np.zeros((npp, nPsiD))
        PsiD_re[0:npp,0:nPsiD], PsiD_im[0:npp,0:nPsiD] = compute_response_matrices_qrt(idx_pup, idx_dz, npp, ndz, corono0)
    
        if k != 0:
            PsiD0 = (1./4)*compute_corono_field_2d_LSasym(Ones_2d2_qrt, Pupil2d2_qrt, LyotStop2d, corono0)
            PsiD0 = np.reshape(PsiD0, (corono0.nlam, corono0.nImg2d**2))
            
    else:        
        PsiD_re[0:npp,0:nPsiD] = compute_response_matrices_qrt(idx_pup, idx_dz, npp, ndz,corono0) 
        if k != 0:
            PsiD0 = (1./4)*compute_corono_field_2d_qrt(Ones_2d2_qrt, Pupil2d2_qrt, LyotStop2d2_qrt, corono0)
            PsiD0 = np.reshape(PsiD0, (corono0.nlam, (corono0.nImg2d//2)**2))     

    if LSRobustness:
        
        print('LS constraints')
        PsiD_grad_x_re = np.empty_like(PsiD_im) 
        PsiD_grad_x_im = np.empty_like(PsiD_re) 
        PsiD_grad_y_re = np.empty_like(PsiD_im) 
        PsiD_grad_y_im = np.empty_like(PsiD_re) 
        if ImPart:
            PsiD_grad_x_re[0:npp,0:nPsiD], PsiD_grad_x_im[0:npp,0:nPsiD] = compute_response_matrices_qrt_LSrobustness_bis(idx_pup, idx_dz, npp, ndz, 'x', corono0)
            PsiD_grad_y_re[0:npp,0:nPsiD], PsiD_grad_y_im[0:npp,0:nPsiD] = compute_response_matrices_qrt_LSrobustness_bis(idx_pup, idx_dz, npp, ndz, 'y', corono0)
        else:
            PsiD_grad_x_re[0:npp,0:nPsiD] = compute_response_matrices_qrt_LSrobustness_bis(idx_pup, idx_dz, npp, ndz, 'x', corono0)
            PsiD_grad_y_re[0:npp,0:nPsiD] = compute_response_matrices_qrt_LSrobustness_bis(idx_pup, idx_dz, npp, ndz, 'y', corono0)

    
    PsiD0bis = 0
    PsiD0bis_t = np.zeros((ncorono))
    if k != 0:
        PsiD0bis = np.reshape(PsiD0[:, idx_dz], (nPsiD))
     
    t1= time.time()
    print(f'response matrix computation time: {t1-t0:.3f}s\n')
    
    #%%
    """
    ### Dimensions
    """
    print(f'dimensions PsiD_re: {np.shape(PsiD_re)}')
    print(f'dimensions PsiD_im: {np.shape(PsiD_im)}')
    print(f'PsiD axis-0: {np.shape(PsiD_re)[0]}')
    if LSRobustness:
        print(f'dimensions PsiD_grad_x_re: {np.shape(PsiD_grad_x_re)}')
        print(f'dimensions PsiD_grad_x_im: {np.shape(PsiD_grad_x_im)}')
        print(f'dimensions PsiD_grad_y_re: {np.shape(PsiD_grad_y_re)}')
        print(f'dimensions PsiD_grad_y_im: {np.shape(PsiD_grad_y_im)}')
    if ImPart:
        print(f'PsiD axis-1: {np.shape(PsiD_im)[1]}\n')
    
    print(f'ncorono: {ncorono}')
    print(f'nlam: {nlam}')
    print(f'npp: {npp}')
    print(f'ndz: {ndz}')
    print(f'neps: {neps}\n')
    
    
    #%%
    """
    ### Computation of the gurobi model
    """
    print('generating gurobi model')
    t00=time.time()
    if problem_name == 'MaxContrastLinf' or problem_name == 'MaxContrastL1':
        
        # Create a new model  
        model = gb.Model("LP max C new")
        
        # Create variables
        Apo = model.addMVar(npp, lb=0.0, ub=1.0, name="Apo")
        if problem_name == 'MaxContrastLinf':
            Eps = model.addMVar(neps, lb=0.0, name="Eps")
        else:
            Eps = model.addMVar(neps*corono0.nlam, lb=0.0, name="Eps")
            
        # Set objective
        model.setObjective(Eps.sum(), gb.GRB.MINIMIZE)
        
        # Add constraint:
        if ImPart:    
            model.addConstr( (PsiD_re + PsiD_im).T @ Apo + (PsiD0bis.real + PsiD0bis.imag) - Eps <= 0)
            model.addConstr( (PsiD_re - PsiD_im).T @ Apo + (PsiD0bis.real - PsiD0bis.imag) - Eps <= 0)
            model.addConstr((-PsiD_re + PsiD_im).T @ Apo +(-PsiD0bis.real + PsiD0bis.imag) - Eps <= 0)
            model.addConstr((-PsiD_re - PsiD_im).T @ Apo +(-PsiD0bis.real - PsiD0bis.imag) - Eps <= 0)

        else:
            model.addConstr( PsiD_re.T @ Apo + PsiD0bis.real - Eps <= 0)
            model.addConstr(-PsiD_re.T @ Apo - PsiD0bis.real - Eps <= 0)

        
        model.addConstr(-Apo.sum() - ApoCst  <= -tau*TR)
        
        # Add contraint of robustness to Lyot Stop misalignment
        if LSRobustness:
            model.addConstr( (PsiD_grad_x_re + PsiD_grad_x_im).T @ Apo + (PsiD0bis.real + PsiD0bis.imag) - Eps/(shift_tot/corono0.nPup) <= 0)
            model.addConstr( (PsiD_grad_x_re - PsiD_grad_x_im).T @ Apo + (PsiD0bis.real - PsiD0bis.imag) - Eps/(shift_tot/corono0.nPup) <= 0)
            model.addConstr((-PsiD_grad_x_re + PsiD_grad_x_im).T @ Apo +(-PsiD0bis.real + PsiD0bis.imag) - Eps/(shift_tot/corono0.nPup) <= 0)
            model.addConstr((-PsiD_grad_x_re - PsiD_grad_x_im).T @ Apo +(-PsiD0bis.real - PsiD0bis.imag) - Eps/(shift_tot/corono0.nPup) <= 0)

            model.addConstr( (PsiD_grad_y_re + PsiD_grad_y_im).T @ Apo + (PsiD0bis.real + PsiD0bis.imag) - Eps/(shift_tot/corono0.nPup) <= 0)
            model.addConstr( (PsiD_grad_y_re - PsiD_grad_y_im).T @ Apo + (PsiD0bis.real - PsiD0bis.imag) - Eps/(shift_tot/corono0.nPup) <= 0)
            model.addConstr((-PsiD_grad_y_re + PsiD_grad_y_im).T @ Apo +(-PsiD0bis.real + PsiD0bis.imag) - Eps/(shift_tot/corono0.nPup) <= 0)
            model.addConstr((-PsiD_grad_y_re - PsiD_grad_y_im).T @ Apo +(-PsiD0bis.real - PsiD0bis.imag) - Eps/(shift_tot/corono0.nPup) <= 0)

         
    else:
        # Create a new model  
        model = gb.Model("LP max tau new")
        
        # Create variables
        Apo = model.addMVar(npp, lb=0.0, ub=1.0, name="Apo")
            
        # Set objective
        model.setObjective(-Apo.sum() - ApoCst, gb.GRB.MINIMIZE)
        
        cst = (10.**(-cDarkHole/2.)/np.sqrt(2.))*corono0.Fmax2d/(corono0.nImg2d*corono0.nPup)
        Psi0 = cst*np.sum(PupilLyotStop_vec)
        
        # Add constraint:
        if ImPart:
            model.addConstr( (PsiD_re + PsiD_im).T @ Apo - Psi0 <= 0)
            model.addConstr( (PsiD_re - PsiD_im).T @ Apo - Psi0 <= 0)
            model.addConstr((-PsiD_re + PsiD_im).T @ Apo - Psi0 <= 0)
            model.addConstr((-PsiD_re - PsiD_im).T @ Apo - Psi0 <= 0)    
        else:
            model.addConstr( PsiD_re.T @ Apo - Psi0 <= 0)
            model.addConstr(-PsiD_re.T @ Apo - Psi0 <= 0)
        
        # Add contraint of robustness to Lyot Stop misalignment
        if LSRobustness:
            model.addConstr( (PsiD_grad_x_re + PsiD_grad_x_im).T @ Apo - Psi0/(shift_tot/corono0.nPup) <= 0)
            model.addConstr( (PsiD_grad_x_re - PsiD_grad_x_im).T @ Apo - Psi0/(shift_tot/corono0.nPup) <= 0)
            model.addConstr((-PsiD_grad_x_re + PsiD_grad_x_im).T @ Apo - Psi0/(shift_tot/corono0.nPup) <= 0)
            model.addConstr((-PsiD_grad_x_re - PsiD_grad_x_im).T @ Apo - Psi0/(shift_tot/corono0.nPup) <= 0)    

            model.addConstr( (PsiD_grad_y_re + PsiD_grad_y_im).T @ Apo - Psi0/(shift_tot/corono0.nPup) <= 0)
            model.addConstr( (PsiD_grad_y_re - PsiD_grad_y_im).T @ Apo - Psi0/(shift_tot/corono0.nPup) <= 0)
            model.addConstr((-PsiD_grad_y_re + PsiD_grad_y_im).T @ Apo - Psi0/(shift_tot/corono0.nPup) <= 0)
            model.addConstr((-PsiD_grad_y_re - PsiD_grad_y_im).T @ Apo - Psi0/(shift_tot/corono0.nPup) <= 0)    
            
            
    # Update model
    model.update()
        
    t11=time.time()
    print(f'gurobi model computation time: {t11-t00:.3f}s\n')
        
    
    #%%
    """
    ### Solving of the model
    """
    print('solving problem with gurobipy package')
    
    
    # solve problem with gurobipy package
    t0 = time.time()
    try:               
        model.Params.Method       = slvMethod
        model.Params.LogToConsole = slvLogToConsole
        model.Params.Crossover    = slvCrossover
            
        print('preparing to save optimization problem')
        print('ok')
        
        model.optimize()
        
    except gb.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
    
    t1 = time.time()
    print(f'optimization time             : {t1-t0:.2f}s\n')
    
    
    #%% Display of the apodizer
    """
    ### Generation of full apodizer for quarter pupil optimization
    """
    print('generation of the final apodizer')
    t0 = time.time()
    
    #if kwd_qrt:
    Apod1 = np.zeros(((corono0.nPup//2)**2))
    # else:
    #     Apod1 = np.zeros((corono0.nPup**2))
    Apod1[idx_pup] = Apo.x
    
#    if kwd_qrt:
    Apod1_2d = np.zeros((corono0.nPup, corono0.nPup))
    
    Apod1_2dtmp =  np.reshape(Apod1, (corono0.nPup//2, corono0.nPup//2))
    Apod1_2d[corono0.nPup//2:, corono0.nPup//2:] = Apod1_2dtmp
    Apod1_2d[:corono0.nPup//2, corono0.nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
    Apod1_2d[:, :corono0.nPup//2]          = np.flip(Apod1_2d[:, corono0.nPup//2:], axis=1)
            
    if k != 0:
        Apod1_2d += Ones_2d2
    Apod1_2d *= Pupil2d
            
    t1 = time.time()
    print(f'apodizer generation time             : {t1-t0:.3f}s\n')
    
    #%%
    nzp0 = (np.abs(Apod1_2d) <= 1e-2)
    nzp1 = (np.abs(Apod1_2d -1) <= 1e-2)
    
    Gray_2d = np.ones((nDim0*2**k, nDim0*2**k))
    Gray_2d[nzp0] = 0.
    Gray_2d[nzp1] = 0.
    
    Ones_2d = np.zeros((nDim0*2**k, nDim0*2**k))
    Ones_2d[np.abs(Apod1_2d) > 0.99] = 1 

    #%%
    """
    Save apodizer
    """
    # fdir_sav = Path('../../../results/2D/dat_pyth').resolve() / pupil_name
    # fdir_sav = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
    if not os.path.exists(fdir_sav):
        os.makedirs(fdir_sav)
        
    fname_sav = get_filename(corono0) + f'{str_dead_act}' + str_LSRcoeff_v9 + str_red_factor + '.fits'
    fpath_sav = fdir_sav / fname_sav
    
    if do_fits is True:
        fits.writeto(fpath_sav, Apod1_2d, overwrite=True)


#%%
plt.figure(1)
plt.clf()
plt.subplot(121)
plt.imshow(Pupil2d, vmin=0, vmax=1)
plt.subplot(122)
plt.imshow(Apod1_2d, vmin=0, vmax=1)
        

#%%
"""
### Edge detection algorithm (canny?)
"""
# t0 = time.time()
# Apod1_2dCopy = np.uint8(Apod1_2d*255)
# Apod1_2d_edges = cv2.Canny(Apod1_2dCopy, threshold1=0, threshold2=255)
# t1 = time.time()
# print(f'edge detection algorithm computation time: {t1-t0:.3f}s')

# kernel = np.ones((3, 3), np.uint8)
# Apod1_2d_edges_dilation = cv2.dilate(Apod1_2d_edges, kernel, iterations=1)

#%%
# plt.figure(20)
# plt.clf()
# plt.subplot(131)
# plt.imshow(Apod1_2d, cmap='gray')
# plt.title('original apodizer')
# plt.subplot(132)
# plt.imshow(Apod1_2d_edges, cmap='gray')
# plt.title('edge apodizer')
# plt.subplot(133)
# plt.imshow(Apod1_2d_edges_dilation, cmap='gray')
# plt.title('edge apodizer - dilation')

#%%
plt.figure(30)
plt.clf()
plt.imshow(Gray_2d)

#%%
# nImg2d = 90
# Fmax2d = 45

# PsiD1 = np.zeros((nImg2d, nImg2d), dtype='complex128')
# PsiD1 = coro.utils.sft(LyotStop2d, nImg2d, Fmax2d, CtrBtwnPix=True)

# PsiD2 = np.zeros((nImg2d, nImg2d), dtype='complex128')
# PsiD2[:nImg2d//2, :] = sft_hlf(LyotStop2d, nImg2d, Fmax2d, CtrBtwnPix=True)
# PsiD2[nImg2d//2:, :] = np.conj(np.fliplr(np.flipud(PsiD2[:nImg2d//2, :])))

# #%%
# plt.figure(40)
# plt.clf()
# plt.subplot(231)
# plt.imshow(PsiD1.real)
# plt.subplot(232)
# plt.imshow(PsiD2.real)
# plt.subplot(233)
# plt.imshow(PsiD1.real-PsiD2.real)
# plt.subplot(234)
# plt.imshow(PsiD1.imag)
# plt.subplot(235)
# plt.imshow(PsiD2.imag)
# plt.subplot(236)
# plt.imshow(PsiD1.imag-PsiD2.imag)
