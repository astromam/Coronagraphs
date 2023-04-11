#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:44:36 2023

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

#%% parameters
"""
Parameters
"""
# Telescope name
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxContrastLinf' # 'MaxTau' # ,'MaxContrastLinf' # 'MaxContrastL1' #
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
nPup = 100
nFPM = 50
Fmax2d = 45#22.5
nImg2d = 90#45

# mask radius in lam0/D units
rMask = 2.8

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  5.0
rho1 = 20.0

# contrast in the dark region
cDarkHole = 7.0

# tau (integrated Pupil transmission)
tau   = 0.5

# CtrBtwnPix2
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True
ImPart = False 
LSRobustness = False

#nlam
bw   = 0.1
nlam = 1

do_fits = True

kwd_qrt = True

#%%
"""
File reading for Pupil and Lyot stop
"""
#fdir = Path('../../../data/2D/pupils/').resolve()
fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('~/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').expanduser()
        sim_case = 'test' # 'test' or 'server'
    elif syst == 'linux':
        fdir = Path('/home/mndiaye/python/Coronagraphs/data/2D/pupils').resolve()
        sim_case = 'server' # 'test' or 'server'            
    else:
        raise ValueError('Unknown operating system {0}'.format(user))
else:
    raise ValueError('Unknown user {0}'.format(user))


if pupil_name == 'lvr':
    fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
    fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
else:
    fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)

fpath_pup = fdir / fname_pup
fpath_lys = fdir / fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)

Pupil2d_qrt = Pupil2d[nPup//2:,nPup//2:]
LyotStop2d_qrt = LyotStop2d[nPup//2:,nPup//2:]


if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
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
                 ImPart = ImPart)

#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params)
elif corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))
    
#%%
"""
### Functions
"""
dtype0 = 'float64'
if ImPart is True:
    dtype0 = 'complex128'

# wavelengths
lam0 = 1.        
dlam = bw*lam0
lam_t = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam)
    
# Focal plane mask
mask2d = coro.utils.uniform_disk(nFPM, nFPM/2., CtrBtwnPix=CtrBtwnPix)
mask2d_qrt = mask2d[nFPM//2:,nFPM//2:]

# mask size at a given wavelength for SFT
mB_t  = 2.*rMask*(lam0/lam_t)
mD_t  = Fmax2d*(lam0/lam_t)


#%%

def compute_corono_field_2d(Apod2d, Pupil2d, mask2d, mB_t, LyotStop2d):
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
    field_A    = Apod2d*Pupil2d
                
    field_Dtmp = np.zeros((nlam,nImg2d,nImg2d), dtype=dtype0)


    for i in range(nlam):
        field = field_A[i]               
        if ImPart is True:                
            field_B       = mask2d*coro.utils.sft(field, nFPM, mB_t[i], 
                                            CtrBtwnPix=CtrBtwnPix)
            field_C       = field - coro.utils.isft(field_B, nPup, mB_t[i], 
                                           CtrBtwnPix=CtrBtwnPix)
            field_L       = field_C*LyotStop2d
            field_Dtmp[i] = coro.utils.sft(field_L, nImg2d, mD_t[i], 
                      CtrBtwnPix=CtrBtwnPix2)
            
        else:
            field_B       = mask2d*coro.utils.sft_even(field, nFPM, mB_t[i], 
                                            CtrBtwnPix=CtrBtwnPix)
            field_C       = field - coro.utils.isft_even(field_B, nPup, mB_t[i], 
                                           CtrBtwnPix=CtrBtwnPix)
            field_L       = field_C*LyotStop2d
            field_Dtmp[i] = coro.utils.sft_even(field_L, nImg2d, mD_t[i], 
                      CtrBtwnPix=CtrBtwnPix2)
                     
    # return (lam0/lam_t[:,None,None])*field_Dtmp (in mathematica)
    return field_Dtmp

#%%       
#    @profile
def compute_response_matrices(corono=None):
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

def compute_corono_field_2d_qrt(Apod2d_qrt):
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
    field_A_qrt    = Apod2d_qrt*Pupil2d_qrt
                
    field_Dtmp_qrt = np.zeros((nlam,nImg2d//2,nImg2d//2), dtype=dtype0)


    for i in range(nlam):                       
        field_B_qrt       = mask2d_qrt*sft_qrt(field_A_qrt, nFPM, mB_t[i], 
                                        CtrBtwnPix=True)
        field_C_qrt       = field_A_qrt - isft_qrt(field_B_qrt, nPup, mB_t[i], 
                                       CtrBtwnPix=True)
        field_L_qrt       = field_C_qrt*LyotStop2d_qrt
        field_Dtmp_qrt[i] = sft_qrt(field_L_qrt, nImg2d, mD_t[i], 
                  CtrBtwnPix=True)
                     
    # return (lam0/lam_t[:,None,None])*field_Dtmp (in mathematica)
    return field_Dtmp_qrt

#%%
def compute_response_matrices_qrt(npp,corono=None):
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

        if ImPart is True:
            corono_field_re_t_tmp = np.empty((npp, corono.nlam, 
                                                   corono.nImg2d**2))
            Apod2d = np.zeros((corono.nPup, corono.nPup))             
            corono_field_im_t_tmp = np.empty((npp, corono.nlam, 
                                               corono.nImg2d**2))

            for i, val in enumerate(idx_pup):
                (i0,j0) = np.unravel_index(val, (corono.nPup, corono.nPup))
                Apod2d[i0,j0] = 1            
                #corono_field_re_t_tmp[i], corono_field_im_t_tmp[i] = \
                #corono.compute_corono_field_2d_vec(Apod2d)
                test = corono.compute_corono_field_2d(Apod2d)
                corono_field_re_t_tmp[i] = np.reshape(test.real, (nlam, nImg2d**2))
                corono_field_im_t_tmp[i] = np.reshape(test.imag, (nlam, nImg2d**2))
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
            corono_field_re_t_tmp = np.empty((npp, corono.nlam, 
                                                   (corono.nImg2d//2)**2))
            Apod2d_qrt = np.zeros((corono.nPup//2, corono.nPup//2)) 
            for i, val in enumerate(idx_pup):
                (i0,j0) = np.unravel_index(val, (corono.nPup//2, corono.nPup//2))
                Apod2d_qrt[i0,j0] = 1
                test_qrt = (1./4)*compute_corono_field_2d_qrt(Apod2d_qrt)
                corono_field_re_t_tmp[i] = \
                np.reshape(test_qrt, (nlam, (nImg2d//2)**2))
                # corono.compute_corono_field_2d_real_vec(Apod2d)
                Apod2d_qrt[i0,j0] = 0 
                
            corono_field_re_t = np.reshape(
                    corono_field_re_t_tmp[:,:, idx_dz], 
                    (npp, corono.nlam*ndz))
            
            corono_field_re_t_tmp = None
            del corono_field_re_t_tmp

        return corono_field_re_t


#%%
def get_filename():
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
    params = corono0.params.copy()
    params = corono0.utils.update_params(params)
            
    if problem_name == 'MaxTau':
        str_opt = '_C={cDarkHole:.1f}'
    elif problem_name == 'MaxContrastL1' or problem_name == 'MaxContrastLinf':
        str_opt = '_tau={tau:.3f}'
    else:
        raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

    str_FirstDerGlobal = ''
    if MinIsland is True:
        str_FirstDerGlobal = '_1stderglo={FirstDerGlobalLim}'
        
    str_LSRobustness = ''
    if LSRobustness is True:
        str_LSRobustness = '_LSRobustness=1'

    fname_corono = corono0.get_filename()             
        
    fname_gen_optim  = '{problem_name}' \
    + str_opt + str_FirstDerGlobal + str_LSRobustness \
    + '_{solver}'        
    
    return '{pupil_name}_'.format(**params) + fname_corono + fname_gen_optim.format(**params)
  
#%%
"""
Problem definition without class
"""
print('definition of the variables for the problem')
t0 = time.time()

ncorono = 1
neps = 1 

dz2d, rad2d = corono0.generate_area()

#%%
if kwd_qrt:
    if corono0.Pupil2dSym == False:        
        dz      = np.reshape(dz2d, (corono0.nImg2d**2))
        aaa     = np.arange(corono0.nImg2d**2)
    else:
        dz2d_qrt = dz2d[nImg2d//2:,nImg2d//2:] 
        dz = np.reshape(dz2d_qrt, ((corono0.nImg2d//2)**2))
        aaa     = np.arange((corono0.nImg2d//2)**2)          
        
    idx_dz  = list(aaa[dz])  
    ndz     = len(idx_dz)
    
    if corono0.Pupil2dSym == False:
        Pupil_vec = np.reshape(corono0.Pupil2d, (corono0.nPup**2))
        pup     = (Pupil_vec > 0.)
        bbb     = np.arange(corono0.nPup**2)
    else:
        Pupil_vec = np.reshape(Pupil2d_qrt, ((corono0.nPup//2)**2))
        pup     = (Pupil_vec > 0.)
        bbb     = np.arange((corono0.nPup//2)**2)
    
    idx_pup = list(bbb[pup])
    npp     = len(idx_pup)
    
else:
    if corono0.Pupil2dSym == False:        
        dz      = np.reshape(dz2d, (corono0.nImg2d**2))
    else:
        Image2dquarter = np.zeros_like(dz2d)
        Image2dquarter[corono0.nImg2d//2:, corono0.nImg2d//2:] = 1.
        dz = np.reshape(dz2d*Image2dquarter, (corono0.nImg2d**2))           
        
    aaa     = np.arange(corono0.nImg2d**2)
    idx_dz  = list(aaa[dz])  
    ndz     = len(idx_dz)
    
    if corono0.Pupil2dSym == False:
        Pupil_vec = np.reshape(corono0.Pupil2d, (corono0.nPup**2))
    else:
        Pupil2dquarter = np.zeros_like(corono0.Pupil2d)
        Pupil2dquarter[corono0.nPup//2:, corono0.nPup//2:] = 1.
        Pupil_vec = np.reshape(corono0.Pupil2d*Pupil2dquarter, (corono0.nPup**2))
    
    pup     = (Pupil_vec > 0.)
    bbb     = np.arange(corono0.nPup**2)
    idx_pup = list(bbb[pup])
    npp     = len(idx_pup)
    
#%%
TR = np.sum(Pupil_vec)

nI1 = 1 
if ImPart is True:
    nI1 = 2
nPsiD = nI1*corono0.nlam*ndz

t1 = time.time()
print(f'variable definition time: {t1-t0:.3f}s\n')

#%%
"""
### Computation of the matrix A
"""
print('computing matrix A')
t00=time.time() 
PsiD = np.zeros((npp, nPsiD))

# Compute coronagraph response matrix
t0 = time.time()
if kwd_qrt:
    PsiD[0:npp,0:nPsiD] = compute_response_matrices_qrt(npp,corono0)
else:
    PsiD[0:npp,0:nPsiD] = compute_response_matrices(corono0) 
t1= time.time()
print(f'response matrix computation time: {t1-t0:.3f}s\n')

#%%
"""
### Dimensions
"""
print(f'dimensions PsiD: {np.shape(PsiD)}')
print(f'PsiD axis-0: {np.shape(PsiD)[0]}')
print(f'PsiD axis-1: {np.shape(PsiD)[1]}\n')

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
# Create a new model  
model = gb.Model("LP max C new")

# Create variables
Apo = model.addMVar(npp, lb=0.0, ub=1.0, name="Apo")
Eps = model.addMVar(neps, lb=0.0, name="Eps")  
    
# Set objective
model.setObjective(Eps.sum(), gb.GRB.MINIMIZE)

# Add constraint:
model.addConstr( PsiD.T @ Apo - Eps <= 0)
model.addConstr(-PsiD.T @ Apo - Eps <= 0)
model.addConstr(-Apo.sum()  <= -tau*TR)
    
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

    if kwd_qrt:
        Apod1 = np.zeros(((corono0.nPup//2)**2))
    else:
        Apod1 = np.zeros((corono0.nPup**2))
    Apod1[idx_pup] = Apo.x
    
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

if kwd_qrt:
    Apod1_2d = np.zeros((corono0.nPup, corono0.nPup))
    
    if Pupil2dSym == True:
            Apod1_2dtmp =  np.reshape(Apod1, (corono0.nPup//2, corono0.nPup//2))
            Apod1_2d[corono0.nPup//2:, corono0.nPup//2:] = Apod1_2dtmp
            Apod1_2d[:corono0.nPup//2, corono0.nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
            Apod1_2d[:, :corono0.nPup//2]          = np.flip(Apod1_2d[:, corono0.nPup//2:], axis=1)
    
else:
    Apod1_2d = np.reshape(Apod1, (corono0.nPup, corono0.nPup))
    
    if Pupil2dSym == True:
            Apod1_2dtmp =  Apod1_2d[corono0.nPup//2:, corono0.nPup//2:]
            Apod1_2d[:corono0.nPup//2, corono0.nPup//2:] = np.flip(Apod1_2dtmp, axis=0)
            Apod1_2d[:, :corono0.nPup//2]          = np.flip(Apod1_2d[:, corono0.nPup//2:], axis=1)
        
t1 = time.time()
print(f'apodizer generation time             : {t1-t0:.3f}s\n')

#%%
plt.figure(1)
plt.clf()
plt.subplot(121)
plt.imshow(Pupil2d, vmin=0, vmax=1)
plt.subplot(122)
plt.imshow(Apod1_2d, vmin=0, vmax=1)
        
#%%
"""
Save apodizer
"""
# fdir = Path('../../../results/2D/dat_pyth').resolve() / pupil_name
# #fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/results/2D/dat_pyth/').resolve() / pupil_name
# if not os.path.exists(fdir):
#     os.makedirs(fdir)
    
# fname = problem1.get_filename() + '.fits'
# fpath = fdir / fname

# if do_fits is True:
#     fits.writeto(fpath, Apod1_2d, overwrite=True)

#%%
"""
### Test sft quarter
"""
Psi_A = Pupil2d 
Psi_B = coro.utils.sft(Psi_A, nFPM, rMask*2, CtrBtwnPix=True)*mask2d 
Psi_C = Psi_A - coro.utils.isft(Psi_B, nPup, rMask*2, CtrBtwnPix=True)
Psi_L = Psi_C * LyotStop2d 
Psi_D = coro.utils.sft(Psi_L, nImg2d, Fmax2d, CtrBtwnPix=True)

Psi_Aq = Pupil2d[nPup//2:,nPup//2:]
Psi_Bq = sft_qrt(Psi_Aq, nFPM, rMask*2, CtrBtwnPix=True)*mask2d[nFPM//2:,nFPM//2:]
Psi_Cq = Psi_Aq - isft_qrt(Psi_Bq, nPup, rMask*2, CtrBtwnPix=True) 
Psi_Lq = Psi_Cq * LyotStop2d[nPup//2:,nPup//2:]
Psi_Dq = sft_qrt(Psi_Lq, nImg2d, Fmax2d, CtrBtwnPix=True)

#%%
plt.figure(2)
plt.clf()
plt.subplot(431)
plt.imshow(np.abs(Psi_A[nPup//2:,nPup//2:]))
plt.subplot(432)
plt.imshow(np.abs(Psi_Aq))
plt.subplot(433)
plt.imshow(np.abs(Psi_Aq)-np.abs(Psi_A[nPup//2:,nPup//2:]))
plt.subplot(434)
plt.imshow(Psi_B[nFPM//2:, nFPM//2:].real)
plt.subplot(435)
plt.imshow(Psi_Bq.real)
plt.subplot(436)
plt.imshow(Psi_Bq.real-Psi_B[nFPM//2:, nFPM//2:].real)
plt.subplot(437)
plt.imshow(Psi_C[nPup//2:, nPup//2:].real)
plt.subplot(438)
plt.imshow(Psi_Cq.real)
plt.subplot(439)
plt.imshow(Psi_Cq.real-Psi_C[nPup//2:, nPup//2:].real)
plt.subplot(4,3,10)
plt.imshow(Psi_D[nImg2d//2:, nImg2d//2:].real)
plt.subplot(4,3,11)
plt.imshow(Psi_Dq.real)
plt.subplot(4,3,12)
plt.imshow(Psi_Dq.real-Psi_D[nImg2d//2:, nImg2d//2:].real)
