# ============================================================================================
# utils_corono.py
# ============================================================================================
import numpy as np
import time
import os
from pathlib import Path
import sys
sys.path.insert(0, "/user/home/fromentp/Code")
import corono as coro

# --- FROZEN GUROBI PARAMETERS (DO NOT MODIFY) ---
GUROBI_FIXED_PARAMS = {
    "slvLogToConsole": 1,
    "slvCrossover": 0,
    "slvMethod": 2,
    "slvSparse": 0,
    "allLogToConsole": 1,
    "MinIsland": False,
    "Binarity": False,
    "FirstDerGlobalLim": 100.,
    "BinarityReg": 0.1,
    "nProgRef" : 1,
    "CtrBtwnPix": True,
    "CtrBtwnPix2": True,
    "Pupil2dSym": True,
    "ImPart": True,
}

# --------------------------------------------------------------------------------------------
# 1. FONCTIONS DE TRANSFORMEE DE FOURIER SEMI-ANALYTIQUE (SFT)
# --------------------------------------------------------------------------------------------
def sft_qrt(A2, NB, m, inv=False, CtrBtwnPix=False):
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    NA    = 2*np.shape(A2)[0]
    coeff = (m)/(NA*NB)
    
    U = np.zeros((1, NB // 2))
    X = np.zeros((1, NA // 2))
    
    X[0, :] = (1. / NA) * (np.arange(NA // 2) + val)
    U[0, :] = (m / NB) * (np.arange(NB // 2) + val)
       
    XU = 2. * np.pi * X.T.dot(U)
    A3 = np.cos(XU)
    A1 = A3.T
    
    B = A1.dot(A2.dot(A3))
    return 4 * coeff * B

def isft_qrt(A2, NB, m, CtrBtwnPix=False):
    return sft_qrt(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

def sft_hlf(A2, NB, m, inv=False, CtrBtwnPix=False):
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    
    sign = -1.0
    if inv:
        sign = 1.0

    U = np.zeros((1, NB))
    X = np.zeros((1, NA))
    
    X[0, :] = (1. / NA) * (np.arange(NA) - NA / 2. + val)    
    U[0, :] = (m / NB) * (np.arange(NB) - NB / 2. + val)
       
    XU = 2. * np.pi * X.T.dot(U)
    YV = 2. * np.pi * X.T.dot(U[:, :NB // 2])
    
    A3 = sign * 1j * np.sin(XU) + np.cos(XU)
    A1 = (sign * 1j * np.sin(YV) + np.cos(YV)).T
    
    B = A1.dot(A2.dot(A3))
    return coeff * B

def isft_hlf(A2, NB, m, CtrBtwnPix=False):
    return sft_hlf(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

def sft_full(A2, NB, m, inv=False, CtrBtwnPix=False):
    val = 0
    if CtrBtwnPix is True :
        val=1/2
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    sign = -1.0
    if inv:
        sign = 1.0

    U = np.zeros((1, NB))
    X = np.zeros((1, NA))
    X[0, :] = (1. / NA) * (np.arange(NA) - NA / 2. + val)    
    U[0, :] = (m / NB) * (np.arange(NB) - NB / 2. + val)

    XU = 2. * np.pi * X.T.dot(U)
    
    kernel = np.exp(sign * 1j * XU)
    B = kernel.T.dot(A2.dot(kernel))    
    return coeff * B

def isft_full(A2, NB, m, CtrBtwnPix=False):
    return sft_full(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

# --------------------------------------------------------------------------------------------
# 2. CALCULS CHAMPS ELECTRIQUES PROPAGES (PLANS A -> B -> C -> L -> D)
# --------------------------------------------------------------------------------------------
def compute_corono_field_2d_qrt(Apod2d_qrt, Pupil2d_qrt, LyotStop2d_qrt, mask2d_qrt, dtype0, corono=None):
    if corono is None:
        pass
    else :
        field_A_qrt = Apod2d_qrt * Pupil2d_qrt
        field_Dtmp_qrt = np.zeros((corono.nlam, corono.nImg2d // 2, corono.nImg2d // 2), dtype=dtype0)

        for i in range(corono.nlam):                       
            field_B_qrt = mask2d_qrt * sft_qrt(field_A_qrt, corono.nFPM, corono.mB_t[i], CtrBtwnPix=True)
            field_C_qrt = field_A_qrt - isft_qrt(field_B_qrt, corono.nPup, corono.mB_t[i], CtrBtwnPix=True)
            field_L_qrt = field_C_qrt * LyotStop2d_qrt
            field_Dtmp_qrt[i] = sft_qrt(field_L_qrt, corono.nImg2d, corono.mD_t[i], CtrBtwnPix=True)
                         
        return field_Dtmp_qrt

def compute_corono_field_2d_full(Apod2d, Pupil2d, LyotStop2d, mask2d, dtype0, corono=None):
    if corono is None:
        pass
    else :
        field_A = Apod2d * Pupil2d
        field_Dtmp = np.zeros((corono.nlam, corono.nImg2d, corono.nImg2d), dtype=dtype0)
        for i in range(corono.nlam):
            field_B = mask2d * sft_full(field_A, corono.nFPM, corono.mB_t[i], CtrBtwnPix=True)
            field_C = field_A - isft_full(field_B, corono.nPup, corono.mB_t[i], CtrBtwnPix=True)
            field_L = field_C * LyotStop2d
            field_Dtmp[i] = sft_full(field_L, corono.nImg2d, corono.mD_t[i], CtrBtwnPix=True)
                         
        return field_Dtmp


# --------------------------------------------------------------------------------------------
# 3. FONCTIONS GENERATION MATRICES DE REPONSE
# --------------------------------------------------------------------------------------------
def compute_response_matrices(idx_pup, idx_dz, npp, ndz, ImPart, Pupil2d, mask2d, dtype0, corono=None):
    if corono is None:
        return None, None
        
    corono_field_re_t_tmp = np.empty((npp, corono.nlam, corono.nImg2d**2))
    corono_field_im_t = None
    Apod2d = np.zeros((corono.nPup, corono.nPup))

    if ImPart:
        corono_field_im_t_tmp = np.empty((npp, corono.nlam, corono.nImg2d**2))

        for i, val in enumerate(idx_pup):
            (i0, j0) = np.unravel_index(val, (corono.nPup, corono.nPup))
            Apod2d[i0, j0] = 1            
            field_complex = compute_corono_field_2d_full(Apod2d, Pupil2d, corono.LyotStop2d, mask2d, dtype0, corono)
            corono_field_re_t_tmp[i] = field_complex.real.flatten()
            corono_field_im_t_tmp[i] = field_complex.imag.flatten()
            Apod2d[i0, j0] = 0 
            
        corono_field_re_t = np.reshape(corono_field_re_t_tmp[:, :, idx_dz], (npp, corono.nlam * ndz))
        corono_field_im_t = np.reshape(corono_field_im_t_tmp[:, :, idx_dz], (npp, corono.nlam * ndz))
        
        del corono_field_re_t_tmp, corono_field_im_t_tmp
    
    else:    
        for i, val in enumerate(idx_pup):
            (i0, j0) = np.unravel_index(val, (corono.nPup, corono.nPup))
            Apod2d[i0, j0] = 1            
            field_complex = compute_corono_field_2d_full(Apod2d, Pupil2d, corono.LyotStop2d, mask2d, dtype0, corono)
            corono_field_re_t_tmp[i] = field_complex.real.flatten()
            Apod2d[i0, j0] = 0 
            
        corono_field_re_t = np.reshape(corono_field_re_t_tmp[:, :, idx_dz], (npp, corono.nlam * ndz))
        del corono_field_re_t_tmp

    return corono_field_re_t, corono_field_im_t

def compute_response_matrices_qrt(idx_pup, idx_dz, npp, ndz, ImPart, Pupil2d_qrt, mask2d_qrt, dtype0, corono=None):
    if corono is None:
        pass
    else:
        Apod2d_qrt = np.zeros((corono.nPup // 2, corono.nPup // 2))        

        if ImPart is True:
            corono_field_re_t_tmp = np.empty((npp, corono.nlam, (corono.nImg2d // 2)**2))            
            corono_field_im_t_tmp = np.empty((npp, corono.nlam, (corono.nImg2d // 2)**2))
            

            LyotStop2d_qrt = corono.LyotStop2d[corono.nPup // 2:, corono.nPup // 2:]

            for i, val in enumerate(idx_pup):
                (i0, j0) = np.unravel_index(val, (corono.nPup // 2, corono.nPup // 2))
                Apod2d_qrt[i0, j0] = 1            
                
                test_qrt = (1. / 4) * compute_corono_field_2d_qrt(Apod2d_qrt, Pupil2d_qrt, LyotStop2d_qrt, mask2d_qrt, dtype0, corono)
                
                corono_field_re_t_tmp[i] = np.reshape(test_qrt.real, (corono.nlam, (corono.nImg2d // 2)**2))
                corono_field_im_t_tmp[i] = np.reshape(test_qrt.imag, (corono.nlam, (corono.nImg2d // 2)**2))
                
                Apod2d_qrt[i0, j0] = 0 
            
            corono_field_re_t = np.reshape(corono_field_re_t_tmp[:, :, idx_dz], (npp, corono.nlam * ndz))
            corono_field_re_t_tmp = None
            del corono_field_re_t_tmp
    
            corono_field_im_t = np.reshape(corono_field_im_t_tmp[:, :, idx_dz], (npp, corono.nlam * ndz))
            corono_field_im_t_tmp = None
            del corono_field_im_t_tmp
        
            return corono_field_re_t, corono_field_im_t

        else:
            corono_field_re_t_tmp = np.empty((npp, corono.nlam, (corono.nImg2d // 2)**2))
            LyotStop2d_qrt = corono.LyotStop2d[corono.nPup // 2:, corono.nPup // 2:]
        
            for i, val in enumerate(idx_pup):
                (i0, j0) = np.unravel_index(val, (corono.nPup // 2, corono.nPup // 2))
                Apod2d_qrt[i0, j0] = 1
                test_qrt = (1. / 4) * compute_corono_field_2d_qrt(Apod2d_qrt, Pupil2d_qrt, LyotStop2d_qrt, mask2d_qrt, dtype0, corono)
                corono_field_re_t_tmp[i] = np.reshape(test_qrt, (corono.nlam, (corono.nImg2d // 2)**2))
                Apod2d_qrt[i0, j0] = 0 
            
            corono_field_re_t = np.reshape(corono_field_re_t_tmp[:, :, idx_dz], (npp, corono.nlam * ndz))
            corono_field_re_t_tmp = None
            del corono_field_re_t_tmp

            return corono_field_re_t