# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:55:41 2024

@author: asp
"""

import numpy as np
import slow_fourier_transform as sft

def ncpa(ncpa_rms, nOPD, nPup, Pupil, pwr):
    
    ncpa_d = np.zeros((nOPD,nPup,nPup))
    
    ncpa_d += 1.

    # image dimensions, even, twice the pupil size, even too...    
    nMap = nPup * 2  #  same as Pupil.shape[0] * 2
    
    # 2D frequency space
    kx = (np.arange(nMap)-nMap//2)/(nMap/2)
    ky = (np.arange(nMap)-nMap//2)/(nMap/2)
    kx2, ky2 = np.meshgrid(kx, ky)
    k = np.sqrt(kx2**2 + ky2**2)
    
    # DSP law in f^pwr / add epsilon to avoid zero division
    pwr*=-1
    epsilon = 1e-15
    k_pwr = k**pwr
    dsp = 1./(np.where(k_pwr!=0,k_pwr,epsilon))
    amplitude = np.sqrt(dsp)

    # uniform random phase generation then complex valued field
    # amplitude.shape = tuple : *amplitude.shape tuple elements...
    random = np.random.uniform(low=-0.5,high=0.5,size=amplitude.shape)
    ncpa_field = amplitude * np.exp(1j*2.*np.pi*random)
    ncpa = np.real(sft.isft(ncpa_field,nMap,nMap//2))
    
    N = nMap//2
    hlf = N//2
    rnd=np.random.randn(nOPD)
    rnd /= 2.
    xi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)
    rnd = rnd[::-1]
    yi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)
    ncpa_d *= Pupil[None,:,:].copy()
    iok = np.nonzero(Pupil.copy())
    
    for n in np.arange(nOPD):
        
        temp = ((ncpa[hlf+xi[n]:hlf+N+xi[n],
                               hlf+yi[n]:hlf+N+yi[n]]).copy())
        ncpa_d[n,:,:] *= temp
        ncpa_d[n,:,:] -= np.mean(ncpa_d[n,:,:][iok])
        ncpa_d[n,:,:] *= ncpa_rms/np.std(ncpa_d[n,:,:][iok])

    return ncpa_d