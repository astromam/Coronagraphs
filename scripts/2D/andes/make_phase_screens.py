# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:07:41 2024

@author: asimmonnin, mndiaye, asp
"""
# create sets of phase screens files with given ncpa value

# %%

import numpy as np
import slow_fourier_transform as sft
from pathlib import Path
from astropy.io import fits


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_dat   = Path('D:/Andes/Data_corono/data/').resolve()  #  fits data


#%%

std_tgt = 10e-8 #☺ target std in meters
nb_ncpa = 4096

fdir_pupil = fdir_dat / 'Pupil'
fname_elt = 'ELT_pupil_400.fits' # New pupil with new spider
fpath_elt = fdir_pupil / fname_elt
pup = fpath_elt.stem
fnm = ('ncpa_'+pup+'_'+str(int(np.rint(std_tgt*1e9)))+"nm_"+
       str(int(nb_ncpa))+"screens.fits")

"""
### Read pupil file
"""
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)

# image dimensions, even, twice the pupil size
nMap = Pupil.shape[0] * 2

# 2D frequency space
kx = (np.arange(nMap)-nMap//2)/(nMap/2)
ky = (np.arange(nMap)-nMap//2)/(nMap/2)
kx2, ky2 = np.meshgrid(kx, ky)
k = np.sqrt(kx2**2 + ky2**2)

# DSP law in f^pwr
# add epsilon to avoid zero division
pwr=-2.
pwr*=-1
epsilon = 1e-15
k_pwr = k**pwr
dsp = 1./(np.where(k_pwr!=0,k_pwr,epsilon))

# uniform random phase generation
# complex valued spectra
amplitude = np.sqrt(dsp)
#amplitude.shape est un tuple et *amplitude.shape sont les éléments de ce tuple
random = np.random.uniform(low=-0.5,high=0.5,size=amplitude.shape)
ncpa_field = amplitude * np.exp(1j*2.*np.pi*random)

ncpaX2 = np.real(sft.isft(ncpa_field,nMap,nMap//2))

N=nMap//2
hlf=N//2
rnd=np.random.randn(nb_ncpa)
rnd /= 2.
xi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)
rnd=np.random.randn(nb_ncpa)
rnd /= 2.
yi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)
ncpa = np.ones((N,N,nb_ncpa))
ncpa *= Pupil[:,:,None].copy()
iok = np.nonzero(Pupil.copy())

for n in np.arange(nb_ncpa):
    
    temp = (ncpaX2[hlf+xi[n]:hlf+N+xi[n],hlf+yi[n]:hlf+N+yi[n]]).copy()
    ncpa[:,:,n] *= temp
    ncpa[:,:,n] -= np.mean(ncpa[:,:,n][iok])
    ncpa[:,:,n] *= std_tgt/np.std(ncpa[:,:,n][iok])

fits.writeto(
    (fdir_dat/fnm),
    np.transpose(ncpa,(2,0,1)), overwrite=True)


