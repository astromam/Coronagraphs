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

elif user == 'Adrien':
    # Directory for the OPD with the corresponding seed value
    fdir_dat   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/data/').resolve()

elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    # Directory for the OPD with the corresponding seed value
    fdir_dat   = Path( fdir_base / 'data' ).resolve()


#%%

std_tgt = 5e-8 #☺ target std in meters

fdir_pupil = fdir_dat / 'Pupil'
fname_elt = 'ELT_pupil_400.fits' # New pupil with new spider
fpath_elt = fdir_pupil / fname_elt

"""
### Read pupil file
"""
# Read ELT pupil 
Pupil = fits.getdata(fpath_elt,)

# image dimensions, even, twice the pupil size
N = Pupil.shape[0]*2

# 2D frequency space
kx = (np.arange(N)-N//2)/(N/2)  # spatial frequencies
ky = (np.arange(N)-N//2)/(N/2)
kx2, ky2 = np.meshgrid(kx, ky)
k = np.sqrt(kx2**2 + ky2**2)  # spatial fréquencies norm vector

# DSP law in f^-pwr
# add epsilon to avoid zero division
pwr=-2.
epsilon = 1e-10
dsp = 1./(k**pwr + (k==0)*epsilon)

# uniform random phase generation
# complex valued spectra
amplitude = np.sqrt(dsp)
#amplitude.shape est un tuple et *amplitude.shape sont les éléments de ce tuple
random = np.random.uniform(low=-0.5,high=0.5,size=amplitude.shape)
ncpa_field = amplitude * np.exp(1j*2.*np.pi*random)

# Transformée de Fourier inverse pour obtenir l'image spatiale
rndOpd_powLaw = np.real(sft.isft(ncpa_field,N,N//2))

# boucle extraction aleatoire d'un masque random de dimension identique
# à la pupille

# N = pupil size now
N=N//2
hlf=N//2
nb_ncpa = 2048
rnd=np.random.randn(nb_ncpa)
xi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)
rnd=np.random.randn(nb_ncpa)
yi=np.round(rnd*hlf/np.max([-np.min(rnd),np.max(rnd)])).astype(int)

ncpa = np.ones((N,N,nb_ncpa))
ncpa *= Pupil[:,:,None].copy()
iok = np.nonzero(Pupil.copy())

for n in np.arange(nb_ncpa):
    
    temp = (rndOpd_powLaw[hlf+xi[n]:hlf+N+xi[n],hlf+yi[n]:hlf+N+yi[n]]).copy()
    ncpa[:,:,n] *= temp
    ncpa[:,:,n] -= np.mean(ncpa[:,:,n][iok])
    ncpa[:,:,n] *= std_tgt/np.std(ncpa[:,:,n][iok])

fits.writeto(
    (fdir_dat/('ncpa_pupil_new_'+str(int(np.rint(std_tgt*1e9)))+"nm.fits")),
    np.transpose(ncpa,(2,0,1)), overwrite=True)


