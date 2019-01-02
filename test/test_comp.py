#quick and dirty test to check sameness of .fits files

import numpy as np
import time
import os
#from pathlib import Path
import pylab as pl
from astropy.io import fits


nPup = 200

def quarter_dat_apod_to_fits(npup,directory,filename):
	
	filepath = directory + filename + '.dat'
	
	Apod_ampl_raw = np.loadtxt(filepath)
	Apod_ampl_quarter = np.reshape(Apod_ampl_raw[:, 2], (nPup//2, nPup//2))
	
	Apod_ampl = np.zeros((nPup, nPup))
	Apod_ampl[nPup//2:, nPup//2:] = Apod_ampl_quarter
	Apod_ampl[:nPup//2, nPup//2:] = np.flip(Apod_ampl_quarter, axis=0)
	Apod_ampl[:,:nPup//2] = np.flip(Apod_ampl[:,nPup//2:], axis=1)
	
	fpath = directory + filename + '.fits'
	fits.writeto(fpath, Apod_ampl, overwrite=True)
	
	return Apod_ampl

def dat_apod_to_fits(npup,directory,filename):
	
	filepath = directory + filename + '.dat'
	
	Apod_ampl_raw = np.loadtxt(filepath)
	Apod_ampl_full = np.reshape(Apod_ampl_raw[:, 2], (nPup, nPup))
	
	fpath = directory + filename + '.fits'
	fits.writeto(fpath, Apod_ampl_full, overwrite=True)
	
	return Apod_ampl_full


def quarter_pup_to_fits(npup,filename,directory):
	
	filepath = directory + filename + '.dat'
	Apod_ampl_quarter = np.loadtxt(filepath)
	
	Apod_ampl = np.zeros((nPup, nPup))
	Apod_ampl[nPup//2:, nPup//2:] = Apod_ampl_quarter
	Apod_ampl[:nPup//2, nPup//2:] = np.flip(Apod_ampl_quarter, axis=0)
	Apod_ampl[:,:nPup//2] = np.flip(Apod_ampl[:,nPup//2:], axis=1)

	fpath = directory + filename + '.fits'
	fits.writeto(fpath, Apod_ampl, overwrite=True)

directory = 'apodizers/HiCAT/'


full_name = 'Test_S_telserv3_HiCAT_MaxTau_nPup=0048_nFPM=050_APLC_rMask=4.271_IWA=5.0_OWA=10.0_BW=0.00_nlam=01_C=8.0_LS-Ann-bw-ID345-OD0807_gurobipysparse_A_float32'
stdgrbpath = directory + full_name + '.fits'
full = fits.getdata(stdgrbpath)

sparse_name = 'Test_S_telserv3_HiCAT_MaxTau_nPup=0048_nFPM=050_APLC_rMask=4.271_IWA=5.0_OWA=10.0_BW=0.00_nlam=01_C=8.0_LS-Ann-bw-ID345-OD0807_gurobipy_sparseA'
stdgrbpath = directory + sparse_name + '.fits'
sparse = fits.getdata(stdgrbpath)


test = sparse - full
filename = 'Test_S_HiCAT_sparse_minus_sparseFloat32_gurobipy'
fpath = directory + filename + '.fits'
fits.writeto(fpath, test, overwrite=True)



