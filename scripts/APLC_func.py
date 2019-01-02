import numpy as np
import time
import os
from pathlib import Path
from astropy.io import fits

import corono as coro

def coron_inputs():
	
	fdir = Path('input_files/').resolve()
	
	if pupil_name == 'lvr':
    	fname_pup = 'apertures/ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
    	fname_lys = 'lyot_stops/ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
	elif pupil_name == 'HiCAT':
		fname_pup = 'apertures/HiCAT-Aper_{0}-N0{1}_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004.fits'.format(sym,nPup//2,)
		fname_lys = 'lyot_stops/HiCAT-Lyot_{0}-N0{1}_LS-Ann-bw-ID0345-OD0807-SpX0036.fits'.format(sym,nPup//2,)
	else:
    	fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    	fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)

	fpath_pup = fdir / fname_pup
	fpath_lys = fdir / fname_lys
	Pupil2d    = fits.getdata(fpath_pup)
	LyotStop2d = fits.getdata(fpath_lys)
	
	return Pupil2d,LyotStop2d
	
def quarter_dat_apod_to_fits():
	
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
	
def quarter_pup_to_fits(npup,filename,directory):
	
	filepath = directory + filename + '.dat'
	Apod_ampl_quarter = np.loadtxt(filepath)
	
	Apod_ampl = np.zeros((nPup, nPup))
	Apod_ampl[nPup//2:, nPup//2:] = Apod_ampl_quarter
	Apod_ampl[:nPup//2, nPup//2:] = np.flip(Apod_ampl_quarter, axis=0)
	Apod_ampl[:,:nPup//2] = np.flip(Apod_ampl[:,nPup//2:], axis=1)

	fpath = directory + filename + '.fits'
	fits.writeto(fpath, Apod_ampl, overwrite=True)
	
