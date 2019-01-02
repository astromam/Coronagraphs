#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

import numpy as np
import pylab as pl
from pathlib import Path
import os
from matplotlib import cm
from astropy.io import fits
import corono as coro

#%% parameters
"""
Parameters
"""
# Telescope name
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'HiCAT' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxTau' # 'MaxTau' # ,'MaxContrastLinf' # 'MaxContrastL1' #
solver       = 'stdgrb' #,'stdgrb' #  'gurobipy', 'scipy.linprog'
slvLogToConsole = 0
slvCrossover    = 0
slvMethod       = 2
allLogToConsole = 0
LSRobustness = True

MinIsland   = False
Binarity    = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1

#nPup = corono0.params['nPup']
nPup = 200
nFPM = 50
Fmax2d = 50
nImg2d = 500

# mask radius in lam0/D units
rMask = 8.543/2

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  5.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 8.0

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True

#nlam
bw   = 0.10
nlam = 3
nlambis = 11

pix_max   = 1

do_fits = True

def dat_input_to_fits(npup,directory,filename):
	
	filepath = directory + filename + '.dat'
	
	input_data_raw = np.loadtxt(filepath)
	input_data = np.reshape(input_data_raw, (nPup, nPup))
	
	fpath = directory + filename + '.fits'
	fits.writeto(fpath, input_data, overwrite=True)
	
	return input_data

#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = 'input_files/'
if pupil_name == 'lvr':
    fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
    fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
elif pupil_name == 'HiCAT':
	fname_pup = 'apertures/HiCAT-Aper_F-N0{0}_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004'.format(nPup,)
	fname_lys = 'lyot_stops/HiCAT-Lyot_F-N0{0}_LS-Ann-bw-ID0345-OD0807-SpX0036_shiftX+000'.format(nPup,)
else:
    fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)

#fpath_pup = fdir / fname_pup
#fpath_lys = fdir / fname_lys
#Pupil2d    = fits.getdata(fpath_pup)
#LyotStop2d = fits.getdata(fpath_lys)

Pupil2d = dat_input_to_fits(nPup,fdir,fname_pup)
LyotStop2d = dat_input_to_fits(nPup,fdir,fname_lys)

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
                 LSRobustness = LSRobustness, pix_max = pix_max)

#%%
"""
Working directories
"""
fdir = Path('apodizers/HiCAT').resolve()

fdir_pdf = Path('plots/HiCAT').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

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
Problem defintion
"""
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_2d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#%%
"""
Read files obtained with gurobi
"""
fname_gen = problem1.get_filename()
fname_python     = 'Test_007_' + fname_gen + '.fits'
fpath     = fdir / fname_python

print(fpath)

Apod_pyth = fits.getdata(fpath,)

print('stdgrb file: {0}'.format(fpath))

#%%

# Read files obtained with ampl

fdir_ampl = Path('apodizers/HiCAT').resolve()

#fname_gen_AMPL  = 'Test_007_HiCAT-Apod_F-N0100_nImg0032_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004_GreyFPM8543-M025_LS-Ann-bw-ID0345-OD0807-SpX0036_DZ-C080-Sep050-100_Bw10-Lam3_shiftXY100.dat'

fname_gen_AMPL    = 'Test_007_telserv3_' + fname_gen + '.fits'
#print(fname_gen_AMPL)
#print('Test_007_telserv3_HiCAT_APLC_IWA=5.0_OWA=10.0_BW=0.10_nlam=03_2D_nPup=0200_rMask=4.271_MaxTau_C=8.0_LSRobustness=1_pix_max=1_stdgrb.fits') 

#exit()
fpath_ampl = fdir_ampl / fname_gen_AMPL

print(fpath_ampl)

#Apod_ampl_raw = np.loadtxt(fpath_ampl)
#Apod_ampl = np.reshape(Apod_ampl_raw[:, 2], (nPup, nPup))

Apod_ampl = fits.getdata(fpath,)
	
#fpath = directory + filename + '.fits'
#fits.writeto(fpath, Apod_ampl_full, overwrite=True)

print('ampl file: {0}'.format(fpath_ampl))


#%% Display of the apodizer
"""
Plot display of the apodizers
"""
pl.figure(1)
pl.clf()
pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil transmission')

fname = 'Test_007_' + problem1.get_filename() + '_apodisation_telserv2.pdf'
fpath = fdir_pdf / fname

pl.figure(2)
pl.clf()
pl.imshow(Apod_pyth*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Apod transmission - {0}, telserv2'.format(solver))
pl.tight_layout()
pl.savefig(str(fpath))

fname = 'Test_007_' + problem1.get_filename() + '_apodisation_telserv3.pdf'
fpath = fdir_pdf / fname

pl.figure(3)
pl.clf()
pl.imshow(Apod_ampl*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Apod transmission - {0}, telserv3'.format(solver))
pl.tight_layout()
pl.savefig(str(fpath))

fname = 'Test_007_' + problem1.get_filename() + '_apodisation_telserv3-telserv2.pdf'
fpath = fdir_pdf / fname

pl.figure(4)
pl.clf()
pl.imshow((Apod_ampl*corono0.Pupil2d) - (Apod_pyth*corono0.Pupil2d), cmap = cm.Greys_r)
pl.title('Apod transmission - telserv3-telserv2')
pl.tight_layout()
pl.savefig(str(fpath))


#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
fname_gen  = problem1.get_filename(nlam=nlambis)
params2    = coro.update_params(params, nlam=nlambis)

if corono_name == 'SP':
    corono0 = coro.design.SP2d(**params)
else:
    corono0 = coro.design.APLC2d(**params)

if corono_name == 'APLC':
    poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod_pyth)
    poly_direct_image2 = corono0.compute_direct_intensity_2d(Apod_ampl)
else:
    poly_direct_image1 = corono0.compute_direct_intensity_2d(corono0.Pupil2d)
    poly_direct_image2 = corono0.compute_direct_intensity_2d(corono0.Pupil2d)    
    
    
poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod_pyth)
poly_corono_image2 = corono0.compute_corono_intensity_2d(Apod_ampl)

#%% image plot
"""
Display direct and coronagraphic images
"""
fname = 'Test_007_' + fname_gen + '_direct_image_telserv2.pdf'
fpath = fdir_pdf / fname
pl.figure(10)
pl.clf()
pl.imshow(poly_direct_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - direct image - {0},telserv2'.format(solver))
pl.tight_layout()
pl.savefig(str(fpath))
pl.close()

fname = 'Test_007_' + fname_gen + '_direct_image_telserv3.pdf'
fpath = fdir_pdf / fname
pl.figure(11)
pl.clf()
pl.imshow(poly_direct_image2**0.25, cmap = cm.inferno)
pl.title('Apod1 - direct image - {0},telserv3'.format(solver))
pl.tight_layout()
pl.savefig(str(fpath))
pl.close()

fname  = 'Test_007_' + fname_gen + '_apodized_image_telserv2.pdf'
fpath = fdir_pdf / fname
pl.figure(12)
pl.clf()
pl.imshow(poly_corono_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - apodized image - {0},telserv2'.format(solver))
pl.tight_layout()
pl.savefig(str(fpath))
pl.close()

fname  = 'Test_007_' + fname_gen + '_apodized_image_telserv3.pdf'
fpath = fdir_pdf / fname
pl.figure(13)
pl.clf()
pl.imshow(poly_corono_image2**0.25, cmap = cm.inferno)
pl.title('Apod1 - apodized image - {0},telserv3'.format(solver))
pl.tight_layout()
pl.savefig(str(fpath))
pl.close()

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

xi2d = corono0.xi2d_ctr
if CtrBtwnPix2 is False:
    xi2d = corono0.xi2d

nImg2d = corono0.params['nImg2d']

fname = 'Test_007_' +problem1.get_filename() + '_averaged_intensity_profiles_server_test.pdf'
fpath = fdir_pdf / fname

pl.figure(20)
pl.clf()
pl.title('Averaged Radial intensity profiles of the images')
if corono_name == 'SP':
    pl.semilogy(xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_corono_image1.max(),label='Apod - {0},telserv2'.format(solver))
    pl.semilogy(xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_corono_image2.max(),label='Apod - {0},telserv3'.format(solver))
else:
    pl.semilogy(xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='APLC - {0},telserv2'.format(solver))
    pl.semilogy(xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label='APLC - {0},telserv3'.format(solver))    
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-9, 2e0)
pl.legend()
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#%%
pl.show()

values = range(nlam)
colors = pl.cm.rainbow(np.linspace(0,1,nlam))

mono_direct_image1 = corono0.compute_direct_intensity_2d(Apod_pyth, poly=False)
mono_corono_image1 = corono0.compute_corono_intensity_2d(Apod_pyth, poly=False)

mono_direct_image2 = corono0.compute_direct_intensity_2d(Apod_ampl, poly=False)
mono_corono_image2 = corono0.compute_corono_intensity_2d(Apod_ampl, poly=False)


fname = 'Test_007_' + problem1.get_filename() + '_intensity_profiles_server_test.pdf'
fpath = fdir_pdf / fname

pl.figure(21)
pl.clf()
pl.title('Radial intensity profiles of the images')
for i in range(corono0.nlam):
    if corono_name == 'SP':
        pl.semilogy(xi2d,mono_corono_image1[i, nImg2d//2,nImg2d//2:]/mono_corono_image1.max(),label='Apod - {0}, telserv2'.format(solver), color = colors[i])
        pl.semilogy(xi2d,mono_corono_image2[i, nImg2d//2,nImg2d//2:]/mono_corono_image2.max(),label='Apod - {0}, telserv3'.format(solver), color = colors[i])                    
    else:
        pl.semilogy(xi2d,mono_corono_image1[i, nImg2d//2,nImg2d//2:]/mono_direct_image1[(corono0.nlam+1)//2].max(),
                    label='APLC - {0},telserv2'.format(solver), color = colors[i])
        pl.semilogy(xi2d,mono_corono_image2[i, nImg2d//2,nImg2d//2:]/mono_direct_image2[(corono0.nlam+1)//2].max(),
                    label='APLC - {0},telserv3'.format(solver), color = colors[i])                
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-9, 2e0)
pl.legend()
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#%%
pl.show()

