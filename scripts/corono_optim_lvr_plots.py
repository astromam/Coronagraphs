import numpy as np
import pylab as pl
import time
import os
from pathlib import Path
#from shutil import copyfile
import shutil

from astropy.io import fits
import corono as coro

## Parameters
## This section lists all the parameters for the coronagraph and the optimization problem.

# coronagraph, problem, and solver types
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'lvr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxTau' # 'MaxTau' # ,'MaxContrastLinf' # 'MaxContrastL1' #
solver       = 'stdgrb' #,'stdgrb' #  'gurobipy', 'scipy.linprog'

# keywords for solver
slvLogToConsole = 0
slvCrossover    = 0
slvMethod       = 2
allLogToConsole = 1

# additional constraints and their parameters
MinIsland         = False
Binarity          = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1

# Pixel centering of the pupils and image for optimization
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = False

#Sampling
nPup = 250
nFPM = 50
Fmax2d = 32
nImg2d = 64

#Optical System
# spectral bandwidth
bw   = 0.18
nlam = 8

#padding factor for pupil features
gap = 2

# mask radius in lam0/D units
rMask = 3.82

# LS inner and outer diameter
iD = 19
oD = 94

inD = float(iD)/100
outD = float(oD)/100

#Optimization

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  3.00
rho1 = 12.00

# contrast in the dark region
cDarkHole = 10.0

# tau (integrated Pupil transmission)
tau   = 0.4

#Export fits files
do_fits = True

print('parameters loaded')

# File reading for pupil and lyot stop

fdir = 'input_files/'
if pupil_name == 'lvr':
    fname_pup = 'apertures/TelAp_full_luvoir2017novAp05ss100cobs1gap{0}_N{1:04d}.fits'.format(gap,nPup)
    fname_lys = 'lyot_stops/luvoir_LS_ann{0:02d}D{1:02d}_clear_N{2:04d}.fits'.format(int(round(100*inD)), int(round(100*outD)), nPup)
elif pupil_name == 'HiCAT':
    #fname_pup = 'apertures/HiCAT-Aper_F-N0{0}_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004.fits'.format(nPup,)
    #fname_lys = 'lyot_stops/HiCAT-Lyot_F-N0{0}_LS-Ann-gy-ID0345-OD0{1}-SpX0036.fits'.format(nPup,oD,)
    fname_pup = 'apertures/HiCAT-Aper_F-N0{0}_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004'
    fname_lys = 'lyot_stops/HiCAT-Lyot_F-N0{0}_LS-Ann-gy-ID0345-OD0{1}-SpX0036'
else:
    fname_pup = 'apertures/pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    fname_lys = 'lyot_stops/pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)

fpath_pup = fdir + fname_pup
fpath_lys = fdir + fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)

print('aperture and lyot stop loaded')

# Definition of a dictionary of parameters 
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
                 allLogToConsole = allLogToConsole,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg)

# Coronagraph definition
# Definition of an object from the coronagraph class in design module

if corono_name == 'SP':
    # Shaped Pupil
    corono0 = coro.design.SP2d(**params)
elif corono_name == 'APLC':
    # Apodized Pupil Lyot Coronagraph
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

# Problem definition
# Definition of an object from the optim_1d class in optim_1d module.

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

#Import apodizer
fdir_apod = Path('./apodizers/').resolve() / pupil_name
#fname_apod = problem1.get_filename() + '.fits'
apod = 'HiCAT-Apod_F-N0050_nImg0032_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004_GreyFPM8543-M050_LS-Ann-bw-ID0345-OD0807-SpX0036_DZ-C030-080-Sep050-100_Bw00-Lam3_shiftXY000'
fname_apod = apod + '.fits'
fpath_apod = fdir_apod / fname_apod

Apod1_2d = fits.getdata(fpath_apod)

print('plotting...')
#----------------
#Plot parameters
# spectral sampling
nlambis = 11

# image sampling
nImg2dbis = 500

# maximum spatial frequency in the image
Fmax2dbis = 50

#Filename root for plots
#fname_gen = problem1.get_filename() + '_LS{0:02d}D{1:02d}'.format(int(round(100*inD)), int(round(100*outD)))
fname_gen = apod
#fname_gen = 'lvr_APLC_IWA=3.0_OWA=12.0_BW=0.15_nlam=07_2D_nPup=0250_rMask=3.820_MaxTau_C=10.0_gurobipy_LS19D94_19D94'

#fdir_pdf = Path('plots/').resolve() /pupil_name
fdir_pdf = 'plots/{0}/'.format(pupil_name)
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

#Aperture
fname = 'TelAp_full_luvoir2017novAp05ss100cobs1gap{0}_N{1:04d}.pdf'.format(gap,nPup)
fpath = fdir_pdf + fname
pl.figure(4)
pl.clf()
pl.imshow(corono0.Pupil2d, cmap = 'Greys_r')
pl.title('Aperture')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#Lyot Stop
fname = 'LS_full_ann{0:02d}D{1:02d}_clear_N{2:04d}.pdf'.format(int(round(100*inD)), int(round(100*outD)), nPup)
fpath = fdir_pdf + fname
pl.figure(6)
pl.clf()
pl.imshow(corono0.LyotStop2d, cmap = 'Greys_r')
pl.title('Lyot Stop')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)


#Lyot stop superimposed on aperture
fname = 'TelAp_full_luvoir2017novAp05ss100cobs1gap{0}_N{0:04d}_LS_{1:02d}D{2:02d}_clear.pdf'.format(nPup,int(round(100*inD)), int(round(100*outD)))
fpath = fdir_pdf + fname
pl.figure(6)
pl.clf()
pl.imshow(corono0.Pupil2d+corono0.LyotStop2d, cmap = 'Greys_r')
pl.title('Aperture and Lyot Stop')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)


#Apodizer
fname = fname_gen + '_{0:02d}D{1:02d}_apodisation_pix_max={2}.pdf'.format(int(round(100*inD)), int(round(100*outD)),pix_max)
fpath = fdir_pdf + fname
pl.figure(7)
pl.clf()
pl.imshow(Apod1_2d*corono0.Pupil2d, cmap = 'Greys_r')
pl.title('Apodizer solution - {0} problem - {1} solver'.format(problem_name, solver))
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#Computation of the direct and coronagraphic images
#Update of the params dictionary for plot purposes

params2    = coro.update_params(params, nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis)

#Definition of the corono.design with updated paramters

if corono_name == 'SP':
    # Shaped Pupil
    corono0 = coro.design.SP2d(**params2)
elif corono_name == 'APLC':
    # Apodized Pupil Lyot Coronagraph
    corono0 = coro.design.APLC2d(**params2)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#Computation of the broadband light images
# broadband light image computation with and without coronagraph mask
poly_direct_image1 = corono0.compute_direct_intensity_2d(Apod1_2d)
poly_corono_image1 = corono0.compute_corono_intensity_2d(Apod1_2d)

# image normalization
peak_norm_poly = 1./poly_direct_image1.max()
poly_direct_image1 *= peak_norm_poly
poly_corono_image1 *= peak_norm_poly

#Computation of the monochromatic light images
# monochromatic light image computation with and without coronagraph mask
mono_direct_image1 = corono0.compute_direct_intensity_2d(Apod1_2d, poly=False)
mono_corono_image1 = corono0.compute_corono_intensity_2d(Apod1_2d, poly=False)

# image normalization
peak_norm_mono = 1./mono_direct_image1[(corono0.nlam+1)//2].max()
mono_direct_image1 *= peak_norm_mono
mono_corono_image1 *= peak_norm_mono

#Plot display of the images

#Direct image
fname = fname_gen + '_direct_image_pix_max={0}.pdf'.format(pix_max)
fpath = fdir_pdf + fname
pl.figure(8)
pl.clf()
pl.imshow(np.log10(poly_direct_image1), cmap = 'inferno', vmin=-9, vmax=0.)
pl.title('Direct image')
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#Write direct image to fits
#fname = fname_gen + '_direct_image_pix_max={0}.fits'.format(pix_max)
#fpath = fdir_pdf + 'fits_images/' + fname
#fits.writeto(fpath, poly_direct_image1, overwrite=True)

#Coronagraphic image
fname = fname_gen + '_apodized_image_pix_max={0}.pdf'.format(pix_max)
fpath = fdir_pdf + fname
pl.figure(9)
pl.clf()
pl.imshow(np.log10(poly_corono_image1), cmap = 'inferno', vmin=-9, vmax=0.)
pl.title('Coronagraphic image')
cbar = pl.colorbar()
cbar.set_label('Normalized intensity in log scale')
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#Write direct image to fits
#fname = fname_gen + '_apodized_image_pix_max={0}.fits'.format(pix_max)
#fpath = fdir_pdf + 'fits_images/' + fname
#fits.writeto(fpath, poly_corono_image1, overwrite=True)

#Broadband light
xi2d = corono0.xi2d
if nImg2dbis%2 == 0:
    xi2d = corono0.xi2d_ctr

nImg2d = corono0.params['nImg2d']

fname = fname_gen + '_broadband_intensity_profiles_pix_max={0}.pdf'.format(pix_max)
fpath = fdir_pdf + fname

pl.figure(10)
pl.clf()
pl.title('Radial intensity profiles in broadband light')
pl.semilogy(xi2d,poly_corono_image1[nImg2dbis//2,nImg2dbis//2:],
                label='Corono')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-12, 2e0)
pl.legend()
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)

#Monochromatic light
values = range(nlambis)
colors = pl.cm.rainbow(np.linspace(0,1,nlambis))

fname = fname_gen + '_monochrom_intensity_profiles_pix_max={0}.pdf'.format(pix_max)
fpath = fdir_pdf + fname

pl.figure(11)
pl.clf()
pl.title('Radial intensity profiles in monochromatic light')
for i in range(corono0.nlam):
    pl.semilogy(xi2d,mono_corono_image1[i, nImg2dbis//2,nImg2dbis//2:], '-',
                    label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[i]),
                    color = colors[i])
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-12, 2e0)
pl.legend()
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)


#rename and move log file to logs directory
#fdir_logs = Path('logs/').resolve() /pupil_name
#fdir_logs = 'logs/{0}/'.format(pupil_name)

#if not os.path.exists(fdir_logs):
#    os.makedirs(fdir_logs)

#fname = fname_gen + '_telserv3.log'
#fpath = fdir_logs + fname

#os.rename('gurobi.log', fpath)



