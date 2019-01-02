#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import os
import pylab as pl
from astropy.io import fits

import corono as coro

t0_total = time.time()

#%% parameters
"""
Parameters
"""
# Telescope name
pupil_name   = 'lvr' # 'vlt' or 'sbr' or 'lvr'

# optimization parameters
problem_name = 'MaxTau' # 'MaxContrastL1' # 'MaxTau' # ,'MaxContrastLinf' # #  
solver       = 'stdgrb' #,'stdgrb' #  'gurobipy', 'scipy.linprog'
slvLogToConsole = 0
slvCrossover    = 0
slvMethod       = 2
allLogToConsole = 1

MinIsland   = False
Binarity    = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1
LSRobustness = False

# Coronagraph type
# Set Pupil2dSym to False if pupil is not symmetric or optimization to Lyot
# stop position
corono_name = 'APLC' # 'SP' or 'APLC'
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = False
ImPart      = True

# sampling
nPup = 250
nFPM = 50
Fmax2d = 32 #22.5
nImg2d = 64 #45

# mask radius in lam0/D units
rMask = 3.82

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  3.00 #3.75 #5.0
rho1 = 12.00 #15.0 #10.0

# contrast in the dark region
cDarkHole = 10.0 #6.0

# tau (integrated Pupil transmission)
tau   = 0.4

# spectral bandpass
bw   = 0.18
nlam = 8

# maximum pixel shift along a given axis for Lyot stop; total robostness is x2 pix_max
pix_max   = 1

if LSRobustness == False:
	pix_max = 0

# LS inner and outer diameter -> LUVOIR/SCDA format
iD = 19
oD = 94

inD = float(iD)/100
outD = float(oD)/100

# save in fits file
do_fits = True

#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = 'input_files/'
if pupil_name == 'lvr':
    fname_pup = 'ATLAST_Aperture_nPup={0}.fits'.format(nPup,)
    fname_lys = 'ATLAST_LyotStop_nPup={0}.fits'.format(nPup,)
elif pupil_name == 'HiCAT':
	fname_pup = 'HiCAT-Aper_F-N0{0}_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004'.format(nPup,)
	fname_lys = 'HiCAT-Lyot_F-N0{0}_LS-Ann-bw-ID0345-OD0807-SpX0036_shiftX+000'.format(nPup,)
else:
    fname_pup = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)
    fname_lys = 'pupil={0}_nPup={1}.fits'.format(pupil_name, nPup,)

#fpath_pup = fdir / fname_pup
#fpath_lys = fdir / fname_lys
#Pupil2d    = fits.getdata(fpath_pup)
#LyotStop2d = fits.getdata(fpath_lys)

Pupil2d = dat_input_to_fits(nPup,fdir,'apertures/'+fname_pup)
LyotStop2d = dat_input_to_fits(nPup,fdir,'lyot_stops/'+fname_lys)

# List of Lyot stops for the design optimization
LyotStop2d_t = [LyotStop2d]

# List construction for Lyot stop position shifts
pix_t = []
if pix_max >= 1 and LSRobustness == True:
    pix_pos_t = 1+np.arange(pix_max)
    pix_neg_t = - pix_pos_t
    pix_t = list(-pix_pos_t) + list(pix_pos_t)
    pix_t.sort()

# List construstion for the Lyot stops 
for j in range(2):
    for i in range(len(pix_t)):
        LyotStop2d_t.append(np.roll(LyotStop2d, pix_t[i], axis=j))

# number of coronagraph configuration
ncorono      = len(LyotStop2d_t)
print('# of coronagraph configurations: {0}'.format(ncorono))

# default solver 
if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

# list parameters for the coronagraph and the optimization problem
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
                 Binarity = Binarity, BinarityReg = BinarityReg,
                 ImPart = ImPart, LSRobustness = LSRobustness, pix_max = pix_max)

# list of parameters for each coronagraph configuration
params_t = []
for k in range(ncorono):
    params_t.append(coro.update_params(params, LyotStop2d=LyotStop2d_t[k])) 

# initialization of coronagraph list
corono_t = []


#%%  
""" 
Coronagraph defintion
"""
# generation of a coronagraph object
if corono_name == 'SP':
    corono_t.append(coro.design.SP2d(**params))
elif corono_name == 'APLC':
    for k in range(ncorono):
        corono_t.append(coro.design.APLC2d(**params_t[k])) 
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
# generation of a optimzation problem object
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = coro.optim_2d.MaxTau(corono=corono_t, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono_t, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = coro.optim_2d.MaxContrast(corono=corono_t, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))

#%% Apodizer solution for the problems
"""
Apodizer import
"""
fdir = 'apodizers/{0}/'.format(pupil_name)   
apod = 'Test_009_HiCAT-Apod_F-N0100_nImg0032_Hex3-Ctr0972-Obs0195-SpX0017-Gap0004_GreyFPM8543-M025_LS-Ann-bw-ID0345-OD0807-SpX0036_DZ-C080-Sep050-100_Bw10-Lam3_shiftXY100'
#apod = 'Test_008_telserv2_' + problem1.get_filename()

fname = apod + '.fits'
fpath = fdir + fname

Apod1_2d = fits.getdata(fpath)

"""
Generate and save plots
"""

print('plotting...')

#----------------
#Plot parameters
# spectral sampling
nlambis = 11

# image sampling
nImg2dbis = 500

# maximum spatial frequency in the image
Fmax2dbis = 50

# test for lyot stop robustness in plots 

LSRobustness_plot = True
pix_max = 1

if LSRobustness_plot == False:
	pix_max = 0

# List of Lyot stops for the design optimization
LyotStop2d_t = [LyotStop2d]

# List construction for Lyot stop position shifts
pix_t = []
if pix_max >= 1 and LSRobustness_plot == True:
    pix_pos_t = 1+np.arange(pix_max)
    pix_neg_t = - pix_pos_t
    pix_t = list(-pix_pos_t) + list(pix_pos_t)
    pix_t.sort()

# List construstion for the Lyot stops 
for j in range(2):
    for i in range(len(pix_t)):
        LyotStop2d_t.append(np.roll(LyotStop2d, pix_t[i], axis=j))

# number of coronagraph configuration
ncorono      = len(LyotStop2d_t)
print('# of coronagraph configurations in plots: {0}'.format(ncorono))

# list of parameters for each coronagraph configuration in plotting
params_t = []
for k in range(ncorono):
    params_t.append(coro.update_params(params, LyotStop2d=LyotStop2d_t[k])) 


#Filename root for plots

fname_gen = apod
fdir_pdf = 'plots/{0}/'.format(pupil_name)
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)

for k in [0]:
	#Aperture
	fname = fname_pup + '.pdf'
	fpath = fdir_pdf + fname 
	pl.figure(1)
	pl.clf()
	pl.imshow(corono_t[k].Pupil2d, cmap = 'Greys_r')
	pl.title('Aperture')
	pl.tight_layout()
	pl.savefig(str(fpath), transparent=True)
	pl.close()

	#Lyot stop superimposed on aperture, first (central) configuration
	fname = fname_pup + '_' + fname_lys + '.pdf'
	fpath = fdir_pdf + fname
	pl.figure(4)
	pl.clf()
	pl.imshow(corono_t[k].Pupil2d+corono_t[k].LyotStop2d, cmap = 'Greys_r')
	pl.title('Aperture and Lyot Stop')
	pl.tight_layout()
	pl.savefig(str(fpath), transparent=True)
	pl.close()

	#Apodizer
	fname = fname_gen + '.pdf'
	fpath = fdir_pdf + fname
	pl.figure(7)
	pl.clf()
	pl.imshow(Apod1_2d*corono_t[k].Pupil2d, cmap = 'Greys_r')
	pl.title('Apodizer solution - {0} problem - {1} solver'.format(problem_name, solver))
	pl.tight_layout()
	pl.savefig(str(fpath), transparent=True)
	pl.close() 

#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""
#fname_gen  = problem1.get_filename(nlam=nlambis)
fname_gen = apod

params2_t  = []
for k in range(ncorono):
    params2_t.append(coro.update_params(params_t[k], nlam=nlambis, Fmax2d = Fmax2dbis, nImg2d = nImg2dbis)) 

corono_t = []
if corono_name == 'SP':
    corono_t.append(coro.design.SP2d(**params2_t[0]))
elif corono_name == 'APLC':
    for k in range(ncorono):
        corono_t.append(coro.design.APLC2d(**params2_t[k])) 
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

poly_direct_image_t = [] 
poly_corono_image_t = []

mono_direct_image_t = []
mono_corono_image_t = []

values = range(nlambis)
colors = pl.cm.rainbow(np.linspace(0,1,nlambis))

res_intensity_t = []

for k in range(ncorono):
	if corono_name == 'APLC':
		poly_direct_image_t.append(corono_t[k].compute_direct_intensity_2d(Apod1_2d))
	else:
		poly_direct_image_t.append(corono_t[k].compute_direct_intensity_2d(corono_t[k].Pupil2d))
	
	poly_corono_image_t.append(corono_t[k].compute_corono_intensity_2d(Apod1_2d))
    

	#Lyot Stop, per position
	fname = fname_lys + '_config={0}.pdf'.format(k)
	fpath = fdir_pdf + fname
	pl.figure(8)
	pl.clf()
	pl.imshow(corono_t[k].LyotStop2d, cmap = 'Greys_r')
	pl.title('Lyot stop - config={0}'.format(k))
	pl.tight_layout()
	pl.savefig(str(fpath), transparent=True)
	pl.close()

	#Direct image, per lyot stop position
	fname = fname_gen + '_direct_image_config={0}.pdf'.format(k)
	fpath = fdir_pdf + fname

	pl.figure(10*k)
	pl.clf()
	pl.imshow(np.log10(poly_direct_image_t[k]), cmap = 'inferno', vmin=-7, vmax=0.)
	pl.title('Apod1 - direct image - config={0}'.format(k))
	cbar = pl.colorbar()
	cbar.set_label('Normalized intensity in log scale')
	pl.tight_layout()
	pl.savefig(str(fpath), transparent=True)
	pl.close()
	
	
	#Coronagraphic image, per lyot stop position
	fname = fname_gen + '_apodized_image_config={0}.pdf'.format(k)
	fpath = fdir_pdf + fname
	pl.figure(10*k+1)
	pl.clf()
	pl.imshow(np.log10(poly_corono_image_t[k]), cmap = 'inferno', vmin=-7, vmax=0.)
	pl.title('Apod1 - apodized image - config={0}'.format(k))
	cbar = pl.colorbar()
	cbar.set_label('Normalized intensity in log scale')
	pl.tight_layout()
	pl.savefig(str(fpath), transparent=True)
	pl.close()

#%% Intensity profiles of the direct and coronagraphic images
	"""
	Display of the intensity profiles of the coronagraphic images
	"""
	
	xi2d = corono_t[k].xi2d
	if nImg2dbis%2 == 0:
		xi2d = corono_t[k].xi2d_ctr
	
	nImg2d = corono_t[k].params['nImg2d']
	
	fname = fname_gen + '_broadband_intensity_profiles_config={0}.pdf'.format(k)
	fpath = fdir_pdf + fname
	
	pl.figure(10*k+2)
	pl.clf()
	pl.title('Averaged Radial intensity profiles of the images - config={0}'.format(k))
	if corono_name == 'SP':
		pl.semilogy(xi2d,poly_corono_image_t[k][nImg2dbis//2,nImg2dbis//2:]/poly_corono_image_t[k].max(),label=solver)
	else:
		pl.semilogy(xi2d,poly_corono_image_t[k][nImg2dbis//2,nImg2dbis//2:]/poly_direct_image_t[k].max(),label=solver)    
	
	pl.axvline(x=corono_t[k].rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
	pl.axvline(x=corono_t[k].rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
	pl.axhline(10**(-cDarkHole), xmin=corono_t[k].xi.min(), xmax=corono_t[k].xi.max(), linewidth=1, color='k', linestyle='--')
	pl.xlabel(r'Angular separation in $\lambda_0$/D')
	pl.ylabel('Normalized intensity in log scale')
	pl.ylim(1e-9, 2e0)
	pl.legend()
	pl.tight_layout()
	pl.savefig(str(fpath), transparent=True)
	pl.close()
#%%
	mono_direct_image_t.append(corono_t[k].compute_direct_intensity_2d(Apod1_2d, poly=False))
	mono_corono_image_t.append(corono_t[k].compute_corono_intensity_2d(Apod1_2d, poly=False))
	
	fname = fname_gen + '_monochromatic_intensity_profiles_config={0}.pdf'.format(k)
	fpath = fdir_pdf + fname
	
	pl.figure(k*10+3)
	pl.clf()
	pl.title('Radial intensity profiles of the images - config={0}'.format(k))
	for i in range(corono_t[k].nlam):
		if corono_name == 'SP':
			pl.semilogy(xi2d,mono_corono_image_t[k][i, nImg2dbis//2,nImg2dbis//2:]/mono_corono_image_t[k].max(), '-',
				label=r'{0:.2f}$\lambda_0$'.format(corono_t[k].lam_t[i]),
				color = colors[i])
		else:
			pl.semilogy(xi2d,mono_corono_image_t[k][i, nImg2dbis//2,nImg2dbis//2:]/mono_direct_image_t[k][(corono_t[k].nlam+1)//2].max(), 
				'-',label=r'{0:.2f}$\lambda_0$'.format(corono_t[k].lam_t[i]),  color = colors[i])
	pl.axvline(x=corono_t[k].rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
	pl.axvline(x=corono_t[k].rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
	pl.axhline(10**(-cDarkHole), xmin=corono_t[k].xi.min(), xmax=corono_t[k].xi.max(), linewidth=1, color='k', linestyle='--')
	pl.xlabel(r'Angular separation in $\lambda_0$/D')
	pl.ylabel('Normalized intensity in log scale')
	pl.ylim(1e-9, 2e0)
	pl.legend()
	pl.tight_layout()
	pl.savefig(str(fpath), transparent=True)
	pl.close()
#%%
	dz2d, rad2d = corono_t[k].generate_area()
	dz = np.reshape(dz2d, (corono_t[k].nImg2d**2))
	aaa     = np.arange(corono_t[k].nImg2d**2)
	idx_dz  = list(aaa[dz])  
	ndz     = len(idx_dz)
	
	poly_corono_image_1d = np.reshape(poly_corono_image_t[k], nImg2d**2)
	res_intensity_t.append(np.mean(poly_corono_image_1d[idx_dz])/poly_direct_image_t[k].max())

print(len(res_intensity_t))
#%%	

LS_rad_pix = 122
LS_rad_mm = 15.9/2

pix_t   = np.asarray([0, -1, 1, -1, 1])
Dfrac_t = (pix_t/LS_rad_pix)*LS_rad_mm

#
mylist0 = [1,0,2]
mylist1 = [3,0,4] 

fname = fname_gen + '_intensity_Lyot_stop_position.pdf'
fpath = fdir_pdf + fname
#
#    
pl.close('all')    
pl.figure(0)
pl.clf()
pl.title('Mean DZ intensity vs Lyot stop position')
pl.plot(Dfrac_t[mylist0], np.log10(np.asarray(res_intensity_t)[mylist0]), 'x-',
            label='x-axis')
pl.plot(Dfrac_t[mylist1], np.log10(np.asarray(res_intensity_t)[mylist1]), 'x-', 
            label='y-axis')
pl.xlabel(r'Lyot stop offset in mm (approx.)')
pl.ylabel(r'Normalized mean intensity in log scale'.format(10**(-cDarkHole-1)))
pl.ylim(-11, -4)
#pl.xlim(-0.004,0.004)
pl.legend()
pl.tight_layout()
pl.savefig(str(fpath), transparent=True)	

#rename and move log file to logs directory
#fdir_logs = Path('logs/').resolve() /pupil_name
fdir_logs = 'logs/{0}/'.format(pupil_name)

if not os.path.exists(fdir_logs):
    os.makedirs(fdir_logs)

fname = fname_gen + '_telserv3.log'
fpath = fdir_logs + fname

if solver == 'stdgrb':
	os.rename('dense.log', fpath)
elif solver == 'gurobipy':
	os.rename('gurobi.log', fpath)

    
t1_total = time.time()
print('total run time             : {0:.2f}s'.format(t1_total-t0_total))