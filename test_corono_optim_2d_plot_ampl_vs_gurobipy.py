#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

@author: mndiaye
"""
import numpy as np
import time
import pylab as pl
from pathlib import Path

from corono import corono_design as cd
from corono import corono_optim_2d as co2d

from corono.utils import to_dict

from matplotlib import cm

from astropy.io import fits

#%% parameters
"""
Parameters
"""
# Telescope name
pupil_name = 'sbr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxTau' # 'MaxContrastL1' # 'MaxTau' # ,'MaxContrastL1' # #  
solver       = 'gurobipy' #,'stdgrb' #  'gurobipy', 'scipy.linprog'

#nPup = corono0.params['nPup']
nPup = 50

Fmax2d = 50 
nImg2d = 500

# mask radius in lam0/D unit
rMask = 2.8

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  5.0
rho1 = 10.0

# contrast in the dark region
cDarkHole = 7

# tau (integrated Pupil transmission)
tau   = 0.4

# CtrBtwnPix2
corono_name   = 'SP' # 'SP' or 'APLC'
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True

#nlam
nlam = 5
bw   = 0.1

do_fits = True

#%%
"""
File reading for Pupil and Lyot stop
"""
fdir = Path('./pupils/2D/').resolve()
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

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name)

#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'SP':
    corono0 = cd.SP2d(**params)
elif corono_name == 'APLC':
    corono0 = cd.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
Problem defintion
"""
t0 = time.time()
if problem_name == 'MaxTau':
    # Maximization of the integrated amplitude transmission of the apodizer
    problem1 = co2d.MaxTau(corono=corono0, **params)
elif problem_name == 'MaxContrastL1':
    # Maximization of the contrast under L1-norm
    problem1 = co2d.MaxContrast(corono=corono0, Lnorm='L1',**params)
elif problem_name == 'MaxContrastLinf':
    # Maximization of the contrast under L-infinite norm
    problem1 = co2d.MaxContrast(corono=corono0, Lnorm='Linf',**params)
else:
    raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))
    
t1 = time.time()
print('problem definition time       : {0:.2f}s'.format(t1-t0))

#%%
"""
Read files obtained with gurobi
"""
fdir = Path('./results/2D/dat_pyth').resolve() / pupil_name

fname = problem1.get_filename()
fpath = fdir / fname

Apod_pyth = fits.getdata(fpath,)

#%%
"""
Read files obtained with ampl
"""
fdir_ampl = Path('/Users/mndiaye/Dropbox/central storage/AMPL/PupilDataFiles/2D/Subaru/dat/').resolve()

if corono_name == 'SP':
    fname_gen  = 'SP00_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}_ampl.dat'
else:
    fname_gen  = 'APLC_IWA={rho0}_OWA={rho1}_BW={bw}_nlam={nlam:02d}_C={cDarkHole:.1f}_2D_nPup={nPup:04d}_ampl.dat'


fpath_ampl = fdir_ampl / fname_gen.format(**{key: corono0.params[key] for key in corono0.params})

print('{0}'.format(fpath_ampl))

Apod_ampl_raw = np.loadtxt(fpath_ampl)
Apod_ampl_quarter = np.reshape(Apod_ampl_raw[:, 2], (nPup//2, nPup//2))

Apod_ampl = np.zeros((nPup, nPup))
Apod_ampl[nPup//2:, nPup//2:] = Apod_ampl_quarter
Apod_ampl[:nPup//2, nPup//2:] = np.flip(Apod_ampl_quarter, axis=0)
Apod_ampl[:,:nPup//2] = np.flip(Apod_ampl[:,nPup//2:], axis=1)

#%% Display of the apodizer
"""
Plot display of the apodizers
"""
pl.figure(1)
pl.clf()
pl.imshow(corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Pupil transmission')

fdir_pdf = Path('./results/2D/plots/').resolve()
fname = '{0}_apodisation_pyth_nPup={1:04d}_{2}.pdf'.format(pupil_name, nPup, corono_name)
fpath = fdir_pdf / fname
pl.figure(2)
pl.clf()
pl.imshow(Apod_pyth*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 transmission - {0}'.format(solver))
pl.tight_layout()
pl.savefig(str(fpath))

fdir_pdf = Path('./results/2D/plots/').resolve()
fname = '{0}_apodisation_ampl_nPup={1:04d}_{2}.pdf'.format(pupil_name, nPup, corono_name)
fpath = fdir_pdf / fname
pl.figure(3)
pl.clf()
pl.imshow(Apod_ampl*corono0.Pupil2d, cmap = cm.Greys_r)
pl.title('Apod 1 transmission - ampl')
pl.tight_layout()
pl.savefig(str(fpath))


#%% Signal in intensity
"""
Computation of the direct and coronagraphic images
"""

nlam = 5

params = to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2=CtrBtwnPix2, 
                 nlam=nlam, bw=bw,
                 Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, solver= solver)

if corono_name == 'SP':
    corono0 = cd.SP2d(**params)
else:
    corono0 = cd.APLC2d(**params)

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
fname = '{0}_direct_image_nPup={1:04d}_{2}.pdf'.format(pupil_name, nPup, corono_name)
fpath = fdir_pdf / fname
pl.figure(10)
pl.clf()
pl.imshow(poly_direct_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - direct image - {0}'.format(solver))
pl.tight_layout()
pl.savefig(str(fpath))

pl.figure(11)
pl.clf()
pl.imshow(poly_direct_image2**0.25, cmap = cm.inferno)
pl.title('Apod1 - direct image - ampl')
pl.tight_layout()
pl.savefig(str(fpath))

fname  = '{0}_apodized_image_pyth_nPup={1:04d}_{2}.pdf'.format(pupil_name, nPup, corono_name)
fpath = fdir_pdf / fname
pl.figure(12)
pl.clf()
pl.imshow(poly_corono_image1**0.25, cmap = cm.inferno)
pl.title('Apod1 - apodized image - {0}'.format(solver))
pl.tight_layout()
pl.savefig(str(fpath))

fname  = '{0}_apodized_image_ampl_nPup={1:04d}_{2}.pdf'.format(pupil_name, nPup, corono_name)
fpath = fdir_pdf / fname
pl.figure(13)
pl.clf()
pl.imshow(poly_corono_image2**0.25, cmap = cm.inferno)
pl.title('Apod1 - apodized image - ampl')
pl.tight_layout()
pl.savefig(str(fpath))

#%% Intensity profiles of the direct and coronagraphic images
"""
Display of the intensity profiles of the coronagraphic images
"""

xi2d = corono0.xi2d_ctr
if CtrBtwnPix2 is False:
    xi2d = corono0.xi2d

nImg2d = corono0.params['nImg2d']
fname = '{0}_intensity_profiles_nPup={1:04d}_{2}.pdf'.format(pupil_name, nPup, corono_name)
fpath = fdir_pdf / fname

pl.figure(20)
pl.clf()
pl.title('Radial intensity profiles of the images')
#pl.semilogy(corono0.xi2d,poly_direct_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='Direct - pyth', linestyle='--')
#pl.semilogy(corono0.xi2d,poly_direct_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label='Direct - ampl', linestyle='--')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
if corono_name == 'SP':
    pl.semilogy(xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_corono_image1.max(),label='Apod - {0}'.format(solver))
    pl.semilogy(xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_corono_image2.max(),label='Apod - ampl')
else:
    pl.semilogy(xi2d,poly_corono_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='APLC - {0}'.format(solver))
    pl.semilogy(xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label='APLC - ampl')    
#pl.semilogy(corono0.xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
#pl.semilogy(corono0.xi2d,poly_corono_image3[nImg2d//2,nImg2d//2:]/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
#pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
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



pl.figure(21)
pl.clf()
pl.title('Radial intensity profiles of the images')
#pl.semilogy(corono0.xi2d,poly_direct_image1[nImg2d//2,nImg2d//2:]/poly_direct_image1.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
#pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
for i in range(corono0.nlam):
    if corono_name == 'SP':
        pl.semilogy(xi2d,mono_corono_image1[i, nImg2d//2,nImg2d//2:]/mono_corono_image1.max(),label='Apod - {0}'.format(solver), 
                    color = colors[i])
    else:
        pl.semilogy(xi2d,mono_corono_image1[i, nImg2d//2,nImg2d//2:]/mono_direct_image1[(corono0.nlam+1)//2].max(),
                    label='APLC - {0}'.format(solver), color = colors[i])
#pl.semilogy(corono0.xi2d,poly_corono_image2[nImg2d//2,nImg2d//2:]/poly_direct_image2.max(),label=r'MaxContrast, L$_1$-norm')
#pl.semilogy(corono0.xi2d,poly_corono_image3[nImg2d//2,nImg2d//2:]/poly_direct_image3.max(),label=r'MaxContrast, L$_{\infty}$-norm')
#pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.ylim(1e-9, 2e0)
pl.legend()
pl.tight_layout()

#%%
pl.show()

