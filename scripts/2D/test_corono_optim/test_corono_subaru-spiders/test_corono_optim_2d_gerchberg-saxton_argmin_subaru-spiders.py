#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:18:24 2021

@author: mndiaye
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:52:23 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

import numpy as np
import time
import os
from pathlib import Path

import corono as coro

from astropy.io import fits

import pwd
import sys

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

import matplotlib.pyplot as pl
ftsz = 12 
pl.rcParams.update({'font.size': ftsz})

#%% parameters
"""
Parameters
"""
# Telescope name
corono_name  = 'APLC' # 'SP' or 'APLC'
pupil_name   = 'sbr' # 'vlt' or 'sbr' or 'lvr'
problem_name = 'MaxTau' # 'MaxTau' # ,'MaxContrastLinf' # 'MaxContrastL1' #
solver       = 'stdgrb' #,'stdgrb' #  'gurobipy', 'scipy.linprog'
slvLogToConsole = 1
slvCrossover    = 0
slvMethod       = 2
slvSparse       = 0
allLogToConsole = 1

MinIsland   = False
Binarity    = False
FirstDerGlobalLim = 100.
BinarityReg       = 0.1

#nPup = corono0.params['nPup']
nPup = 200
nFPM = 50
Fmax2d = 16#22.5
nImg2d = 4*Fmax2d#45

# telescope parameters
pdiam, odiam = 7.92, 2.3  # tel. and obst. diameters (meters)
thick = 0.25              # adopted spider thickness (meters)
offset = 1.278            # spider intersection offset (meters)
beta = 51.75              # spider angle beta

kpdiam_t = np.linspace(0.9, 1.0, 11)
kodiam_t = np.linspace(1.0, 2.0, 21)
kthick_t = np.linspace(1.0, 2.0, 11)

pdiam2_t = np.asarray(kpdiam_t)*pdiam
odiam2_t = np.asarray(kodiam_t)*odiam
thick2_t = np.asarray(kthick_t)*thick 

npdiam = len(kpdiam_t)
nodiam = len(kodiam_t)
nthick = len(kthick_t)

nIter = npdiam*nodiam*nthick

Fratio    = 64

# Focal plane mask 
mas2rad   = np.pi/(180.*3600*1000) # Conversion factor from mas to rads
rad2mas   = 1/mas2rad

# mask radius in lam0/D units
rMask_m = 453e-6/2 

#rMask = 2.8

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 5.0
rho1 = 7.0

# contrast in the dark region
cDarkHole = 7.0

# tau (integrated Pupil transmission)
tau   = 0.5

# CtrBtwnPix2
CtrBtwnPix  = True
CtrBtwnPix2 = True
Pupil2dSym  = True
ImPart = False 

#nlam
band = 'GPI_J'
#bw   = 0.1
nlam = 5
#nlam1 = 101

do_fits = True
do_num_mask = True
do_plot = True
do_apod_spiders = True

thick_apod = 0.
str_apod_spiders = '_apodnospiders'
if do_apod_spiders:
    thick_apod = thick*1
    str_apod_spiders = ''

#%%
"""
### Spectral parameters
"""
wv0_z   = 8925.96e-10
width_z = 200.0e-10
bw_z  = width_z/wv0_z
rMask_z = rMask_m/(wv0_z*Fratio) 

wv0_Y   = 10433.59e-10
width_Y = 1889.08e-10
bw_Y  = width_Y/wv0_Y
rMask_Y = rMask_m/(wv0_Y*Fratio) 

wv0_J   = 12317.58e-10
wv1_J   = (1.72/1.65)*wv0_J
width_J = 2273.20e-10
bw_J  = width_J/wv0_J
rMask_J = rMask_m/(wv0_J*Fratio) 

wv0_H   = 16444.09e-10
wv1_H   = (1.72/1.65)*wv0_H
width_H = 2984.82e-10
bw_H  = width_H/wv0_H
rMask_H = rMask_m/(wv0_H*Fratio)            

if band == 'HSC_z':
    wv0 = wv0_z
    wv1 = wv0*1
    width = width_z 
elif band == 'GPI_Y':
    wv0 = wv0_Y
    wv1 = wv0*1
    width = width_Y
elif band == 'GPI_J':
    wv0 = wv0_J
    wv1 = wv1_J
    width = width_J
elif band == 'GPI_H':
    wv0 = wv0_H
    wv1 = wv1_H
    width = width_H           
else:
    raise ValueError(f'Unknown {band} band')
    
# wavelength sampling
bw     = width/wv0 
lam0   = 1. 
dlam   = bw*lam0
lam_t  = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam)
wv_t   = wv0*lam_t

rMask     = rMask_m/(wv0*Fratio)  # mask size in lam0/D
rMask_mas = rMask * (wv0/pdiam)/mas2rad

# lam1_t  = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam1)
#wv1opt = rMask_m/(rMask*Fratio)
#print(f'lam1={wv1opt}m - wv1opt={wv1opt/wv0_J:.3f}lam0')

rMask1min = np.round(rMask_m/((wv0_H+width_H/2)*Fratio), decimals=2)
rMask1max = np.round(rMask_m/((wv0_z-width_z/2)*Fratio), decimals=2)

nMask1 = int(np.round((rMask1max-rMask1min)*100))+1

rMask1_t = np.linspace(rMask1min, rMask1max, nMask1)

# wv1_t   = rMask_m/(rMask1_t*Fratio)

#%%
"""
File reading for Pupil and Lyot stop
"""
#fdir = Path('../../data/2D/pupils/').resolve()
if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()
    elif syst == 'linux':
        fdir = Path('/scratch/mndiaye/data/Coronagraphs/data/2D/pupils/').resolve()
    else:
        raise ValueError('Unknown operating system {0}'.format(syst))
else:
    raise ValueError('Unknown user {0}'.format(user))

if pupil_name == 'sbr':
    fname_pup = f'pupil=sbr_nPup={nPup}_odiam={int(odiam*100)}_thick={int(thick*100):03d}.fits'
    fname_lys = f'pupil=sbr_nPup={nPup}_odiam={int(odiam*100)}_thick={int(thick*100):03d}.fits'
else:
    raise NameError(f'{pupil_name}: unknown pupil name')

fpath_pup = fdir / fname_pup
fpath_lys = fdir / fname_lys
Pupil2d    = fits.getdata(fpath_pup)
LyotStop2d = fits.getdata(fpath_lys)


if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

pup_ratio = odiam/pdiam

Pupil2dnospiders = coro.utils.uniform_disk(nPup, nPup//2,CtrBtwnPix=True) - \
    coro.utils.uniform_disk(nPup, pup_ratio*nPup//2,CtrBtwnPix=True)

#Input = Pupil2dnospiders*1
Input = Pupil2d*1


params = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                 rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                 CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                 nlam=nlam, bw=bw,
                 Pupil2d = Input, LyotStop2d = LyotStop2d,
                 Pupil2dSym = Pupil2dSym, rMask=rMask,
                 problem_name = problem_name, 
                 solver = solver, 
                 corono_name = corono_name, pupil_name = pupil_name,
                 slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 slvSparse = slvSparse,
                 allLogToConsole = allLogToConsole,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                 Binarity = Binarity, BinarityReg = BinarityReg,
                 ImPart = ImPart)

#%%
"""
Working directories
"""
#fdir = Path('../../results/2D/dat_pyth').resolve() / pupil_name
fdir_pdf = Path('../../results/2D/plots/').resolve()
if not os.path.exists(fdir_pdf):
    os.makedirs(fdir_pdf)


#%%  
""" 
Coronagraph defintion
"""
if corono_name == 'APLC':
    corono0 = coro.design.APLC2d(**params)
else:
    raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))

#%%
"""
### Computation of a numerical mask to compute contrast outside the spiders diffraction pattern
"""
num_mask = np.ones((nImg2d, nImg2d))
str_num_mask = ''
if do_num_mask:
    val = 0
    if nImg2d %2 == 0:
        val = 1/2
    
    xx1, yy1  = np.meshgrid(np.arange(nImg2d)-nImg2d//2, np.arange(nImg2d)-nImg2d//2)
    
    beta2 = (beta+90)*np.pi/180
    beta3 = (-beta+90)*np.pi/180
    xx2 = -np.sin(beta2)*xx1 + np.cos(beta2)*yy1
    xx3 = -np.sin(beta3)*xx1 + np.cos(beta3)*yy1
    
    thick_diff = 6.4#32#(pdiam/thick2)*nImg2d/Fmax2d
    
    num_circ = coro.utils.uniform_disk(nImg2d, rMask_J*nImg2d/Fmax2d)
    
    num_mask[(xx2 <= thick_diff)*(xx2 >= -thick_diff)] = 0
    num_mask[(xx3 <= thick_diff)*(xx3 >= -thick_diff)] = 0
    num_mask[num_circ == 1] = 0
    
    str_num_mask = '_nummask=1'

    
#%%
rr_D = coro.utils.radius_disk(nImg2d, nImg2d//2, CtrBtwnPix=True)
rr_D *= Fmax2d
ind_D = (rr_D <= rho1)*(rr_D >= rho0)*(num_mask == 1)

area_D = np.zeros((nImg2d, nImg2d))
area_D[ind_D] = 1.  

#%%
EE_D_t = np.zeros((nIter, nMask1))

fname_EE_D = f'pupilsbr_nPup{nPup}_EE_D_rho0{int(np.round(rho0*100)):03d}_rho1{int(np.round(rho1*100)):03d}' + str_num_mask + str_apod_spiders +'.fits'
fpath_EE_D = fdir / fname_EE_D

EE_D_t = fits.getdata(fpath_EE_D)

#%%
"""
### Search for minimum
"""

iIter, iMask1 = np.unravel_index(EE_D_t.argmin(), EE_D_t.shape)
print(f'iIter: {iIter:04d}, iMask1: {iMask1:03d}')

#ipdiam, iodiam, ithick = np.unravel_index(iIter, (npdiam, nodiam, nthick))
ipdiam, iodiam, ithick = np.unravel_index(iIter, (npdiam, nodiam, nthick))
print(f'ipdiam: {ipdiam}, iodiam: {iodiam}, ithick: {ithick}')
print(f'pdiam: {kpdiam_t[ipdiam]}, odiam: {kodiam_t[iodiam]}, ithick: {kthick_t[ithick]}')

iIterbis = ithick+ iodiam*nthick + ipdiam*nthick*nodiam
print(f'iIter: {iIter}, iIterbis: {iIterbis}')

ipdiam, iodiam, ithick, iMask1 = np.unravel_index(EE_D_t.argmin(), (npdiam, nodiam, nthick, nMask1))
print(f'pdiam: {kpdiam_t[ipdiam]}, odiam: {kodiam_t[iodiam]}, thick: {kthick_t[ithick]}, rMask1: {rMask1_t[iMask1]}')

#%%
rMask1 = rMask1_t[iMask1]

wv1 = rMask_m/(rMask1*Fratio)
print(f'rMask1: {rMask1}')
print(f'wv1 = {wv1}m = {wv1/wv0_J:.3f}wv0_J')

pdiam2 = pdiam2_t[ipdiam]
odiam2 = odiam2_t[iodiam]
thick2 = thick2_t[ithick]

print(f'pdiam2: {pdiam2}')
print(f'odiam2: {odiam2}')
print(f'thick2: {thick2}')


#%%
"""
### Read optimal apodizer file
"""
fname_apo = f'pupilsbr_nPup{nPup}_pdiam{int(np.round(pdiam*100))}_odiam{int(np.round(odiam*100))}_thick{int(np.round(thick_apod*100)):03d}_Apod_rMask{int(np.round(rMask1*100)):03d}.fits'
fpath_apo = fdir / fname_apo
Apod2d = fits.getdata(fpath_apo)

fname_lys_opt = f'pupilsbr_nPup{nPup}_kpdiam{int(np.round(kpdiam_t[ipdiam]*100)):03d}_kodiam{int(np.round(kodiam_t[iodiam]*100)):03d}_kthick{int(np.round(kthick_t[ithick]*100)):03d}.fits' 
fpath_lys_opt = fdir / fname_lys_opt
LyotStop2d_opt = fits.getdata(fpath_lys_opt)


#%%
"""
### Display optimal apodizer file
"""
pl.figure(0)
pl.clf()
pl.subplot(131)
pl.imshow(Pupil2d)
pl.subplot(132)
pl.imshow(Apod2d)
pl.subplot(133)
pl.imshow(LyotStop2d_opt)

#%%
"""
### Array index
""" 
EE_D2_t = np.reshape(EE_D_t, (npdiam, nodiam, nthick, nMask1))

kpdiam0 = 0.92#0.92#0.91#0.92
kodiam0 = 1.1#1.1#1.05#1.25
kthick0 = 1.0#1.0#1.0
rMask10 = 2.87#2.65#2.87#2.86#2.64



EE_D20_t = EE_D2_t[kpdiam_t == kpdiam0, kodiam_t == kodiam0, :, :].reshape(nthick, nMask1)
EE_D21_t = EE_D2_t[kpdiam_t == kpdiam0, :, kthick_t == kthick0, :].reshape(nodiam, nMask1)
EE_D22_t = EE_D2_t[:, kodiam_t == kodiam0, kthick_t == kthick0, :].reshape(npdiam, nMask1)
EE_D23_t = EE_D2_t[:, :, kthick_t == kthick0, np.abs(rMask1_t-rMask10) < 0.001].reshape(npdiam, nodiam)

#%%
"""
### Display preparation
"""
kpdiam_min = kpdiam_t.min()
kpdiam_max = kpdiam_t.max()

kodiam_min = kodiam_t.min()
kodiam_max = kodiam_t.max()

kthick_min = kthick_t.min()
kthick_max = kthick_t.max()

rMask1_min = rMask1_t.min()
rMask1_max = rMask1_t.max()

Z20 = EE_D20_t
extent20 = [kthick_min, kthick_max, rMask1_min, rMask1_max]
Z20_xlabel = r'thick oversize factor'
Z20_ylabel = r'Mask radius in $\lambda_0/D$'
Z20_text   = f'kpdiam = {kpdiam0:.2f}, kodiam = {kodiam0:.2f}'
Z20_x, Z20_y = np.unravel_index(EE_D20_t.argmin(), EE_D20_t.shape)
Z20_x = kthick_t[Z20_x]
Z20_y = rMask1_t[Z20_y]

Z21 = EE_D21_t
extent21 = [kodiam_min, kodiam_max, rMask1_min, rMask1_max]
Z21_xlabel = r'odiam diameter oversize factor'
Z21_ylabel = r'Mask radius in $\lambda_0/D$'
Z21_text   = f'kpdiam = {kpdiam0:.2f}, thick = {kthick0:.2f}'
Z21_x, Z21_y = np.unravel_index(EE_D21_t.argmin(), EE_D21_t.shape)
Z21_x = kodiam_t[Z21_x]
Z21_y = rMask1_t[Z21_y]

Z22 = EE_D22_t
extent22 = [kpdiam_min, kpdiam_max, rMask1_min, rMask1_max]
Z22_xlabel = r'pdiam oversize factor'
Z22_ylabel = r'Mask radius in $\lambda_0/D$'
Z22_text   = f'kodiam = {kodiam0:.2f}, thick = {kthick0:.2f}'
Z22_x, Z22_y = np.unravel_index(EE_D22_t.argmin(), EE_D22_t.shape)
Z22_x = kpdiam_t[Z22_x]
Z22_y = rMask1_t[Z22_y]

Z23 = EE_D23_t
extent23 = [kpdiam_min, kpdiam_max, kodiam_min, kodiam_max]
Z23_xlabel = r'pdiam oversize factor'
Z23_ylabel = r'odiam oversize factor'
Z23_text   = f'thick = {kthick0:.2f}, rMask = {rMask10:.2f} lam0/D'
Z23_x, Z23_y = np.unravel_index(EE_D23_t.argmin(), EE_D23_t.shape)
Z23_x = kpdiam_t[Z23_x]
Z23_y = kodiam_t[Z23_y]

#%%
"""
### Display C vs (thick, rMask)
"""
#fname_image_plane_f_disp = 'corono_poly_bw_sensitivity_contour_nPup={0}_gbsx.pdf'.format(nPup)
#fpath_image_plane_f_disp = fdir_pdf / fname_image_plane_f_disp

# line width parameter
lw0 = 2.5

ZZ_t = [Z20, Z21, Z22, Z23]
ZZ_x_t = [Z20_x, Z21_x, Z22_x, Z23_x]
ZZ_y_t = [Z20_y, Z21_y, Z22_y, Z23_y]

extentZZ = [extent20, extent21, extent22, extent23]
ZZ_xlabel = [Z20_xlabel, Z21_xlabel, Z22_xlabel, Z23_xlabel]
ZZ_ylabel = [Z20_ylabel, Z21_ylabel, Z22_ylabel, Z23_ylabel]
ZZ_text = [Z20_text, Z21_text, Z22_text, Z23_text]
nZZ = len(ZZ_t)

str_ZZ_t = ['thick_v_rMask', 'odiam_v_rMask', 'pdiam_v_rMask', 'pdiam_v_odiam']

for iZZ, ZZ in enumerate(ZZ_t):

    fname_image_plane_f_disp = 'corono_poly_bw_sensitivity_contour_nPup={0}_gbsx_'.format(nPup) + str_ZZ_t[iZZ] + str_num_mask + str_apod_spiders + '.pdf'
    fpath_image_plane_f_disp = fdir_pdf / fname_image_plane_f_disp
    

    f2 = pl.figure(60 + iZZ, figsize=(12,4.5))
    pl.clf()
    ax0 = f2.add_subplot(111)
    im = ax0.imshow(np.log10(ZZ.T), 
                    cmap = "inferno", 
                    vmin=-7., vmax=-3.,
                    extent = extentZZ[iZZ],
                    origin = 'lower',
                    )
    
    cs = ax0.contour(np.log10(ZZ.T), [-6.0, -5.0, -4.0], colors = 'k',
                extent = extentZZ[iZZ], linestyles = '-')
    ax0.clabel(cs, inline=1, fontsize=ftsz, fmt = '%1.1f')
    
    ax0.set_xlabel(ZZ_xlabel[iZZ])
    ax0.set_ylabel(ZZ_ylabel[iZZ])
    ax0.set_aspect('auto')
    #ax0.set_title(r'New APLC design')
    #exec('ax{0}.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")'.format(1,))
    #exec('ax{0}.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")'.format(1,))
    ax0.text(0.5, 0.95, ZZ_text[iZZ], fontsize=ftsz, 
              horizontalalignment="center", color = "white", transform=ax0.transAxes)
    
    ax0.text(ZZ_x_t[iZZ], ZZ_y_t[iZZ], 'x', fontsize=ftsz, 
              horizontalalignment="center", color = "white")
    
    # ax0.axvline(x=rMask, ymin=-12, ymax =2, linewidth=lw0, color='r', linestyle='--')
    # ax0.axvline(x=rho0, ymin=-12, ymax =2, linewidth=lw0, color='b', linestyle='--')
    ax0.axvline(x=ZZ_x_t[iZZ], linewidth=lw0, color='w', linestyle='--')
    
    # ax0.axhline(y=1.0, xmin=0., xmax =lam0D_max, 
    #             linewidth=lw0, color='g', linestyle=':')
    # ax0.axhline(y=lam_opt_min, xmin=0., xmax =lam0D_max, 
    #             linewidth=lw0, color='k', linestyle=':')
    ax0.axhline(y=ZZ_y_t[iZZ],
                 linewidth=lw0, color='w', linestyle='--')


    
    #ax0.autoscale(False)
    #ax0.set_xscale("log")
    
    # ax2 = ax0.twinx()
    # ax2.set_ylim(lam_min*lam02um, lam_max*lam02um)
    # ax2.set_ylabel(r'$\lambda$ in $\mu$m ($\lambda_0={0:.3f}\mu$m)'.format(wv0*1e6), 
    #                rotation=270, labelpad = ftsz)
    # #
    # ax3 = ax0.twiny()
    # ax3.set_xlim(lam0D_min*lam0D2mas, lam0D_max*lam0D2mas)
    # ##ax3.set_yticks()
    # ax3.set_xlabel(r'Angular separation in mas')
    # ax3.set_xscale("log")
    
    
    
    f2.subplots_adjust(bottom=0.13, top=0.87, left=0.1, right=0.75,
                        wspace=0.02, hspace=0.02)
    #
    #f2.subplots_adjust(right=0.8)
    cbar_ax = f2.add_axes([0.86, 0.15, 0.05, 0.7])
    cbar    = f2.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('1$\sigma$ intensity in log scale', rotation=270, labelpad = 16)
    if do_plot is True:
        pl.savefig(str(fpath_image_plane_f_disp), transparent=True)
    pl.tight_layout()
pl.show()

#%%
