#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:28:48 2025

@author: mndiaye
"""

"""
### Initialization
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from matplotlib.patches import Circle
from uniform_disk import uniform_disk
import slow_fourier_transform as sft

from pathlib import Path

# import corono as coro
import time
# import os

from mpl_toolkits.axes_grid1 import AxesGrid

from scipy.optimize import curve_fit

# from astropy.io import fits


#%%
"""
### Parameters
"""
# Pupil size
nPup = 400

# Sampling of the coronagraph focal plane mask
nFPM = 50

# Image size
nImg = 400

# wavelength in m
#wv = 1600e-9 #1600e-9

#wv_t = np.array([960e-9, 1280e-9, 1440e-9, 1600e-9, 1760e-9])    #(900 + 100*np.arange(3))*1e-9 
wv_t = np.array([1000e-9, 1200e-9, 1400e-9, 1600e-9, 1800e-9])    #(900 + 100*np.arange(3))*1e-9 


nwv = np.shape(wv_t)[0]

# Pupil diameter in m 
D = 38.54

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# conversion lam/D to mas
#lamD2mas = (wv/D)*mas2rad

lamD2mas_t = (wv_t/D)*mas2rad

# plate scale in mas per pixel
pscale = 0.3

# FPM size in lam/D in the focal plane B
dMask_mas = 34.3 * 3.9 / 4.0
rMask_mas = dMask_mas/2.
#mB = rMask_mas/lamD2mas
#rMask = mB/2.

mB_t = dMask_mas/lamD2mas_t

# FoV in lam/D in the final image plane D
#mD = 120/lamD2mas #*(nImg/wv)
FoV_mas = 130
mD_t = FoV_mas/lamD2mas_t

# save fits 
do_sav_fits = True 

# save plots
do_sav_plt = True

nImg2dbis = nImg*1
#Fmax2dbis = mD*1 #nImg2dbis/(2*(wv/950e-9))

Fmax2dbis_t = mD_t*1.

values = range(nwv)
colors = plt.cm.rainbow(np.linspace(0,1,nwv))

# Sampling for the position of the planet
nstep_mas = 0.01
npts = 400
threshold = 0.001

y1=0
y2=rMask_mas

#%%
"""
### Directory and filenames
"""
user = 'Alain'
if user == 'Alain':
    fdir = Path("D:/Andes/Data_corono/data/").resolve()  # opd's seed value
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots
    fdir_mov = Path("D:/Andes/Data_corono/results/").resolve()
    
# fdir = Path('/Users/mndiaye/scratch/data/andes/data/Pupil/').resolve()

# # Directory for the OPD with the corresponding seed value
# fdir_res   = Path('/Users/mndiaye/scratch/data/andes/results/').resolve()

# fdir_mov = Path('/Users/mndiaye/scratch/data/andes/movie_frames/').resolve()
# if not os.path.exists(fdir_mov):
#     os.makedirs(fdir_mov)
#     print(fdir_mov)

# # Directory for the OPD with the corresponding seed value
# fdir_plt   = Path('/Users/mndiaye/scratch/data/andes/plots/').resolve()
# if not os.path.exists(fdir_plt):
#     os.makedirs(fdir_plt)
#     print(fdir_plt)


fname_pup = 'Pupil\ELT_pupil_400.fits'
# fname_lys = 'lyotStop_90_37_7.fits'

fpath_pup = fdir / fname_pup
# fpath_lys = fdir / fname_lys

#%%
"""
### Read file 
"""
Pupil2d = fits.getdata(fpath_pup)
# LyotStop2d = fits.getdata(fpath_lys)

diam = 0.89
obst = 0.35

LyotStop2d = Pupil2d * 0.

LyotStop2d = (Pupil2d.copy() * (uniform_disk(nPup, diam*nPup/2) -
                              uniform_disk(nPup, obst*nPup/2)))
            
#%%
"""
### Coronagraphic components
"""
# Focal plane mask
# mask2d = coro.utils.uniform_disk(nFPM, nFPM/2.)
mask2d = uniform_disk(nFPM, nFPM/2.)

#%%
"""
### Compute perfect telescope PSF at all the wavelengths
"""
norm_peakD_Tel_t = np.zeros((nwv))
Int_D_Tel_t_tmp = np.zeros((nwv, nImg, nImg))
Int_D_Tel_t = np.zeros((nwv, nImg, nImg))

for iwv in range(nwv):
    # Field in the entrance pupil plane A
    Fld_L_Tel = Pupil2d *1.
    
    # Field in the image plane D (no coronagraph)
    # Fld_D_Tel = coro.utils.sft(Fld_L_Tel, nImg, mD_t[iwv])
    Fld_D_Tel = sft.sft(Fld_L_Tel, nImg, mD_t[iwv])

    # Intensity 
    Int_D_Tel_t_tmp[iwv] = np.abs(Fld_D_Tel)**2
    
    # Normalized intensity in focal plane
    norm_peakD_Tel_t[iwv] = 1./np.max(Int_D_Tel_t_tmp[iwv])

Int_D_Tel_t = norm_peakD_Tel_t[:, None, None]*Int_D_Tel_t_tmp


#%%
"""
### Compute perfect LyotStop PSF including Lyot stop at all the wavelengths
"""
norm_peakD_Lys_t = np.zeros((nwv))
Int_D_Lys_t_tmp = np.zeros((nwv, nImg, nImg))
Int_D_Lys_t = np.zeros((nwv, nImg, nImg))


for iwv in range(nwv):
    # Field in the entrance pupil plane A
    Fld_L_Lys = Pupil2d * 1.*LyotStop2d
    
    # Field in the image plane D (no coronagraph)
    # Fld_D_Lys = coro.utils.sft(Fld_L_Lys, nImg, mD_t[iwv])
    Fld_D_Lys = sft.sft(Fld_L_Lys, nImg, mD_t[iwv])

    # Intensity 
    Int_D_Lys_t_tmp[iwv] = np.abs(Fld_D_Lys)**2
    
    # Normalized intensity in focal plane
    norm_peakD_Lys_t[iwv] = 1./np.max(Int_D_Lys_t_tmp[iwv])
    
Int_D_Lys_t = norm_peakD_Lys_t[:, None, None]*Int_D_Lys_t_tmp

#%%
"""
### Compute the coronagraph throughput (based on the energy in the relayed pupil plane)
"""
Int_L_Tel = np.abs(Pupil2d*1.)**2
Int_L_Lys = np.abs(Pupil2d*LyotStop2d)**2
    
EE_Tel = np.sum(Int_L_Tel)
norm_EE_Tel = 1./EE_Tel   

EE_Lys = np.sum(Int_L_Lys) 
norm_EE_Lys = 1./EE_Lys

coro_throughput = EE_Lys/EE_Tel

#%%
"""
### Compute the coronagraph throughput (based on the energy in the re-imaged focal plane) (remember that the FoV is finite)
"""
EE_Tel_bis = np.sum(Int_D_Tel_t_tmp[nwv//2])
EE_Lys_bis = np.sum(Int_D_Lys_t_tmp[nwv//2])

coro_throughput_bis = EE_Lys_bis/EE_Tel_bis


#%%
"""
### Compute perfect coronographic image at all the wavelengths
"""

Int_D_t = np.zeros((nwv, nImg, nImg))
for iwv in range(nwv):
    # Field in the entrance pupil plane A
    Fld_A = Pupil2d
    
    # focal plane B 
    # Fld_B = mask2d*coro.utils.sft(Fld_A, nFPM, mB_t[iwv])
    Fld_B = mask2d*sft.sft(Fld_A, nFPM, mB_t[iwv])

    # pupil plane C before Lyot stop
    # Fld_C = Fld_A - coro.utils.isft(Fld_B, nPup, mB_t[iwv])
    Fld_C = Fld_A - sft.isft(Fld_B, nPup, mB_t[iwv])

    # pupil plane C after Lyot stop
    Fld_L = Fld_C*LyotStop2d
    
    # image plane D 
    # Fld_D = coro.utils.sft(Fld_L, nImg, mD_t[iwv])
    Fld_D = sft.sft(Fld_L, nImg, mD_t[iwv])

    # Intensity
    Int_D = np.abs(Fld_D)**2
    
    # Normalized intensity
    Int_D_t[iwv] = norm_peakD_Lys_t[iwv]*Int_D

#%%
"""
### Planet image in monochormatic light
"""   
# vector of angular separation for the computation of planet transmission
sep_mas_min  = 0.0
sep_mas_max  = 62.5 #Fmax2dbis_t[0]//2-1.
sep_mas_stp  = 0.125  #Fmax2dbis_t[0]/nImg2dbis #0.5

# array of angular separations
nsep     = int(round(1+(sep_mas_max-sep_mas_min)/sep_mas_stp))
sep_mas_arr  = sep_mas_min + sep_mas_stp*np.arange(nsep)
print('{0:5d} sep'.format(nsep))

# array of angular distances in the final image plane
xx,yy   = np.meshgrid(np.arange(nPup)-nPup/2, np.arange(nPup)-nPup/2)
rr      = (2./float(nPup))*np.hypot(yy,xx)
theta   = np.arctan2(yy,xx)
Z       = 2.*rr*np.cos(theta)*Pupil2d  


#%%
"""
### Computation of the coronagraphic image of the planet
"""
EE_L_pla_t = np.zeros((nwv, nsep))
Int_D_pla_Tel_t = np.zeros((nwv, nsep, nImg2dbis, nImg2dbis))
Int_D_pla_Lys_t = np.zeros((nwv, nsep, nImg2dbis, nImg2dbis))

# index of the points inside the pupil
pup_idx = Pupil2d == 1.

for iwv in range(nwv):
    for isep in range(nsep):
        
        opd = (sep_mas_arr[isep]/lamD2mas_t[iwv])*wv_t[iwv]*(1./4.) * Z
        
        # Field in the entrance pupil plane A
        Fld_A_pla = Pupil2d * np.exp(1j*2.*np.pi*opd/wv_t[iwv])
        
        # focal plane B 
        # Fld_B_pla = mask2d*coro.utils.sft(Fld_A_pla, nFPM, mB_t[iwv])
        Fld_B_pla = mask2d*sft.sft(Fld_A_pla, nFPM, mB_t[iwv])

        # pupil plane C before Lyot stop
        # Fld_C_pla = Fld_A_pla - coro.utils.isft(Fld_B_pla, nPup, mB_t[iwv])
        Fld_C_pla = Fld_A_pla - sft.isft(Fld_B_pla, nPup, mB_t[iwv])

        # pupil plane C after Lyot stop
        Fld_L_pla = Fld_C_pla*LyotStop2d
        Int_L_pla = np.abs(Fld_L_pla)**2
        
        # pupil plane L after the Lyot stop
        EE_L_pla_t[iwv, isep] = np.sum(Int_L_pla[pup_idx])
        
        # image plane D 
        # Fld_D_pla = coro.utils.sft(Fld_L_pla, nImg, mD_t[iwv])
        Fld_D_pla = sft.sft(Fld_L_pla, nImg, mD_t[iwv])

        # Intensity
        Int_D_pla_tmp = np.abs(Fld_D_pla)**2

        # Normalized intensity vs telescope PSF 
        Int_D_pla_Tel = norm_peakD_Tel_t[iwv]*Int_D_pla_tmp
        
        # Normalized intensity
        Int_D_pla_Lys = norm_peakD_Lys_t[iwv]*Int_D_pla_tmp
        
        Int_D_pla_Tel_t[iwv, isep] = Int_D_pla_Tel        
        Int_D_pla_Lys_t[iwv, isep] = Int_D_pla_Lys

        
EE_L_pla_Lys_t = norm_EE_Lys*EE_L_pla_t
EE_L_pla_Tel_t = norm_EE_Tel*EE_L_pla_t



#%%
"""
### Save planet image vs angular separation for different wavelengths
"""
fname_Int_D_pla_Tel = 'andes_coro_planet_img_angsep_wvl_TelescopePSFnorm.fits'
fpath_Int_D_pla_Tel = fdir_res / 'planet_images' / fname_Int_D_pla_Tel

fname_Int_D_pla_Lys = 'andes_coro_planet_img_angsep_wvl_LyotstopPSFnorm.fits'
fpath_Int_D_pla_Lys = fdir_res / 'planet_images' / fname_Int_D_pla_Lys

#%%
"""
### Save files
"""
# Planet image with normlisation wrt Telescope pupil
fits.writeto(fpath_Int_D_pla_Tel, Int_D_pla_Tel_t, overwrite=True)
# Planet image with normlisation wrt Lyot Stop
fits.writeto(fpath_Int_D_pla_Lys, Int_D_pla_Lys_t, overwrite=True)

#%%
"""
### Eta_S and Eta_P computation
"""
# photometric aperture circular or ring radius in lam/D
photaper_rad = 1.0

val = 0

# array of angular distances in the final image plane
xxi,yyi  = np.meshgrid(np.arange(nImg2dbis)-nImg2dbis//2+val, np.arange(nImg2dbis)-nImg2dbis//2+val)

eta_P = np.zeros((nwv, nsep))
eta_P0 = np.zeros((nwv))
eta_P_norm = np.zeros((nwv, nsep))

print('eta_P computation')
t0 = time.time()
for iwv in range(nwv):
    mydist_t = (Fmax2dbis_t[iwv]/nImg2dbis)*np.hypot(yyi,xxi)    

    circ_aper = np.zeros((nImg2dbis, nImg2dbis))
    sum_circ_aper = np.zeros((nsep))

    for isep in range(nsep):    
        # compute averaged intensity of the planet at its location within a photmetric aperture of 0.7lam/D
        sep = (sep_mas_arr[isep]/lamD2mas_t[iwv])
        mydist_P = (Fmax2dbis_t[iwv]/nImg2dbis)*np.hypot(yyi,xxi-sep*nImg2dbis/Fmax2dbis_t[iwv])        
        ind_P = (mydist_P <= photaper_rad)
        circ_aper[ind_P] = 1.
        sum_circ_aper[isep] = np.sum(circ_aper)
        eta_P[iwv, isep] = np.sum(circ_aper*Int_D_pla_Tel_t[iwv, isep])/np.sum(circ_aper)

        if isep == 0:
            eta_P0[iwv] = np.sum(circ_aper*Int_D_Tel_t[iwv])/np.sum(circ_aper)
        circ_aper[ind_P] = 0.
    eta_P_norm[iwv] = eta_P[iwv]/eta_P0[iwv]

t1 = time.time()
print('computation time: {0}s'.format(t1-t0))


#%%
"""
### Display entrance and exit pupils
"""
plt.figure(0)
plt.clf()
plt.subplot(121)
plt.imshow(Pupil2d)
plt.title('ELT pupil')
plt.subplot(122)
plt.imshow(LyotStop2d)
plt.title('Lyot Stop')

#%%
"""
### Display images
"""
iwv0 = 0
isep0 = nsep//4


fname_images = 'test.pdf'
# filepath for the direct and coronagraphic images
fpath_images = fdir_plt / fname_images

# boundaries for the images in log scale
vmin0 = -5
vmax0 = 0

fig = plt.figure(1, figsize=(12,6))
plt.clf()

grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                axes_pad=0.3,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.2
                )

# Perfect PSF
im = grid[0].imshow(np.log10(Int_D_Lys_t[iwv0]), vmin=vmin0, vmax=vmax0, cmap='inferno')
grid[0].set_title('perfect PSF')

# Perfect coronagraphic image
im = grid[1].imshow(np.log10(Int_D_t[iwv0]), vmin=vmin0, vmax=vmax0, cmap='inferno')
grid[1].set_title('Perfect coro. image')

# Perfect PSF
# im = grid[2].imshow(np.log10(Int_DD0_t[0]), vmin=vmin0, vmax=vmax0, cmap='inferno')
# grid[0].set_title('perfect PSF')

# Perfect coronagraphic image
im = grid[3].imshow(np.log10(Int_D_pla_Lys_t[iwv0,isep0]), vmin=vmin0, vmax=vmax0, cmap='inferno')
grid[1].set_title('Shifted coro. image')

# colorbar
cbar = grid[0].cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Intensity in log scale', rotation=270)

plt.tight_layout()
if do_sav_plt:
    plt.savefig(fpath_images)

plt.show()

#%%
"""
### Eta_p plot from the focal plane
"""

fname_eta_p_abso_finite_pdf = f'eta_p_photaper_diam_{2*photaper_rad}lamD.pdf'
fname_eta_p_abso_finite_png = f'eta_p_photaper_diam_{2*photaper_rad}lamD.png'
fpath_eta_p_abso_finite_pdf = fdir_plt / fname_eta_p_abso_finite_pdf
fpath_eta_p_abso_finite_png = fdir_plt / fname_eta_p_abso_finite_png

sep_lim = 55

plt.figure(40, (8, 4.5))
plt.clf()
for iwv in range(nwv):
    plt.plot(sep_mas_arr[sep_mas_arr < sep_lim], eta_P_norm[iwv, sep_mas_arr < sep_lim], label=f'$\lambda$={int(np.round(wv_t[iwv]*1e9)):4d}nm', color=colors[iwv])
#    plt.axvline(x=IWA_mas_t[iwv], ymin=0, ymax =1, linewidth=1, linestyle='--', color=colors[iwv])
#    plt.axhline(y=0.5, xmin=0, xmax = np.max(sep_mas_arr), linewidth=1, color='k', linestyle='--')
plt.xlabel('Angular separation in mas')
plt.ylabel(r'Planet throughput $\eta_P$')
plt.xlim(-1, 66)
plt.ylim(-0.02, 1.02)
#plt.ylim(-3, 0.5)
plt.title(f'{2*photaper_rad}$\lambda$/D-diameter photometric aperture')

plt.fill_between(sep_mas_arr, y1, y2, where=(sep_mas_arr <= y2), color='grey', alpha=0.3)
plt.grid()
plt.legend(loc=4, fontsize=14)
plt.tight_layout()
if do_sav_plt is True:
    plt.savefig(fpath_eta_p_abso_finite_pdf, transparent=True)
    plt.savefig(fpath_eta_p_abso_finite_png, transparent=True)
    
#%%
"""
### polynomial fit
"""
def poly1(X, a, b):
    return a*X + b

def poly3(X, a, b, c, d):
    return a*X**3 + b*X**2 + c*X + d


IWA3_mas_t = np.zeros((nwv))
for iwv in range(nwv):
    idx = np.where(np.abs(sep_mas_arr - rMask_mas) <= 2.)
    xdata = sep_mas_arr[idx[0]]
    ydata = EE_L_pla_Lys_t[iwv, idx[0]]
    
    #p_ini1 = [1., 1.]
    p_ini3 = [1., 1., 1., 1.]
    
    #p_opt1, p_cov1 = curve_fit(poly1, xdata, ydata, p0=pini1)
    p_opt3, p_cov3 = curve_fit(poly3, xdata, ydata, p0=p_ini3)
    xdata3 = min(sep_mas_arr[idx[0]])+ nstep_mas*np.arange(npts) 
    
    #poly1fitted = poly1(xdata, popt1[0], popt1[1])
    poly3fitted = poly3(xdata3, p_opt3[0], p_opt3[1], p_opt3[2], p_opt3[3])
    
    idx3 = np.where(np.abs(poly3fitted-0.5) <= threshold)
    IWA3_mas_t[iwv] = xdata3[idx3[0][0]]
    print(f'IWA3={IWA3_mas_t[iwv]:.2f}mas')

#%%
# plt.figure(60)
# plt.clf()
# plt.plot(xdata, ydata, label='data')
# #plt.plot(xdata, poly1fitted)
# plt.plot(xdata3, poly3fitted, label='fit', ls='--')
# plt.xlabel('Angular separation in mas')
# plt.ylabel(r'Normalized planet throughput $\eta_P$')
# plt.axvline(x=IWA3_mas_t[iwv], ymin=0, ymax =1, linewidth=1, linestyle='--', color=colors[iwv])
# plt.axhline(y=0.5, xmin=0, xmax = np.max(sep_mas_arr), linewidth=1, color='k', linestyle='--')
# plt.legend()



#%%
"""
### Eta_p plot from the pupil plane (normalized)
"""
fname_eta_p_norm_pdf = 'eta_p_photaper_diam_infinite_norm.pdf'
fname_eta_p_norm_png = 'eta_p_photaper_diam_infinite_norm.png'
fpath_eta_p_norm_pdf = fdir_plt / fname_eta_p_norm_pdf
fpath_eta_p_norm_png = fdir_plt / fname_eta_p_norm_png

plt.figure(50, (8, 4.5))
plt.clf()
for iwv in range(nwv):
    plt.plot(sep_mas_arr, EE_L_pla_Lys_t[iwv], label=f'$\lambda$={int(np.round(wv_t[iwv]*1e9)):4d}nm', color=colors[iwv])
    plt.axvline(x=IWA3_mas_t[iwv], ymin=0, ymax =1, linewidth=1, linestyle='--', color=colors[iwv])
    plt.axhline(y=0.5, xmin=0, xmax = np.max(sep_mas_arr), linewidth=1, color='k', linestyle='--')
plt.xlabel('Angular separation in mas')
plt.ylabel(r'Normalized planet throughput $\eta_P$')
plt.xlim(-1, 66)
plt.ylim(-0.02, 1.02)
plt.title('Infinite-size photometric aperture')
plt.text(0, 0.52, '0.5')
plt.text(IWA3_mas_t[nwv//2]+0.2, 0.025, 'IWA')
#plt.ylim(-3, 0.5)

plt.fill_between(sep_mas_arr, y1, y2, where=(sep_mas_arr <= y2), color='grey', alpha=0.3)
plt.grid()
plt.legend(loc=4, fontsize=14)
plt.tight_layout()
if do_sav_plt is True:
    plt.savefig(fpath_eta_p_norm_pdf, transparent=True)
    plt.savefig(fpath_eta_p_norm_png, transparent=True)
    
#%%
"""
### Eta_p plot from the pupil plane (not normalized)
"""
fname_eta_p_abso_pdf = 'eta_p_photaper_diam_infinite_abso.pdf'
fname_eta_p_abso_png = 'eta_p_photaper_diam_infinite_abso.png'
fpath_eta_p_abso_pdf = fdir_plt / fname_eta_p_abso_pdf
fpath_eta_p_abso_png = fdir_plt / fname_eta_p_abso_png

plt.figure(51, (8, 4.5))
plt.clf()
for iwv in range(nwv):
    plt.plot(sep_mas_arr, EE_L_pla_Tel_t[iwv], label=f'$\lambda$={int(np.round(wv_t[iwv]*1e9)):4d}nm', color=colors[iwv])
    plt.axvline(x=IWA3_mas_t[iwv], ymin=0, ymax =1, linewidth=1, linestyle='--', color=colors[iwv])
    plt.axhline(y=0.5*coro_throughput, xmin=0, xmax = np.max(sep_mas_arr), linewidth=1, color='k', linestyle='--')
    plt.axhline(y=coro_throughput, xmin=0, xmax = np.max(sep_mas_arr), linewidth=1, color='k', linestyle='--')
plt.xlabel('Angular separation in mas')
plt.ylabel(r'Planet throughput $\eta_P$')
plt.xlim(-1, 66)
plt.ylim(-0.02, 1.02)
plt.title('Infinite-size photometric aperture')
#plt.title('Coronagraph configuration for YJH band')
#plt.text(0, 0.5*coro_throughput, '50%')
plt.text(0, coro_throughput+0.02, f'Coronagraph {coro_throughput*100:.1f}%')
plt.text(IWA3_mas_t[nwv//2]+0.2, 0.025, 'IWA')
#plt.ylim(-3, 0.5)

plt.fill_between(sep_mas_arr, y1, y2, where=(sep_mas_arr <= y2), color='grey', alpha=0.3)
plt.grid()
plt.legend(loc=4, fontsize=14)
plt.tight_layout()
if do_sav_plt is True:
    plt.savefig(fpath_eta_p_abso_pdf, transparent=True)
    plt.savefig(fpath_eta_p_abso_png, transparent=True)
    
#%%
iwv0 = 3


for isep in range(nsep):
    
        fname_movie_frame = f'frame_{isep:04d}.png'
        # filepath for the direct and coronagraphic images
        fpath_movie_frame = fdir_mov / fname_movie_frame
        
        nticks = 5
        ticks_pix_arr= nImg*(np.arange(nticks)/nticks)
        ticks_mas_arr= FoV_mas*(np.arange(nticks)-(nticks//2))/(nticks//2)
        ticks_mas_arr_str = [str(x) for x in ticks_mas_arr]
        
        # boundaries for the images in log scale
        vmin0 = -5
        vmax0 = 0
        
        fig = plt.figure(2, figsize=(6,5))
        plt.clf()
        
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(1,1),
                        axes_pad=0.3,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.2
                        )
        
        # Perfect coronagraphic image
        im = grid[0].imshow(np.log10(Int_D_pla_Lys_t[iwv0,isep]), vmin=vmin0, vmax=vmax0, 
                            extent=[-FoV_mas/2, FoV_mas/2., -FoV_mas/2, FoV_mas/2.], cmap='inferno')
        
        Drawing_colored_circle = Circle( (0., 0.), rMask_mas,
                                      fill = False, color='w', ls='--', linewidth=2)
        
        #ax.set_aspect( 1 )
        grid[0].add_artist( Drawing_colored_circle )
        grid[0].set_title(f'$\lambda$={wv_t[iwv0]*1e6:.3f}$\mu$m - Object offset: {sep_mas_arr[isep]:4.1f} mas\n Total flux: {100*EE_L_pla_Tel_t[iwv0,isep]:4.1f} %')
        grid[0].set_xlabel('Angular separation [mas]')
        grid[0].set_ylabel('Angular separation [mas]')
        grid[0].locator_params(axis='both', nbins=6)
        
        # colorbar
        cbar = grid[0].cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Intensity in log scale', rotation=270)
        
        plt.tight_layout()
        if do_sav_plt:
            plt.savefig(fpath_movie_frame, dpi=72)
    
plt.show()