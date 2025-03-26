#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:29:47 2025

@author: mndiaye
"""

"""
### Initialization
"""
import numpy as np
import matplotlib.pyplot as plt
# to set the font size to 16 in the plots
plt.rcParams.update({'font.size': 16})

from astropy.io import fits
from pathlib import Path

#%%
"""
### Parameters
"""
# array of wavelength in m (provided by Alain)
wvl_t = np.array([9.60e-07, 1.04e-06, 1.12e-06, 1.20e-06, 1.28e-06, 1.36e-06,
                  1.44e-06, 1.52e-06, 1.60e-06, 1.68e-06, 1.76e-06, 1.84e-06,
                  1.92e-06, 2.00e-06, 2.08e-06, 2.16e-06, 2.24e-06, 2.32e-06,
                  2.40e-06])

# array of wavelength in um
wvl_um_t = wvl_t*1e6

# number of wavelengths (should be consistent for all the cubes)
nlam = np.shape(wvl_um_t)[0]

#%%
"""
### Parameters for the plots
"""
# define a set of colors for the intensity curves
colors = plt.cm.rainbow(np.linspace(0,1,nlam))

# define angular separation range for the plot
xlim_min0 = -0.1
xlim_max0 = 60.1 

# define intensity range in log scale for the plot
ylim_min0 = -5.1
ylim_max0 = -0.9 

# selected wavelengths for the plot
lam_t = [0, 4, 9, 14, 18] 


#%%
"""
### Working Directory
"""
# directory of the cubes
fdir  = Path('/Users/mndiaye/scratch/data/andes/results/corono_profiles_alain/').resolve()

#%%
"""
### Filenames and filepaths
"""
# filename for the intensity profiles with Lyot coronagraph in JQ1 conditions 
fname_lyotcoro_jq1 = 'ao_corr_lyotCoro_JQ1.fits'
# filename for the intensity profiles with Lyot coronagraph in JQM conditions 
fname_lyotcoro_jqm = 'ao_corr_lyotCoro_JQM.fits'
# filename for the intensity profiles with the ideal coronagraph in JQ1 conditions
fname_idealcoro_jq1 = 'ao_corr_2ndOrderIdealCoro_profile_JQ1.fits'
# filename for the intensity profiles with the ideal coronagraph in JQ1 conditions (improved by a facotr of 2)
fname_idealcoro_jq1half = 'ao_corr_2ndOrderIdealCoro_profile_halfJQ1.fits' 

# filepath corresponding to the filenames given above
fpath_lyotcoro_jq1 = fdir / fname_lyotcoro_jq1
fpath_lyotcoro_jqm = fdir / fname_lyotcoro_jqm
fpath_idealcoro_jq1 = fdir / fname_idealcoro_jq1
fpath_idealcoro_jq1half = fdir / fname_idealcoro_jq1half

#%%
"""
### readfiles
"""
# cube for the intensity profiles with Lyot coronagraph in JQ1 conditions
lyotcoro_jq1 = fits.getdata(fpath_lyotcoro_jq1)
# cube for the intensity profiles with Lyot coronagraph in JQM conditions 
lyotcoro_jqm = fits.getdata(fpath_lyotcoro_jqm)
# cube for the intensity profiles with the ideal coronagraph in JQ1 conditions
idealcoro_jq1 = fits.getdata(fpath_idealcoro_jq1)
# cube for the intensity profiles with the ideal coronagraph in JQ1 conditions (improved by a facotr of 2)
idealcoro_jq1half = fits.getdata(fpath_idealcoro_jq1half)

#%%
"""
### parameter retrieval
"""
# number of points of the coronagraphic profile
nImg = np.shape(lyotcoro_jq1)[2]

# vector of the angular separation in mas (should be consistent for all the cubes)
x_mas = lyotcoro_jq1[0, 0]

#%%
"""
### Display plots
"""
#ugly plot with the 19 wavelengths
plt.figure(0, (8,4.5))
plt.clf()
for ilam in range(nlam):
    plt.plot(x_mas, np.log10(lyotcoro_jq1[ilam, 1, :]), color=colors[ilam],
             label=f'{wvl_um_t[ilam]:.3f}$\mu$m')
plt.xlabel(r'Angular separation [mas]')
plt.ylabel('Normalized intensity in log scale')
plt.ylim(xlim_min0, xlim_max0)
plt.ylim(ylim_min0, ylim_max0)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.tight_layout()

#%%
#ok plot with 5 wavelengths for Lyot coronagraph in JQ1 
plt.figure(10, (8,4.5))
plt.clf()
for ilam in lam_t:
    plt.plot(x_mas, np.log10(lyotcoro_jq1[ilam, 1, :]), color=colors[ilam],
             label=f'{wvl_um_t[ilam]:.3f}$\mu$m')
plt.xlabel(r'Angular separation [mas]')
plt.ylabel('Normalized intensity in log scale')
plt.title('Lyot coronagraph in JQ1 conditions')
plt.ylim(xlim_min0, xlim_max0)
plt.ylim(ylim_min0, ylim_max0)
plt.legend(ncol=1)
plt.tight_layout()

#%%
#ok plot with 5 wavelengths for Lyot coronagraph in JQM 
plt.figure(11, (8,4.5))
plt.clf()
for ilam in lam_t:
    plt.plot(x_mas, np.log10(lyotcoro_jqm[ilam, 1, :]), color=colors[ilam],
             label=f'{wvl_um_t[ilam]:.3f}$\mu$m')
plt.xlabel(r'Angular separation [mas]')
plt.ylabel('Normalized intensity in log scale')
plt.title('Lyot coronagraph in JQM conditions')
plt.ylim(xlim_min0, xlim_max0)
plt.ylim(ylim_min0, ylim_max0)
plt.legend(ncol=1)
plt.tight_layout()

#%%
#ok plot with 5 wavelengths for ideal coronagraph in JQ1 
plt.figure(12, (8,4.5))
plt.clf()
for ilam in lam_t:
    plt.plot(x_mas, np.log10(idealcoro_jq1[ilam, 1, :]), color=colors[ilam],
             label=f'{wvl_um_t[ilam]:.3f}$\mu$m')
plt.xlabel(r'Angular separation [mas]')
plt.ylabel('Normalized intensity in log scale')
plt.title('Ideal coronagraph in JQ1 conditions')
plt.ylim(xlim_min0, xlim_max0)
plt.ylim(ylim_min0, ylim_max0)
plt.legend(ncol=1)
plt.tight_layout()

#%%
#ok plot with 5 wavelengths for ideal coronagraph in JQ1 (improved by a factor of 2)
plt.figure(13, (8,4.5))
plt.clf()
for ilam in lam_t:
    plt.plot(x_mas, np.log10(idealcoro_jq1half[ilam, 1, :]), color=colors[ilam],
             label=f'{wvl_um_t[ilam]:.3f}$\mu$m')
plt.xlabel(r'Angular separation [mas]')
plt.ylabel('Normalized intensity in log scale')
plt.title('Ideal coronagraph in JQ1 conditions (improved by 2)')
plt.ylim(xlim_min0, xlim_max0)
plt.ylim(ylim_min0, ylim_max0)
plt.legend(ncol=1)
plt.tight_layout()