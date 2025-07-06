#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:00:39 2025

@author: mndiaye
"""

"""
### Initialization
"""
import numpy as np
from pathlib import Path
import imageio as iio
from pygifsicle import optimize

#%%
"""
### Parameters
"""
# vector of angular separation for the computation of planet transmission
sep_mas_min  = 0.0
sep_mas_max  = 62.5 #Fmax2dbis_t[0]//2-1.
sep_mas_stp  = 0.125  #Fmax2dbis_t[0]/nImg2dbis #0.5

# array of angular separations
nsep     = int(round(1+(sep_mas_max-sep_mas_min)/sep_mas_stp))
duration = 10.

#%%
"""
### Directory and filenames
"""

fdir_mov = Path('/Users/mndiaye/scratch/data/andes/movie_frames/').resolve()

#%%
"""
### Make gif file
"""
frames = np.stack([iio.v3.imread(fdir_mov / f'frame_{isep:04d}.png') for isep in 4*np.arange(nsep//4)], axis = 0)
iio.mimwrite(fdir_mov / 'movie_planet.gif', frames, fps=(nsep/4)/duration)


#%%
# fileList = [ fdir_mov / f'frame_{isep:04d}.png' for isep in 4*np.arange(nsep//4)]

# writer = iio.get_writer(fdir_mov / 'movie_planet.mp4')

# for im in fileList:
#     writer.append_data(iio.imread(im))
# writer.close()

#%%
"""
### Optimize gif file
"""
# optimize(fdir_mov / 'movie_planet.gif' , fdir_mov / "movie_planet_optimized.gif") # For creating a new one