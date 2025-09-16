#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:35:58 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

# Initialization
import numpy as np
#import pyzelda.utils.aperture as aperture
from vigan.optics import aperture

import matplotlib.pyplot as plt
from pathlib import Path

from astropy.io import fits

import corono as coro

import xaosim

#%%
"""
### Parameters
"""
pupil_name = 'vlt_btw5'
do_dead_act = False
nPup= 800 #384
do_fits = True

do_zeropad = False
do_fits_zeropad = False
nArr = 400 #384

kwd_spiders = False

#%%
# =========================================================================
def _xyic(ys, xs, between_pix=False):
    ''' --------------------------------------------------------------
    Private utility: returns two arrays of (y, x) image coordinates

    Array values give the pixel coordinates relative to the center of
    the array, that can be offset by 0.5 pixel if between_pix is set
    to True

    Parameters:
    ----------
    - ys: (integer) vertical size of the array (in pixels)
    - xs: (integer) horizontal size of the array (in pixels)
    - between_pix: (boolean) places center of array between pixels

    Returns:
    -------
    A tuple of two arrays: (yy, xx) ... IN THAT ORDER!
    -------------------------------------------------------------- '''
    offset = 0
    if between_pix is True:
        offset = 0.5
    xx = np.outer(np.ones(ys), np.arange(xs)-xs//2+offset)
    yy = np.outer(np.arange(ys)-ys//2+offset, np.ones(xs))
    return (yy, xx)


# ==================================================================
def four_spider_mask_test(ys, xs, pix_rad, pdiam, odiam=0.0,
                     beta=45.0, thick=0.25, offset=0.0,
                     spiders=True, split=False, between_pix=True):
    ''' ---------------------------------------------------------
    tool function called by other routines to generate specific
    pupil geometries. Although the result is scaled by pix_rad in
    pixels, telescope specifics are provided in meters.

    Parameters:
    ----------
    - ys, xs  : dimensions of the 2D array      (in pixels)
    - pix_rad : radius of the circular aperture (in pixels)
    - pdiam   : diameter of the aperture        (in meters)
    - odiam   : diameter of the obstruction     (in meters)
    - beta    : angle of the spiders            (in degrees)
    - thick   : thickness of the spiders        (in meters)
    - offset  : spider intersect point distance (in meters)
    - spiders : flag to true to include spiders (boolean)
    - split   : split the mask into four parts  (boolean)
    --------------------------------------------------------- '''

    beta = beta * np.pi/180.0  # converted to radians
    ro = odiam / pdiam
    yy, xx = _xyic(ys, xs, between_pix=between_pix)
    mydist = np.hypot(yy, xx)

    thick *= pix_rad / pdiam
    offset *= pix_rad / pdiam

    x0 = thick/(2 * np.sin(beta)) + offset
    y0 = thick/(2 * np.cos(beta)) - offset * np.tan(beta)

    if spiders:
        # quadrants left - right
        a = ((xx >= x0) * (np.abs(np.arctan(yy/(xx-x0+1e-8))) < beta))
        b = ((xx <= -x0) * (np.abs(np.arctan(yy/(xx+x0+1e-8))) < beta))
        # quadrants up - down
        c = ((yy >= 0.0) * (np.abs(np.arctan((yy-y0)/(xx+1e-8))) > beta))
        d = ((yy < 0.0) * (np.abs(np.arctan((yy+y0)/(xx+1e-8))) > beta))

    # pupil outer and inner edge
    e = (mydist <= np.round(pix_rad))
    if odiam > 1e-3:  # threshold at 1 mm
        e *= (mydist > np.round(ro * pix_rad))
        
    # e = (mydist < pix_rad)
    # if odiam > 1e-3:  # threshold at 1 mm
    #     e *= (mydist > ro * pix_rad)
    # e1 = coro.utils.uniform_disk(ys, ys//2, CtrBtwnPix=True) - coro.utils.uniform_disk(ys, ro*ys//2, CtrBtwnPix=True) 
    # print(np.max(np.abs(e1-e)))
    
    # e2 = aperture.disc(ys, ys//2, cpix=False) - aperture.disc(ys, ro*ys//2, cpix=False)
    # print(np.max(np.abs(e2-e)))
    
    
    

    if split:
        res = np.array([a*e, b*e, c*e, d*e])
        return(res)

    if spiders:
        return((a+b+c+d)*e)
    else:
        return(e)


#%%
def make_VLT_pupil(
    pupdiam,
    centralobs=0.14,
    spiders=0.00625,
    spiders_bool=True,
    centralobs_bool=True,
):
    """
    Return a VLT pupil
    based on make_VLT function from shesha/shesha/util/make_pupil.py

    Args :
        pupdiam (int) [pixel] : pupil diameter

        centralobs (float, optional) [fraction of diameter] : central obtruction diameter, default = 0.14

        spiders (float, optional) [fraction of diameter] : spider diameter, default = 0.00625

        spiders_bool (bool, optional) : if False, return the VLT pupil without spiders; default = True

        centralobs_bool (bool, optional) : if False, return the VLT pupil without central obstruction; default = True

    Returns :
        VLT_pupil (2D array) : VLT transmission pupil of shape (pupdiam, pupdiam), filled with 0 and 1
    """
    range = 0.5 * (1) - 0.25 / pupdiam
    X = np.tile(np.linspace(-range, range, pupdiam, dtype=np.float32), (pupdiam, 1))
    R = np.sqrt(X**2 + (X.T) ** 2)

    if centralobs_bool:
        VLT_pupil = ((R < 0.5) & (R > (centralobs / 2))).astype(np.float32)
    else:
        VLT_pupil = (R < 0.5).astype(np.float32)

    if spiders_bool:
        angle = 50.5 * np.pi / 180.0  # 50.5 degrees = angle between spiders

        if pupdiam % 2 == 0:
            spiders_map = (
                (
                    (X.T > (X - centralobs / 2 + spiders / np.sin(angle)) * np.tan(angle))
                    + (X.T < (X - centralobs / 2) * np.tan(angle))
                )
                * (X > 0)
                * (X.T > 0)
            )
        elif pupdiam % 2 == 1:
            spiders_map = (
                (
                    (X.T > (X - centralobs / 2 + spiders / np.sin(angle)) * np.tan(angle))
                    + (X.T < (X - centralobs / 2) * np.tan(angle))
                )
                * (X >= 0)
                * (X.T >= 0)
            )
        spiders_map += np.fliplr(spiders_map)
        spiders_map += np.flipud(spiders_map)
        VLT_pupil = VLT_pupil * spiders_map

    return VLT_pupil


#%%
"""
### generation of a VLT like pupil
"""
x_deadact = [0.1534, -0.0984, -0.1963, 0.2766, 0.3297]
y_deadact = [-0.0768, -0.1240, -0.3542, -0.2799, -0.2799] 

if pupil_name == 'vlt':
    if do_dead_act:
        pupil = aperture.vlt_pupil(nPup, nPup, dead_actuators=[x_deadact,y_deadact],
        dead_actuator_diameter=0.025, cpix=False)
    else:
        pupil = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0.)
elif pupil_name == 'vlt_btw':
    # using data in Frantz'code
    pdiam, odiam = 8.00, 1.12
    thick = 0.04             # adopted spider thickness (meters)
    offset = 1.11            # spider intersection offset (meters)
    beta = 50.5              # spider angle beta


    
    # using data in Arthur's code and based on vlt pupil measurement with IRDIS (partial match for spider orientation)
    # pdiam, odiam = 8.00, 1.10*1.03
    # thick = 7*8./384             # adopted spider thickness (meters)
    # offset = 1.11            # spider intersection offset (meters)
    # beta = 50.5              # spider angle beta

    pupil = xaosim.pupil.four_spider_mask(nPup, nPup, nPup/2, 
                                          pdiam=pdiam, odiam=odiam,
                                          beta=beta, thick=thick, offset=offset,
                                          spiders=True,between_pix=True)

elif pupil_name == 'vlt_btw2':
    # using data in Frantz'code and a sightly enhanced spider strut (50cm) 
    pdiam, odiam = 8.00, 1.12
    thick = 0.05             # adopted spider thickness (meters)
    offset = 1.11            # spider intersection offset (meters)
    beta = 50.5              # spider angle beta    


    pupil = xaosim.pupil.four_spider_mask(nPup, nPup, nPup/2, 
                                          pdiam=pdiam, odiam=odiam,
                                          beta=beta, thick=thick, offset=offset,
                                          spiders=True,between_pix=True)

elif pupil_name == 'vlt_btw3':
    # using data in Frantz'code and a sightly enhanced spider strut (50cm) 
    pdiam, odiam = 8.00, 1.12
    thick = 0.10             # adopted spider thickness (meters)
    offset = 1.11            # spider intersection offset (meters)
    beta = 50.5              # spider angle beta    


    pupil = xaosim.pupil.four_spider_mask(nPup, nPup, nPup/2, 
                                          pdiam=pdiam, odiam=odiam,
                                          beta=beta, thick=thick, offset=offset,
                                          spiders=True,between_pix=True)
    
elif pupil_name == 'vlt_btw4':
    # using data in Frantz'code and a sightly enhanced spider strut (50cm) 
    pdiam, odiam = 8.000, 1.120
    thick = 0.100             # adopted spider thickness (meters)
    offset = 1.0545            # spider intersection offset (meters)
    beta = 50.5               # spider angle beta    


    pupil = xaosim.pupil.four_spider_mask(nPup, nPup, nPup/2, 
                                          pdiam=pdiam, odiam=odiam,
                                          beta=beta, thick=thick, offset=offset,
                                          spiders=True,between_pix=True)
    
elif pupil_name == 'vlt_btw5':
    # using data in Frantz'code and a sightly enhanced spider strut (50cm) 
    pdiam, odiam = 8.000, 1.12
    thick = 0.100             # adopted spider thickness (meters)
    offset = 1.0545            # spider intersection offset (meters)
    beta = 50.5               # spider angle beta    


    pupil = four_spider_mask_test(nPup, nPup, nPup/2, 
                                          pdiam=pdiam, odiam=odiam,
                                          beta=beta, thick=thick, offset=offset,
                                          spiders=kwd_spiders, between_pix=True)
    
elif pupil_name == 'sbr':
    pdiam, odiam = 7.92, 2.3
    thick = 0.25              # adopted spider thickness (meters)
    offset = 1.278            # spider intersection offset (meters)
    beta = 51.75              # spider angle beta

    pupil = xaosim.pupil.four_spider_mask(nPup, nPup, nPup/2, 
                                          pdiam=pdiam, odiam=odiam,
                                          beta=beta, thick=thick, offset=offset,
                                          spiders=True,between_pix=True)

#%%
"""
### display vlt-like pupil
"""
plt.figure(0)
plt.clf()
plt.imshow(pupil)

plt.show()

#%%
"""
save pupil
"""
fdir = Path('/Users/mndiaye/scratch/data/Coronagraphs/data/2D/pupils').resolve()
str_dead_act=''
if do_dead_act and pupil_name == 'vlt':
    str_dead_act = '_dead_act'

fname = f'pupil={pupil_name}_nPup={nPup}{str_dead_act}.fits'
fpath = fdir / fname
print(fname)

if do_fits:
    fits.writeto(fpath, pupil*1., overwrite=True)

#%%
if do_zeropad:
    pupilpad = np.zeros((nArr, nArr))
    nBeg = (nArr-nPup)//2
    nEnd = (nArr+nPup)//2
    pupilpad[nBeg:nEnd,nBeg:nEnd] = pupil*1.
    
    plt.figure(1)
    plt.clf()
    plt.imshow(pupilpad)

    plt.show()
    
    fname_pad = f'pupil={pupil_name}_nPup={nPup}{str_dead_act}_nArr={nArr}.fits'
    fpath_pad = fdir / fname_pad
    print(fname_pad)
    
    if do_fits_zeropad:
        fits.writeto(fpath_pad, pupilpad*1., overwrite=True)
    
#%%
"""
### pupil difference
"""
# pupil_true = aperture.vlt_pupil(nPup, nPup, dead_actuator_diameter=0., cpix=False)
# pupil_diff = pupil - pupil_true

# plt.figure(2)
# plt.clf()
# plt.imshow(pupil_diff)

#%%
"""
### Pupil difference with compass
"""
pupil_mnd = pupil*1
pupil_com = make_VLT_pupil(nPup,spiders_bool = kwd_spiders)

pupil_dif = pupil_com - pupil_mnd

plt.figure(3)
plt.clf()
plt.imshow(pupil_dif)

#%%
val_range = 0.5 * (1) - 0.5 / nPup
test_com = np.tile(np.linspace(-val_range, val_range, nPup, dtype=np.float32), (nPup, 1)) 

test_mnd_y, test_mnd_x = _xyic(nPup, nPup, between_pix=True)

plt.figure(4)
plt.clf()
plt.subplot(131)
plt.imshow(test_com*nPup)
plt.subplot(132)
plt.imshow(test_mnd_x)
plt.subplot(133)
plt.imshow(test_com*nPup - test_mnd_x)