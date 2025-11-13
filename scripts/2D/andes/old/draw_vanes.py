# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:35:17 2024

@author: asp

to compute mask that is six arms with equal width

"""

import numpy as np
# import matplotlib.pyplot as plt

def six_arms(nPup, width):
    
    hlf = width/2.
    x = np.linspace(0,nPup-1,nPup)
    y = np.linspace(0,nPup-1,nPup)
    X, Y = np.meshgrid(x, y, copy=False)
    
    arms = np.ones((nPup,nPup))
    slp = np.tan(np.pi/6.)
    oorg = nPup*(1. - np.tan(np.pi/6.))/2.

    dist = np.abs(Y - X*slp - oorg)/np.sqrt(1+slp*slp)
    ig = np.where(dist <= hlf)
    arms[:][ig] = 0.

    oorg = nPup*(1. + np.tan(np.pi/6.))/2.
    dist = np.abs(Y + X*slp - oorg)/np.sqrt(1+slp*slp)
    ig = np.where(dist <= hlf)
    arms[:][ig] = 0.

    ig = np.where(np.abs(X-nPup/2) <= hlf)
    arms[:][ig] = 0.
    
    # plt.imshow(arms)
    
    return arms


def six_petals(nPup):
    
    x = np.linspace(-nPup//2,nPup//2+1,nPup)
    y = np.linspace(-nPup//2,nPup//2+1,nPup)
    X, Y = np.meshgrid(x, y, copy=False)

    # theta = np.arctan(Y/(X+np.sqrt(X**2. + y**2.)))*2. + np.pi/2.
    theta = np.angle(Y+1j*X)
    petals = np.zeros((6,nPup,nPup))

    th_lim = np.linspace(-np.pi,np.pi,7)
    for i in range(6):
    
        th_thr = (th_lim[i+1]+th_lim[i])/2.
        petals[i,:] = np.where(np.abs(theta - th_thr) < np.pi/6.,1.,0.)
    
    return petals

