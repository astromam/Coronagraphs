# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:35:17 2024

@author: asp
"""

import numpy as np
# import matplotlib.pyplot as plt

def six_vanes(nPup, width):
    
    hlf = width/2.
    x = np.linspace(0,nPup-1,nPup)
    y = np.linspace(0,nPup-1,nPup)
    X, Y = np.meshgrid(x, y, copy=False)
    
    vanes = np.ones((nPup,nPup))
    slp = np.tan(np.pi/6.)
    oorg = nPup*(1. - np.tan(np.pi/6.))/2.

    dist = np.abs(Y - X*slp - oorg)/np.sqrt(1+slp*slp)
    ig = np.where(dist <= hlf)
    vanes[:][ig] = 0.

    oorg = nPup*(1. + np.tan(np.pi/6.))/2.
    dist = np.abs(Y + X*slp - oorg)/np.sqrt(1+slp*slp)
    ig = np.where(dist <= hlf)
    vanes[:][ig] = 0.

    ig = np.where(np.abs(X-nPup/2) <= hlf)
    vanes[:][ig] = 0.
    
    # plt.imshow(vanes)
    
    return vanes


