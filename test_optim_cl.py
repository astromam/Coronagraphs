#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:40:02 2018

@author: mndiaye
"""
import pylab as pl
import gurobipy as gb

import numpy as np

from corono import coronagraph_cl as cg
from corono import optim_cl as op

#%% 
#class 
corono0 = cg.APLC1d()
#problem = op.MaxTau()
#problem = op.MaxContrastL1()
#problem = op.MaxContrastLinf()
problem = op.MaxContrast(Lnorm='L1')
A, b, c = problem.compute_matrices()

pl.figure(1)
pl.clf()
pl.imshow(abs(A)**0.25)

pl.show()

#%%
m = problem.compute_gurobi_model()

#%%
#try:   
#    m.Params.Method       = 2
#    m.Params.LogToConsole = 1
#    m.Params.Crossover    = 0
#    
#    m.optimize()
#
#    Apod = np.zeros((corono0.nPup))
#    Apodbis = np.zeros((problem.npp))
#    for i in range(problem.npp):
#        Apodbis[i] = m.getVars()[i].x
#    Apod[problem.idx_pup] = Apodbis
#        
#    test = np.zeros((corono0.nPup, 2))
#    test[:,0] = corono0.r
#    test[:,1] = Apod   
##    if fpath: write_apod1d(fpath, test)           
##    return Apod
#
#except gb.GurobiError as e:
#    print('Error code ' + str(e.errno) + ": " + str(e))
#
#except AttributeError:
#    print('Encountered an attribute error')

#%%

Apod = op.solve

#%%

'''
### Comparison with file generated with AMPL+gurobi
'''
pl.figure(6)
pl.clf()
pl.plot(Apod)
