#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:40:02 2018

@author: mndiaye
"""
import pylab as pl

from corono import coronagraph_cl as cg
from corono import optim_cl as op

#%% 
#class 
corono0 = cg.APLC1d()
#problem = op.MaxTau()
problem = op.MaxContrastL1()
A, b, c = problem.compute_matrices()

pl.figure(1)
pl.clf()
pl.imshow(A)

pl.show()