#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:36:39 2018

@author: mndiaye
"""

#%% Initialization problem
import numpy as np
from corono import coronagraph_cl as cg

#%%
class MatrixProblem(object):

    def __init__(self,corono=cg.APLC1d(),model_type='maxtau',
                **kwargs):
    
        self.model_type = model_type
        self.corono     = corono
        
        self.dz      = (self.corono.xi >= self.corono.rho0) & (self.corono.xi <= self.corono.rho1)
        self.aaa     = np.arange(self.corono.nImg+1) 
        self.idx_dz  = list(self.aaa[self.dz])
        self.ndz     = len(self.idx_dz)
    
        self.pup     = (self.corono.Pupil > 0.)
        self.bbb     = np.arange(self.corono.nPup)
        self.idx_pup = list(self.bbb[self.pup])
    
        self.lys     = (self.corono.LyotStop > 0.)
        self.idx_lys = list(self.bbb[self.lys]) 
    
        self.direct_field_t_re, self.direct_field_t_im = self.corono.prop_direct_matrix()
        self.corono_field_t_re, self.corono_field_t_im = self.corono.prop_corono_matrix()
        
        self.TR = np.sum(2.*np.pi*self.corono.Pupil*np.linspace(0.5,self.corono.nPup+0.5,num=self.corono.nPup)/(2.*self.corono.nPup)**2) 

#%%
class MaxTau(MatrixProblem):

    def __init__(self, **kwargs):
        '''
        to be written
        '''      
        super().__init__(**kwargs)
    
    def compute_matrices(self):
        corono_field_t2 = np.reshape(self.corono_field_t_re[:,:,self.idx_dz], (self.aplc.nPup, self.aplc.nlam*self.ndz))

        direct_field_t2 = np.zeros_like(corono_field_t2)
        for j in range(self.aplc.nlam*self.ndz):
            direct_field_t2[:,j] = self.direct_field_t[:,(self.aplc.nlam-1)//2,0]
        
        A0  =  corono_field_t2 - 10**(-self.aplc.cDarkHole/2)/np.sqrt(2.)*direct_field_t2
        A1  = -corono_field_t2 - 10**(-self.aplc.cDarkHole/2)/np.sqrt(2.)*direct_field_t2
        A2  = -np.identity(self.aplc.nPup)
        A3  =  np.identity(self.aplc.nPup)
    
        b0  = np.zeros((self.aplc.nlam*self.ndz))
        b1  = np.zeros((self.aplc.nlam*self.ndz))
        b2  = np.zeros(self.aplc.nPup)
        b3  = np.ones(self.aplc.nPup)
    
        self.A = np.concatenate((A0,A1,A2,A3), axis=1)
        self.b = np.concatenate((b0,b1,b2,b3))
        self.c = - 2.*np.pi*(np.arange(self.aplc.nPup)+0.5)/(2.*self.aplc.nPup)**2/self.TR
        
        return self.A, self.b, self.c

#%% 
class 