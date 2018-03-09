#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:36:39 2018

@author: mndiaye
"""

#%% Initialization problem
import numpy as np
from corono import coronagraph_cl as cg

def get_default_params_matrix_pb():
    tmp = {'cDarkHole':8,'tau':0.2}
    return tmp

#%%
class MatrixProblem(object):

    default_params = get_default_params_matrix_pb()
    
    def __init__(self,corono=cg.APLC1d(),**kwargs):
    
        self.params  = kwargs
        self.corono  = corono
        
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
    def compute_matrices(self):
        print('Warning: ')

#%%
    def __contains__(self, item):
        '''
        to be written
        '''
        return item in self.params
    
#%%     
    def __getattr__(self, name):
        '''
        to be written
        '''
        return self.params[name]


#%%
class MaxTau(MatrixProblem):

#    default_params = get_default_params_matrix_pb()
    
    def __init__(self, corono=cg.APLC1d(), **kwargs):
        super().__init__(**kwargs)
    
    def compute_matrices(self):
        corono_field_t_re2 = np.reshape(self.corono_field_t_re[:,:,self.idx_dz], (self.corono.nPup, self.corono.nlam*self.ndz))

        direct_field_t_re2 = np.zeros_like(corono_field_t_re2)
        for j in range(self.corono.nlam*self.ndz):
            direct_field_t_re2[:,j] = self.direct_field_t_re[:,(self.corono.nlam-1)//2,0]
        
        A0  =  corono_field_t_re2 - 10**(-self.cDarkHole/2)/np.sqrt(2.)*direct_field_t_re2
        A1  = -corono_field_t_re2 - 10**(-self.cDarkHole/2)/np.sqrt(2.)*direct_field_t_re2
        A2  = -np.identity(self.corono.nPup)
        A3  =  np.identity(self.corono.nPup)
    
        b0  = np.zeros((self.corono.nlam*self.ndz))
        b1  = np.zeros((self.corono.nlam*self.ndz))
        b2  = np.zeros(self.corono.nPup)
        b3  = np.ones(self.corono.nPup)
    
        self.A = np.concatenate((A0,A1,A2,A3), axis=1)
        self.b = np.concatenate((b0,b1,b2,b3))
        self.c = - 2.*np.pi*(np.arange(self.corono.nPup)+0.5)/(2.*self.corono.nPup)**2/self.TR
        
        return self.A, self.b, self.c

#%% 
#class 
corono0 = cg.APLC1d()
max_tau_pb = MaxTau()
A, b, c = max_tau_pb.compute_matrices()
