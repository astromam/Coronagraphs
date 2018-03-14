#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:36:39 2018

@author: mndiaye
"""

#%% Initialization problem
import numpy as np
import json
import gurobipy as gb
from corono import coronagraph_cl as cg

def get_default_params_matrix_pb():
    tmp = {'cDarkHole':8,'tau':0.2}
    return tmp

def get_default_params_MaxContrast_pb():
    tmp = get_default_params_matrix_pb()
    tmp.update({'Lnorm':'L1'})
    return tmp

#%%
class ProblemMatrix(object):

    default_params = get_default_params_matrix_pb()

    
    def __init__(self,corono=cg.APLC1d(),**kwargs):
    
        self.params  = kwargs
        self.check_params()
        
        self.corono  = corono
        
        self.dz      = (self.corono.xi >= self.corono.rho0) \
                & (self.corono.xi <= self.corono.rho1)
        self.aaa     = np.arange(self.corono.nImg+1) 
        self.idx_dz  = list(self.aaa[self.dz])
        self.ndz     = len(self.idx_dz)
    
        self.pup     = (self.corono.Pupil > 0.)
        self.bbb     = np.arange(self.corono.nPup)
        self.idx_pup = list(self.bbb[self.pup])
        self.npp     = len(self.idx_pup)
    
        self.lys     = (self.corono.LyotStop > 0.)
        self.idx_lys = list(self.bbb[self.lys]) 
    
        self.direct_field_t_re, self.direct_field_t_im = \
                self.corono.prop_direct_matrix()
        self.corono_field_t_re, self.corono_field_t_im = \
                self.corono.prop_corono_matrix()
        
        self.corono_field_t2_re = np.reshape(
                self.corono_field_t_re[:,:,self.idx_dz], 
                (self.corono.nPup, self.corono.nlam*self.ndz))[self.idx_pup,:]
        
        self.A = None
        self.b = None
        self.c = None
        
        self.TR = np.sum(2.*np.pi*self.corono.Pupil *np.linspace(
                0.5,self.corono.nPup+0.5,num=self.corono.nPup)\
                /(2.*self.corono.nPup)**2) 

#%%    
    def compute_matrices(self):
        print('Warning: virtual fct - no A, b and c matrices will be computed')

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
    def check_params(self):
        '''
        to be written
        '''        
        for key in self.default_params:
            if not key in self.params:
                self.params[key] = self.default_params[key]
  
#%%    
    def load_params(self, fname):
        '''
        load params from fname using JSON (JavaScript Object Notation)
        ----
        '''
        
        f=open(fname,'r')
        params=json.loads(f.read())
        self.__init__(**params)
        f.close()              
        
#%%
class MaxTau(ProblemMatrix):

    def __init__(self, corono=cg.APLC1d(), **kwargs):
        super().__init__(**kwargs)
    
    def compute_matrices(self):
               
        direct_field_t1_re = np.zeros_like(self.corono_field_t1_re)
        for j in range(self.corono.nlam*self.ndz):
            direct_field_t1_re[:,j] = \
            self.direct_field_t_re[:,(self.corono.nlam-1)//2,0]
        direct_field_t2_re = self.corono_field_t1_re[self.idx_pup,:]
        
        A0  =  self.corono_field_t2_re \
                - 10**(-self.cDarkHole/2)/np.sqrt(2.)*direct_field_t2_re
        A1  = -self.corono_field_t2_re \
                - 10**(-self.cDarkHole/2)/np.sqrt(2.)*direct_field_t2_re
        A2  = -np.identity(self.npp)
        A3  =  np.identity(self.npp)
    
        b0  = np.zeros((self.corono.nlam*self.ndz))
        b1  = np.zeros((self.corono.nlam*self.ndz))
        b2  = np.zeros(self.npp)
        b3  = np.ones(self.npp)
    
        self.A = np.concatenate((A0,A1,A2,A3), axis=1)
        self.b = np.concatenate((b0,b1,b2,b3))
        self.c = - 2.*np.pi*(np.arange(self.corono.nPup)[self.idx_pup]+0.5)\
                /(2.*self.corono.nPup)**2/self.TR
        
        return self.A, self.b, self.c

    def compute_gurobi_model(self):
        
        if self.A & self.b & self.c is not None:
            nA = np.shape(self.A)[1]
        
            # Create a new model               
            m = gb.Model("LP max tau new")
            # Create variables
            ApodTmp = m.addVars(self.npp, lb=0.0, ub=1.0, name="ApodTmp")
            # Set objective
            m.setObjective(gb.quicksum((self.c[i]*ApodTmp[i] 
                    for i in range(self.npp))), gb.GRB.MINIMIZE)
            # Add constraint:                
            m.addConstrs((gb.quicksum((ApodTmp[i]*self.A[i,j] 
                    for i in range(self.npp) if self.A[i,j])) <=  self.b[j] 
                    for j in range(nA)), "cpos")
            m.update()           
        else:
            raise ValueError('Be careful: A or b or c is not defined')
        return m
 

#%%
class MaxContrast(ProblemMatrix):

    default_params = get_default_params_MaxContrast_pb()
    
    def __init__(self, corono=cg.APLC1d(), **kwargs):
        super().__init__(**kwargs)
    
    def compute_matrices(self):
                                                           
        if self.Lnorm == 'Linf':
            I1 = np.ones(self.ndz*self.corono.nlam)
            I1 = I1[None,:]
            I0 = np.ones(self.ndz)
            I0 = I0[None,:]
            N0 = np.zeros((1, self.npp))
            Z0 = np.zeros(1)
            c1 = [1]
        else:
            I0 = np.identity(self.ndz)
            I1 = np.hstack([I0 for k in range(self.corono.nlam)])            
            N0 = np.zeros((self.ndz, self.npp))
            Z0 = np.zeros(self.ndz)
            c1 = 2.*np.pi*np.array(self.idx_dz)*(self.corono.Fmax\
                                  /self.corono.nImg)**2
        
        A0  = np.concatenate(( self.corono_field_t2_re, -I1), axis=0)
        A1  = np.concatenate((-self.corono_field_t2_re, -I1), axis=0)
        A2  = np.concatenate((-np.identity(self.npp), N0), axis=0)
        A3  = np.concatenate(( np.identity(self.npp), N0), axis=0)
        A4  = np.concatenate((np.zeros((self.npp, self.ndz)), -I0), axis=0)
        A5  = np.concatenate((- 2.*np.pi*(
                np.arange(self.corono.nPup)[self.idx_pup]+0.5)\
            *self.corono.Pupil[self.idx_pup]/(2.*self.corono.nPup)**2/self.TR, 
                                  Z0))
        
        b0  = np.zeros((self.corono.nlam*self.ndz))
        b1  = np.zeros((self.corono.nlam*self.ndz))
        b2  = np.zeros(self.npp)
        b3  = np.ones(self.npp)
        b4  = np.zeros(self.ndz)
        b5  = [-self.tau]
        
        self.A = np.concatenate((A0,A1,A2,A3,A4,A5[:,None]), axis=1)
        self.b = np.concatenate((b0,b1,b2,b3,b4,b5))        
        self.c = np.concatenate((np.zeros(self.npp), c1), axis=0)
        
        return self.A, self.b, self.c

    def compute_gurobi_model(self):
        
        if self.A & self.b & self.c is not None:        
            nn = np.shape(self.A)[1]
            # Create a new model  
            m = gb.Model("LP max C new")
            
            if self.Lnorm == 'Linf':
                self.neps = 1
            else:
                self.neps = self.ndz
            # Create variables
            ApodEpsTmp = m.addVars(self.npp + self.neps, lb=0.0, name="ApodTmp")        
            # Set objective
            m.setObjective(gb.quicksum((self.c[i+self.npp]*ApodEpsTmp[i+self.npp] 
                    for i in range(self.neps))), gb.GRB.MINIMIZE)
            # Add constraint:
            m.addConstrs((gb.quicksum((ApodEpsTmp[i]*self.A[i,j] 
                    for i in range(self.npp + self.neps))) <=  self.b[j] 
                    for j in np.arange(nn)), "cpos")
            
            m.update()
        else:
            raise ValueError('Be careful: A or b or c is not defined')
            
        return m   

#%%