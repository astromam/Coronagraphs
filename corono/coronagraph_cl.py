#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:57:28 2018

@author: mndiaye
"""

#%%
'''
### initialization
'''
import numpy as np
#import pylab as pl
#from astropy.io import fits
from .utils import besselJ0
import json

#%%
'''
### Coronagraph class
'''
def get_default_params_coronagraph():
    tmp = {'PupilObs':0.14,'LyotStopObs':0.28,
           'rho0':5,'rho1':10,'cDarkHole':8,
           'nPup':500,'nImg':200,'Fmax':25,
           'bw':0.2,'lam0':1.0,'nlam':5, 
           'R':1,
           'fdir':''
           }
    return tmp

def get_default_params_APLC1d():
    tmp = get_default_params_coronagraph()
    tmp.update({'rMask':2.8,'nFPM':100})
    return tmp

def get_default_params_DZPM1d():
    tmp = get_default_params_coronagraph()
    tmp.update({'rMask1':0.875/2, 'rMask2':1.453/2.,
           'OPDx1':0.309, 'OPDx2':0.672,
           'ome1':-2.340, 'ome2':2.051, 'beta':-0.236,
           'gFPM':68.82312456985547})
    return tmp
    
fname_coronagraph = 'obs={PupilObs}_ls={LyotStopObs}\
_IWA={rho0}_OWA={rho1}_C={cDarkHole:02d}\
_nPup={nPup:04d}_nImg={nImg}_Fmax={Fmax}\
_bw={bw}_nlam={nlam:02d}'

class Coronagraph(object):
        
    default_params = get_default_params_coronagraph()
    fname_format   = fname_coronagraph
#%%
    def __init__(self, **kwargs):
        '''
        to be written
        '''
        self.params      = kwargs 
        self.check_params()
        
        # wavelengths        
        self.dlam       = self.bw*self.lam0
        self.lam_t      = np.linspace(
                self.lam0-self.dlam/2*(self.nlam>1),
                self.lam0+self.dlam/2,self.nlam)

        # Pupil radial coordinate 
        self.r          = np.arange(self.nPup)*self.R/self.nPup\
                +self.R/(2*self.nPup)
        
        # clear Pupil
        self.ClearPupil = np.ones((self.nPup))
        # Telescope aperture
        self.Pupil      = (self.r>self.PupilObs)*1.0
        # Lyot stop 
        self.LyotStop   = (self.r>self.LyotStopObs)*1.0

        # Final image plane coordinate
        self.xi  = np.arange(self.nImg+1)*self.Fmax/self.nImg
        # final image plane coordinate weighted with wavelength
        self.xii = self.xi[None,:]*self.lam0/self.lam_t[:,None]

        # identity matrix for the apodization
        self.Apod_t = np.identity(self.nPup)

        # Hankel kernel (no wavelength variation)
        self.hankel_kernel     = besselJ0(
                np.pi/self.R*self.xi[:,None]*self.r[None,:])
        # Hankel kernel (including wavelength variation)
        self.hankel_kernel_all = besselJ0(
                np.pi/self.R*self.xii[:,:,None]*self.r[None,None,:])
        
#%%        
    def save_params(self, fname):
        '''
        save params in fname using JSON (JavaScript Object Notation)
        ----
        
        - input:
            fname: string
                filename
        
        '''
        f=open(fname,'w')
        f.write(json.dumps(self.params,sort_keys=True,indent=4))
        f.close()
        
#%%    
    def load_params(self, fname):
        '''
        load params from fname using JSON (JavaScript Object Notation)
        ----
        '''
        f=open(fname,'r')
        self.params=json.loads(f.read())
        f.close()
        
#%%        
    def get_filename(self):
        '''
        to be written
        '''
        return self.fname_format.format(**self.params)
    
#%%    
    def get_cache(self,varname):
        '''
        to be written
        '''
        try:
            self.get_filename()+ '_' + varname
        except FileNotFoundError:
            return False
        
#%%        
    def check_params(self):
        '''
        to be written
        '''        
        for key in self.default_params:
            if not key in self.params:
                self.params[key] = self.default_params[key]
                
#%%
    def __repr__(self):        
        '''
        to be written
        '''
        res=''
        for key in sorted(self.params):
            res+='{:>20s} : {}\n'.format(key,self.params[key])        
        return res
    
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

#%%    # direct signal in intensity
    def compute_direct_intensity_1d(self,Apod,poly=True):
        '''
        to be written
        '''
        direct_field = self.compute_direct_field_1d(Apod)
        
        if poly:
            return np.sum(np.abs(direct_field)**2,0)
        else:
            return np.abs(direct_field)**2

#%%    # coronagraphic signal in intensity
    def compute_corono_intensity_1d(self,Apod,poly=True):
        
        corono_field = self.compute_corono_field_1d(Apod)
        
        if poly:
            return np.sum(np.abs(corono_field)**2,0)
        else:
            print('Warning: No normalization for multiple lambda!')
            return np.abs(corono_field)**2       

#%%    # generation of the direct response matrix
    def prop_direct_matrix(self,):
        direct_field_t = np.zeros((self.nPup, self.nlam, self.nImg+1), 
                                  dtype='complex128')
        print('generating direct response matrices')
        for i in np.arange(self.nPup):
            direct_field_t[i] = self.compute_direct_field_1d(self.Apod_t[i])
        direct_field_t_re = direct_field_t.real
        direct_field_t_im = direct_field_t.imag
        return direct_field_t_re, direct_field_t_im    
    
#%%    # generation of the coronagraphic response matrix        
    def prop_corono_matrix(self,):
        corono_field_t = np.zeros((self.nPup, self.nlam, self.nImg+1), 
                                  dtype='complex128')
        print('generating corono response matrices')
        for i in np.arange(self.nPup):
            corono_field_t[i] = self.compute_corono_field_1d(self.Apod_t[i])
        corono_field_t_re = corono_field_t.real
        corono_field_t_im = corono_field_t.imag
        return corono_field_t_re, corono_field_t_im 
    
#%%
class APLC1d(Coronagraph):
    
    default_params = get_default_params_APLC1d()
    
    def __init__(self, **kwargs):
        '''
        to be written
        '''      
        super().__init__(**kwargs)      
        
        # mask size at Apod given wavelength
        self.rMask_t    = (self.lam0/self.lam_t)*self.rMask
        
        # mask sampling at Apod given wavelength and max nFPM_max 
        self.nFPM_t     = self.rMask_t*self.nFPM
        self.nFPM_max   = int(np.max(self.nFPM_t))
        self.mask_lam   = (np.arange(self.nFPM_max+1)[None,:]\
                           <self.rMask_t[:,None]*self.nFPM)
        self.xi_FPM_lam = np.arange(self.nFPM_max+1)[None,:]\
                *self.mask_lam/self.nFPM

        # Hankel kernel for the focal plane mask (FPM) 
        self.hankel_kernel_FPM_all  = besselJ0(
                np.pi/self.R*self.xi_FPM_lam[:,:,None]*self.r[None,None,:])
        self.hankel_kernel_iFPM_all = besselJ0(
                np.pi/self.R*self.xi_FPM_lam[:,None,:]*self.r[None,:,None])
        
#%%    # direct propagation (no focal plane mask)
    def compute_direct_field_1d(self,Apod):
        '''
        computation of the direct electric field
        '''
        return self.lam0/self.lam_t[:,None]*np.pi*self.hankel_kernel_all.dot(
                self.Pupil*Apod*self.LyotStop*self.r/self.R)*self.R/self.nPup

#%%    # propagation through coronagraph (with focal plane mask)
    def compute_corono_field_1d(self,Apod):
        '''
        computation of the coronagraphic electric field
        '''
        FPM_field  = np.pi*self.hankel_kernel_FPM_all.dot(
                Apod*self.Pupil*self.r/self.R)\
                *(self.R/self.nPup)*self.xi_FPM_lam
                
        iFPM_field = np.zeros((self.nlam,self.nPup))
        for i in range(self.nlam):
            iFPM_field[i,:] = np.pi*self.hankel_kernel_iFPM_all[i,:,:].dot(
                    FPM_field[i,:])*(1/self.nFPM)
        
        nolyot_field = (Apod[None,:]*self.Pupil[None,:]-iFPM_field)\
                *self.r[None,:]/self.R
        
        lyot_field   = nolyot_field*self.LyotStop[None,:]
        
        corono_field_tmp = np.zeros((self.nlam,self.nImg+1))
        for i in range(self.nlam):
            corono_field_tmp[i,:] = self.hankel_kernel_all[i,:,:].dot(
                    lyot_field[i,:])
        
        return self.lam0/self.lam_t[:,None]*np.pi\
            *corono_field_tmp*self.R/self.nPup
     
#%%
class DZPM1d(Coronagraph):
    
    default_params = get_default_params_DZPM1d()    

    def __init__(self, **kwargs):
        '''
        to be discussed with Remi F.
        '''
        super().__init__(**kwargs)
        
        # Optical path difference introduced by the mask
        self.OPD1 = self.OPDx1*self.lam0
        self.OPD2 = self.OPDx2*self.lam0
        
        # phase shift induced by the mask
        self.phi1_t = 2.*np.pi*self.OPD1/self.lam_t
        self.phi2_t = 2.*np.pi*self.OPD2/self.lam_t
        
        self.eps1_t   = 1j*np.sin(self.phi1_t) + np.cos(self.phi1_t)
        self.eps2_t   = 1j*np.sin(self.phi2_t) + np.cos(self.phi2_t)

        # mask size at Apod given wavelength
        self.rMask1_t = (self.lam0/self.lam_t)*self.rMask1
        self.rMask2_t = (self.lam0/self.lam_t)*self.rMask2
        
        # mask sampling at Apod given wavelength and max nFPM_max 
        self.nFPM1_t   = self.rMask1_t*self.gFPM
        self.nFPM1_max = int(np.max(self.nFPM1_t))     
        self.nFPM2_t   = self.rMask2_t*self.gFPM
        self.nFPM2_max = int(np.max(self.nFPM2_t))

        self.mask1_lam   = (np.arange(self.nFPM1_max+1)[None,:]\
                            <=self.rMask1_t[:,None]*self.gFPM)
        self.xi_FPM1_lam = np.arange(self.nFPM1_max+1)[None,:]\
        *self.mask1_lam/self.gFPM       
        self.mask2_lam   = (np.arange(self.nFPM2_max+1)[None,:]\
                            <=self.rMask2_t[:,None]*self.gFPM)
        self.xi_FPM2_lam = np.arange(self.nFPM2_max+1)[None,:]\
        *self.mask2_lam/self.gFPM

        self.hankel_kernel_FPM1_all  = besselJ0(
                np.pi/self.R*self.xi_FPM1_lam[:,:,None]*self.r[None,None,:])
        self.hankel_kernel_iFPM1_all = besselJ0(
                np.pi/self.R*self.xi_FPM1_lam[:,None,:]*self.r[None,:,None])
        self.hankel_kernel_FPM2_all  = besselJ0(
                np.pi/self.R*self.xi_FPM2_lam[:,:,None]*self.r[None,None,:])
        self.hankel_kernel_iFPM2_all = besselJ0(
                np.pi/self.R*self.xi_FPM2_lam[:,None,:]*self.r[None,:,None])

        self.Apod_w = \
        1j*np.sin(2.*np.pi*(self.r/2)**2*self.beta*self.lam0/self.lam_t[:,None])\
        + np.cos(2.*np.pi*(self.r/2)**2*self.beta*self.lam0/self.lam_t[:,None])

#%%       
    # direct propagation (no focal plane mask)
    def compute_direct_field_1d(self,Apod):
        field = np.zeros((self.nlam,self.nImg+1), dtype='complex128')
        for i in range(self.nlam):
            field[i,:] = \
            self.lam0/self.lam_t[i]*np.pi*self.hankel_kernel_all[i].dot(
                    self.Pupil*Apod*self.Apod_w[i]*self.LyotStop*self.r/self.R)\
                    *self.R/self.nPup
        return field

#%%
    # propagation through coronagraph (with focal plane mask)
    def compute_corono_field_1d(self,Apod):
    
        FPM1_field = np.zeros((self.nlam, self.nFPM1_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM1_field[i,:] = np.pi*self.hankel_kernel_FPM1_all[i,:,:].dot(
                    Apod*self.Apod_w[i]*self.Pupil*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM1_lam[i,None]

        iFPM1_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM1_field[i,:]= np.pi*self.hankel_kernel_iFPM1_all[i,:,:].dot(
                    FPM1_field[i,:])*(1/self.gFPM)
    
        FPM2_field = np.zeros((self.nlam, self.nFPM2_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM2_field[i,:] = np.pi*self.hankel_kernel_FPM2_all[i,:,:].dot(
                    Apod*self.Apod_w[i]*self.Pupil*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM2_lam[i,None]

        iFPM2_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM2_field[i,:]=np.pi*self.hankel_kernel_iFPM2_all[i,:,:].dot(
                    FPM2_field[i,:])*(1/self.gFPM)
             
        nolyot_field = (Apod[None,:]*self.Apod_w*self.Pupil[None,:]\
                        -(self.eps2_t[:,None] - self.eps1_t[:,None])*iFPM1_field \
                        -(1.-self.eps2_t[:, None])*iFPM2_field)*self.r[None,:]/self.R
    
        lyot_field   = nolyot_field*self.LyotStop[None,:]
        
        corono_field_tmp = np.zeros((self.nlam,self.nImg+1), dtype='complex128')
        for i in range(self.nlam):
            corono_field_tmp[i,:] = self.hankel_kernel_all[i,:,:].dot(lyot_field[i,:])
        
        return self.lam0/self.lam_t[:,None]*np.pi*corono_field_tmp*self.R/self.nPup
