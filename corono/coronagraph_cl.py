#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:57:28 2018

@author: mndiaye

Class for different types of coronagraphs
"""

#%%
"""
Initialization
"""
import numpy as np
#import pylab as pl
#from astropy.io import fits
from .utils import besselJ0, sft, isft, uniform_disk
import json

#%% 
"""
Default parameters
"""
#%%
def get_default_params_Coronagraph():
    """
    Gets the default parameters for the Coronagraph class.
        
    Parameters
    ---------- 
    PupilObs : float
        Pupil central obstruction size in fraction of the pupil diameter
        
    LyotStopObs : float
        Lyot stop central obstruction size in fraction of the pupil diameter
        
    rho0 : float
        Inner edge of the search area in the coronagraphic image
        
    rho1 : float
        Outer edge of the search area in the coronagraphic image
        
    nPup : int
        Sampling across the pupil diameter
        
    nImg : int
        Sampling across the final image plane along one axis for 1D simulations
        
    Fmax : float
        Maximum spatial frequency in the final image plane for 1D simulations
        
    nImg2d : int
        Final image plane sampling for 2D simulations
        
    Fmax2d : float
        Maximum spatial frequency in the final image plane for 2D simulations

    lam0 : float
        Central wavelength 

    bw : float 
        Spectral bandwidth in fraction of lam0            
    
    nlam : int 
        Spectral sampling
  
    R : float 
        Unitary radius of the pupil
        
    fdir : string
        Directory
        
    ctr : boolean
        Keyword to work with pupil arrays that are centered between four pixels
        if True
        
    ctr2 : boolean
        Keyword to work with image arrays that are centered between four pixels
        if True           
            
    Returns    
    ----------
    tmp : dict
        Dictionary with parameters and all their default values.
            
    """
    tmp = {'PupilObs':0.14,'LyotStopObs':0.28,
           'rho0':5,'rho1':10,
           'nPup':200,'nImg':200,'Fmax':25,
           'nImg2d':44, 'Fmax2d':22,
           'bw':0.2,'lam0':1.0,'nlam':5, 
           'R':1,
           'fdir':'',
           'ctr':True, 'ctr2':False
           }
    return tmp

#%%
def get_default_params_APLC1d():
    """
    Gets the default parameters for the APLC1d Coronagraph subclass.
        
    Parameters
    ---------- 
    rMask : float
        Focal plane mask radius in lam0/D
        
    nFPM : int
        Mask sampling
            
    Returns    
    ----------
    tmp : dict
        Dictionary that contains all the default values of the 
        Coronagraph class and the APLC1d subclass
    
    """
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask':2.8,'nFPM':50})
    return tmp

#%%
def get_default_params_DZPM1d():
    """
    Gets the default parameters for the DZPM1d Coronagraph subclass.
        
    Parameters
    ---------- 
    rMask1 : float
        Inner part of the focal plane mask radius in lam0/D
        
    rMask2 : float
        Outer part of the focal plane mask radius in lam0/D    
        
    OPDx1 : float
        Optical path difference for the inner part of the focal plane mask
        
    OPDx2 : float
        Optical path difference for the outer part of the focal plane mask
        
    ome1 : float
        Second order term for an apodization with amplitude transmission 
        polynomial function
        
    ome2 : float
        Forth order term for an apodization with amplitude transmission 
        polynomial function
        
    beta : float
        Coefficient in lam0 related to a defocus shift that is applied to the 
        focal plane mask. Equivalent to a phase entrance pupil apodization. 
        
    gFPM : float
        Mask sampling            
            
    Returns    
    ----------
    tmp : dict
        Dictionary with all the default values of the 
        Coronagraph class and the DZPM1d subclass.
    
    """   
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask1':0.875/2, 'rMask2':1.453/2.,
           'OPDx1':0.309, 'OPDx2':0.672,
           'ome1':-2.340, 'ome2':2.051, 'beta':-0.236,
           'gFPM':68.82312456985547})
    return tmp

#%%
def get_default_params_APLC2d():
    """
    Gets the default parameters for the APLC2d Coronagraph subclass.
        
    Parameters
    ---------- 
    rMask : float
        Focal plane mask radius in lam0/D
    
    nPup : int
        Sampling across the pupil diameter
    
    nFPM : int
        Sampling across the mask diameter
            
    Returns    
    ----------
    tmp : dict
        Dictionary with all the default values of the 
        Coronagraph class and the APLC2d subclass
        
    """    
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask':2.8,
                'nPup':50, 'nFPM':25
                })
    return tmp

#%%
"""
Coronagraph filename
""" 
fname_coronagraph = 'obs={PupilObs}_ls={LyotStopObs}\
_IWA={rho0}_OWA={rho1}\
_nPup={nPup:04d}_nImg={nImg}_Fmax={Fmax}\
_bw={bw}_nlam={nlam:02d}'

#%% 
"""
Coronagraph class
"""
class Coronagraph(object):
    """
    Defines the class for coronagraphs.
    """    
    default_params = get_default_params_Coronagraph()
    fname_format   = fname_coronagraph
#%%
    def __init__(self, **kwargs):
        """
        __init__ : method
            Builds the constructor for the Coronagraph class.
        
        Attributes
        ----------            
        params : dict
            Dictionary of parameters for the Coronagraph class
            
        lam0 : float
            Central wavelength 
        
        bw : float 
            Spectral bandwidth in fraction of lam0
        
        dlam : float 
            Spectral bandwidth in lam0
            
        r : array_like 
            vector containing the Pupil radial coordinate
            
        nPup : float 
            Sampling across the pupil diameter
            
        R : float 
            Unitary radius of the pupil

        PupilObs : float
            Central obstruction size for the pupil in pupil diameter
        
        LyotStopObs : float
            Central obstruction size for the Lyot stop in pupil diameter
        
        ClearPupil1d : array_like 
            Clear pupil (no obstruction nor spiders)
        
        Pupil1d : array_like
            1D Entrance pupil
        
        LyotStop1d : array_like 
            1D Lyot Stop
        
        xi : array_like 
            Vector for the final image pupil plane coordinate
        
        xii : array_like 
            Vector for the final image pupil plane coordinate that is 
            weighted with wavelength
        
        Apod1d_t : array_like 
            Identity matrix for the computation of the response matrix for all
            the points in the Entrance pupil plane for 1D problem
        
        hankel_kernel : array_like
            Hankel kernel without wavelength variation
        
        hankel_kernel_all : 3D array
            Hankel kernel including wavelength variation
        
        ClearPupil2d : array_like 
            2D clear pupil
        
        Pupil2d : array_like     
            2D entrance pupil
        
        mask2d : array_like 
            2D focal plane mask
        
        LyotStop2d : array_like 
            2D Lyot stop
        
        xi2d : array_like 
            Final image plane coordinate vector centered on a pixel
            for 2D problem
        
        xi2d_ctr : array_like
            Final image plane coordinate vector centered between 4 pixels
            for 2D problem
        
        Apod1d_t : array_like 
            Identity matrix for the computation of the response matrix for all
            the points in the Entrance pupil plane for 2D problem
        
        """
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
        self.ClearPupil1d = np.ones((self.nPup))
        # Telescope aperture
        self.Pupil1d      = (self.r>self.PupilObs)*1.0
        # Lyot stop 
        self.LyotStop1d   = (self.r>self.LyotStopObs)*1.0
        
        # Final image plane coordinate
        self.xi  = np.arange(self.nImg+1)*self.Fmax/self.nImg
        # final image plane coordinate weighted with wavelength
        self.xii = self.xi[None,:]*self.lam0/self.lam_t[:,None]

        # identity matrix for the apodization
        self.Apod1d_t = np.identity(self.nPup)

        # Hankel kernel (no wavelength variation)
        self.hankel_kernel     = besselJ0(
                np.pi/self.R*self.xi[:,None]*self.r[None,:])
        # Hankel kernel (including wavelength variation)
        self.hankel_kernel_all = besselJ0(
                np.pi/self.R*self.xii[:,:,None]*self.r[None,None,:])

        # clear Pupil
        self.ClearPupil2d = uniform_disk(self.nPup, self.nPup/2., ctr=self.ctr)
        # Telescope aperture
        self.Pupil2d      = uniform_disk(self.nPup, self.nPup/2., ctr=self.ctr)\
        - uniform_disk(self.nPup, self.PupilObs*self.nPup/2., ctr=self.ctr)
        # Focal plane mask
        self.mask2d       = uniform_disk(self.nFPM, self.nFPM/2., ctr=self.ctr)
        # Lyot stop 
        self.LyotStop2d   = uniform_disk(self.nPup, self.nPup/2., ctr=self.ctr)\
        - uniform_disk(self.nPup, self.LyotStopObs*self.nPup/2., ctr=self.ctr)

        # Final image plane coordinate
        self.xi2d     = (np.arange(self.nImg2d//2))* self.Fmax2d/self.nImg2d
        self.xi2d_ctr = (np.arange(self.nImg2d//2)+1/2)* self.Fmax2d/self.nImg2d

        # identity matrix for the apodization
        self.Apod2d_t = np.identity(self.nPup**2)

        
#%%        
    def save_params(self, fname):
        """
        Saves params in fname using JavaScript Object Notation (JSON).
        
        Parameters
        ---------- 
        fname : string
            Filename in which parameters are to be written
        
        """
        f=open(fname,'w')
        f.write(json.dumps(self.params,sort_keys=True,indent=4))
        f.close()
        
#%%    
    def load_params(self, fname):
        """
        Loads params from fname using JavaScript Object Notation (JSON).
        
        Parameters
        ---------- 
        fname : string
            Filename in which parameters are to be load
                
        """
        
        f=open(fname,'r')
        params=json.loads(f.read())
        self.__init__(**params)
        f.close()
        
#%%        
    def get_filename(self):
        """
        Gets params from the params list.
        
        Returns
        -------
        res 
            Params from the list.
                
        """
        return self.fname_format.format(**self.params)
    
#%%    
    def get_cache(self,varname):
        """
        Sets filename with a given varname.
        
        Parameters
        ---------- 
        varname : string
            Varname to append to the default filename.
            
        Returns
        ---------
        res : string    
            Filename with the appended varname.
                
        """
        try:
            self.get_filename()+ '_' + varname
        except FileNotFoundError:
            return False
        
#%%        
    def check_params(self):
        """
        Sets default values to the parameters that have not been set
        by the user.
        
        Returns
        -------
        res
            Values for all the parameters
        
        """        
        for key in self.default_params:
            if not key in self.params:
                self.params[key] = self.default_params[key]
                
#%%
    def __repr__(self):        
        """
        Prints the official string representation of all 
        the values in the params object.
        
        Returns    
        ----------
        res
            Displays all the values in the object params
            
        """
        res=''
        for key in sorted(self.params):
            res+='{:>20s} : {}\n'.format(key,self.params[key])        
        return res
    
#%%
    def __contains__(self, item):
        """
        Requests the value of an item in the params object.
            
        Parameters
        ----------  
        item
            Parameter in params
        
        Returns    
        ----------
        item
            Value of the requested parameter
            
        """
        return item in self.params
    
#%%     
    def __getattr__(self, name):
        """
        Gets the attribute of a key in the params object.
        
        Parameters
        ----------  
        name : string
            Name of the requested parameters
                
        Returns    
        ----------
        res
            Value of parameter that is related to name 
                
        """
        return self.params[name]

#%% direct signal in intensity
    def compute_direct_intensity_1d(self,Apod,poly=True):
        """
        Computes the intensity of the direct image for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization
        
        poly : thuth value, True by default
            parameter to compute broadband image or monochromatic images
            at all the wavelengths
                
        Returns    
        ----------
        res : array_like
            Intensity of the direct broadband image or monochromatic images. 
            
        """
        direct_field = self.compute_direct_field_1d(Apod)
        
        if poly:
            return np.sum(np.abs(direct_field)**2,0)
        else:
            return np.abs(direct_field)**2

#%% coronagraphic signal in intensity
    def compute_corono_intensity_1d(self,Apod,poly=True):
        """
        Computes the intensity of the coronagraphic image for 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization
            
        poly : boolean (default=True)
            parameter to compute broadband image or monochromatic images
            at all the wavelengths
                
        Returns    
        ----------
        res : array_like
            Intensity of the coronagraphic broadband image or monocrhomatic
            images.
            
        """        
        corono_field = self.compute_corono_field_1d(Apod)
        
        if poly:
            return np.sum(np.abs(corono_field)**2,0)
        else:
            print('Warning: No normalization for multiple lambda!')
            return np.abs(corono_field)**2       

#%% generation of the direct response matrix
    def prop_direct_matrix_1d(self,):
        """
        Computes the response matrix of the direct image for all the points in
        the pupil for 1D problem.
                
        Returns    
        ----------
        res : array_like, array_like
            Real and imaginary parts of the response matrix 
            for the direct image 
            
        """        
        direct_field_t = np.zeros((self.nPup, self.nlam, self.nImg+1), 
                                  dtype='complex128')
        print('generating direct response matrices')
        for i in np.arange(self.nPup):
            direct_field_t[i] = self.compute_direct_field_1d(self.Apod1d_t[i])
        direct_field_t_re = direct_field_t.real
        direct_field_t_im = direct_field_t.imag
        return direct_field_t_re, direct_field_t_im    
    
#%% generation of the coronagraphic response matrix        
    def prop_corono_matrix_1d(self,):
        """
        Computes the response matrix of the coronagraphic image for all the 
        points in the pupil for 1D problem.
                
        Returns    
        ----------
        res : array_like, array_like
            Real and imaginary parts of the response matrix 
            for the coronagraphic image 
            
        """
        corono_field_t = np.zeros((self.nPup, self.nlam, self.nImg+1), 
                                  dtype='complex128')
        print('generating corono response matrices')
        for i in np.arange(self.nPup):
            corono_field_t[i] = self.compute_corono_field_1d(self.Apod1d_t[i])
        corono_field_t_re = corono_field_t.real
        corono_field_t_im = corono_field_t.imag
        return corono_field_t_re, corono_field_t_im 

#%% direct signal in intensity
    def compute_direct_intensity_2d(self,Apod2d,poly=True):
        """
        Computes the intensity of the direct image for the 2D problem.
        
        Parameters
        ---------- 
        Apod2d : array_like
            Entrance pupil apodization
        
        poly : thuth value, True by default
            Parameter to compute broadband image or monochromatic images
            at all the wavelengths
                
        Returns    
        ----------
        res : array_like
            Intensity of the direct broadband image or monochromatic images. 
            
        """        
        direct_field_2d = self.compute_direct_field_2d(Apod2d)
        
        if poly:
            return np.sum(np.abs(direct_field_2d)**2,0)
        else:
            return np.abs(direct_field_2d)**2

#%% coronagraphic signal in intensity
    def compute_corono_intensity_2d(self,Apod2d,poly=True):
        """
        Computes the intensity of the coronagraphic image for the 2D problem.
        
        Parameters
        ---------- 
        Apod2d : array_like
            Entrance pupil apodization
            
        poly : boolean (default=True)
            Parameter to compute broadband image (if True) 
            or monochromatic images at all the wavelengths
                
        Returns    
        ----------
        res : array_like
            Intensity of the coronagraphic broadband image or monocrhomatic
            images.
        
        """
        corono_field_2d = self.compute_corono_field_2d(Apod2d)
        
        if poly:
            return np.sum(np.abs(corono_field_2d)**2,0)
        else:
            print('Warning: No normalization for multiple lambda!')
            return np.abs(corono_field_2d)**2

#%%
    def compute_direct_field_2d_vec(self,Apod2d):
        """
        Computes the electric field of the direct image 
        for the vectorized 2D problem.
        
        Parameters
        ---------- 
        Apod2d : array_like
            Entrance pupil apodization
                
        Returns    
        ----------
        res : array_like, array_like
            Real and imag parts of the direct electric field 
            
        """        
        test = self.compute_direct_field_2d(Apod2d)
        test_re = np.reshape(test.real, (self.nlam, self.nImg2d**2))
        test_im = np.reshape(test.imag, (self.nlam, self.nImg2d**2))
        return test_re, test_im

#%%
    def compute_corono_field_2d_vec(self,Apod2d):
        """
        Computes the electric field of the coronagraphic image 
        for the vectorized 2D problem.
        
        Parameters
        ---------- 
        Apod2d : array_like
            Entrance pupil apodization
                
        Returns    
        ----------
        res : array_like, array_like
            Real and imag parts of the coronagraphic electric field 
            
        """ 
        test = self.compute_corono_field_2d(Apod2d)
        test_re = np.reshape(test.real, (self.nlam, self.nImg2d**2))
        test_im = np.reshape(test.imag, (self.nlam, self.nImg2d**2))
        return test_re, test_im

#%% generation of the direct response matrix
    def prop_direct_matrix_2d(self):
        """
        Computes the response matrix of the direct image for all the points in
        the pupil for the vectorized 2D problem.
                
        Returns    
        ----------
        res : array_like, array_like
            Real and imaginary parts of the direct image response matrix 
            
        """
        
        direct_field_t_re = np.zeros((self.nPup**2, self.nlam, self.nImg2d**2))
        direct_field_t_im = np.zeros((self.nPup**2, self.nlam, self.nImg2d**2))
        print('generating direct response matrix')
        for i in np.arange(self.nPup**2):
            Apod2d = np.reshape(self.Apod2d_t[i], (self.nPup,self.nPup))  
            direct_field_t_re[i],direct_field_t_im[i] = self.compute_direct_field_2d_vec(Apod2d)
        return direct_field_t_re, direct_field_t_im    

#%% generation of the coronagraphic response matrix        
    def prop_corono_matrix_2d(self):
        """
        Computes the response matrix of the coronagraphic image for all the 
        points in the pupil for vectorized 2D problem.
                
        Returns    
        ----------
        res : array_like, array_like
            Real and imaginary parts of the coronagraphic image response matrix 
            
        """
        corono_field_t_re = np.zeros((self.nPup**2, self.nlam, self.nImg2d**2))
        corono_field_t_im = np.zeros((self.nPup**2, self.nlam, self.nImg2d**2))
        print('generating corono response matrix')
        for i in np.arange(self.nPup**2):
            Apod2d = np.reshape(self.Apod2d_t[i], (self.nPup,self.nPup))  
            corono_field_t_re[i], corono_field_t_im[i] = self.compute_corono_field_2d_vec(Apod2d)
        return corono_field_t_re, corono_field_t_im 

#%%
    def generate_area(self,):
        """
        Computes the list of points with a given area in the final image plane 
        of the coronagraph. The area is defined by an annulus with minimum and 
        maximum angular separation from the star.
    
        Parameters
        ----------     
        mD : float
            Spatial frequency range in the final image plane D in lam0/D
            
        nImg2d : int
            Linear number of points in the final image plane D
            
        sep_min : float
            Minimum angular separation from the star for the area
            
        sep_max : float
            Maximum angular separation from the star for the area
    
        Returns    
        ----------
        res : array_like
            2D array with 1 and 0 for the points inside and outside the area 
            in the coronagraphic image
    
        """
        val = 0
        if self.ctr is True:
            val = 1/2
        # array of angular distances in the final image plane
        xx,yy  = np.meshgrid(np.arange(self.nImg2d)-self.nImg2d/2+val, np.arange(self.nImg2d)-self.nImg2d/2+val)
        mydist = (self.Fmax2d/self.nImg2d)*np.hypot(yy,xx)
    
        # array with 1 and 0 for points inside and outside the area in the coronagraphic image
        #res    = np.zeros_like(mydist)
        #res[(mydist <= self.rho1)*(mydist >= self.rho0)] = 1.0
        res = (mydist <= self.rho1)*(mydist >= self.rho0)
        return res, mydist[res]
    
#%% 
"""
APLC 1d class
"""
class APLC1d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Apodized Pupil Lyot Coronagraph
    for one-dimension geometry.
    """
    default_params = get_default_params_APLC1d()
    
    def __init__(self, **kwargs):
        """
        __init__ : method
            Builds the constructor for the APLC1d class
        
        Attributes:
        ----------  
        rMask_t : array
            Focal plane mask radius scaled with wavelength
            
        nFPM_t : array
            Mask sampling at a given wavelength
            
        nFPM_max : float
            Maximum mask sampling over all the wavelengths
            
        mask_lam : array
            Focal plane mask in lam/D unit
            
        xi_FPM_lam : array
            Focal plane mask coordinate in lam/D
            
        hankel_kernel_FPM_all : array
            Hankel kernel for the focal plane mask at all the wavelengths
            
        hankel_kernel_iFPM_all : array
            Inverse hankel transform for the focal plane mask at all the 
            wavelengths

        """      
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
        
#%% # direct propagation (no focal plane mask)
    def compute_direct_field_1d(self,Apod):
        """
        Computes the electric field of the direct image with APLC
        for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization
                
        Returns    
        ----------
        res : array_like
            Direct electric field at all the wavelengths
            
        """
        return self.lam0/self.lam_t[:,None]*np.pi*self.hankel_kernel_all.dot(
                self.Pupil1d*Apod*self.LyotStop1d*self.r/self.R)*self.R/self.nPup

#%% # propagation through coronagraph (with focal plane mask)
    def compute_corono_field_1d(self,Apod):
        """
        Computes the electric field of the coronagraphic image with APLC
        for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization
                
        Returns    
        ----------
        res : array_like
            Coronagraphic electric field at all the wavelengths
            
        """
        FPM_field  = np.pi*self.hankel_kernel_FPM_all.dot(
                Apod*self.Pupil1d*self.r/self.R)\
                *(self.R/self.nPup)*self.xi_FPM_lam
                
        iFPM_field = np.zeros((self.nlam,self.nPup))
        for i in range(self.nlam):
            iFPM_field[i,:] = np.pi*self.hankel_kernel_iFPM_all[i,:,:].dot(
                    FPM_field[i,:])*(1/self.nFPM)
        
        nolyot_field = (Apod[None,:]*self.Pupil1d[None,:]-iFPM_field)\
                *self.r[None,:]/self.R
        
        lyot_field   = nolyot_field*self.LyotStop1d[None,:]
        
        corono_field_tmp = np.zeros((self.nlam,self.nImg+1))
        for i in range(self.nlam):
            corono_field_tmp[i,:] = self.hankel_kernel_all[i,:,:].dot(
                    lyot_field[i,:])
        
        return self.lam0/self.lam_t[:,None]*np.pi\
            *corono_field_tmp*self.R/self.nPup
            
#%% 
"""
class DZPM 1d     
"""
class DZPM1d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Dual Zone Phase Mask Coronagraph
    for one-dimension geometry.
    """    
    default_params = get_default_params_DZPM1d()    

    def __init__(self, **kwargs):
        """
        __init__ : method
            build the constructor for the DZPM1d class
        
        Attributes:
        ----------      
        OPD1 : float
            Optical path difference for the inner part of the mask
            
        OPD2 : float
            Optical path difference for the outer part of the mask
            
        phi1_t : array
            Phase shift induced by the inner part of the mask at all the 
            wavelengths

        phi2_t : array
            Phase shift induced by the outer part of the mask at all the 
            wavelengths
        
        eps1_t : array
            Phasor induced by the inner part of the mask at all the 
            wavelengths

        eps2_t : array
            Phasor induced by the outer part of the mask at all the 
            wavelengths
        
        rMask1_t : array
            Focal plane mask inner part radius scaled with wavelength
            
        rMask2_t : array
            Focal plane mask outer part radius scaled with wavelength
            
        nFPM1_t : array
            Inner mask sampling at a given wavelength
        
        nFPM2_t : array
            Outer mask sampling at a given wavelength
        
        nFPM1_max : float
            Maximum inner mask sampling over all the wavelengths
            
        nFPM2_max : float
            Maximum outer mask sampling over all the wavelengths
        
        mask1_lam : array
            Inner focal plane mask in lam/D unit
        
        mask2_lam : array
            Outer focal plane mask in lam/D unit
        
        xi_FPM1_lam : array
            Inner focal plane mask coordinate in lam/D
        
        xi_FPM2_lam : array
            Outer focal plane mask coordinate in lam/D       
 
        hankel_kernel_FPM1_all : array
            Hankel kernel for the inner focal plane mask
            at all the wavelengths
        
        hankel_kernel_FPM2_all : array
            Hankel kernel for the outer focal plane mask
            at all the wavelengths
        
        hankel_kernel_iFPM1_all : array
            Inverse hankel transform for the inner focal plane mask
            at all the wavelengths
            
        hankel_kernel_iFPM2_all : array
            Inverse hankel transform for the outer focal plane mask
            at all the wavelengths
            
        Apod_w : array
            Complex apodization at all the wavelengths

        """
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

#%% direct propagation (no focal plane mask)
    def compute_direct_field_1d(self,Apod):
        """
        Computes the electric field of the direct image with DZPM
        for 1D problem.
        
        Parameters
        ---------- 
        Apod: array_like
            Entrance pupil apodization
                
        Returns    
        ----------
        res : array_like
            Direct electric field at all the wavelengths
            
        """
        field = np.zeros((self.nlam,self.nImg+1), dtype='complex128')
        for i in range(self.nlam):
            field[i,:] = \
            self.lam0/self.lam_t[i]*np.pi*self.hankel_kernel_all[i].dot(
                    self.Pupil1d*Apod*self.Apod_w[i]*self.LyotStop1d*self.r/self.R)\
                    *self.R/self.nPup
        return field

#%% propagation through coronagraph (with focal plane mask)
    def compute_corono_field_1d(self,Apod):
        """
        Computes the electric field of the coronagraphic image with DZPM
        for 1D problem.
        
        Parameters
        ---------- 
        Apod: array_like
            Entrance pupil apodization
                
        Returns    
        ----------
        res : array_like
            Coronagraphic electric field at all the wavelengths
            
        """
        FPM1_field = np.zeros((self.nlam, self.nFPM1_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM1_field[i,:] = np.pi*self.hankel_kernel_FPM1_all[i,:,:].dot(
                    Apod*self.Apod_w[i]*self.Pupil1d*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM1_lam[i,None]

        iFPM1_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM1_field[i,:]= np.pi*self.hankel_kernel_iFPM1_all[i,:,:].dot(
                    FPM1_field[i,:])*(1/self.gFPM)
    
        FPM2_field = np.zeros((self.nlam, self.nFPM2_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM2_field[i,:] = np.pi*self.hankel_kernel_FPM2_all[i,:,:].dot(
                    Apod*self.Apod_w[i]*self.Pupil1d*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM2_lam[i,None]

        iFPM2_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM2_field[i,:]=np.pi*self.hankel_kernel_iFPM2_all[i,:,:].dot(
                    FPM2_field[i,:])*(1/self.gFPM)
             
        nolyot_field = (Apod[None,:]*self.Apod_w*self.Pupil1d[None,:]\
                        -(self.eps2_t[:,None] - self.eps1_t[:,None])*iFPM1_field \
                        -(1.-self.eps2_t[:, None])*iFPM2_field)*self.r[None,:]/self.R
    
        lyot_field   = nolyot_field*self.LyotStop1d[None,:]
        
        corono_field_tmp = np.zeros((self.nlam,self.nImg+1), dtype='complex128')
        for i in range(self.nlam):
            corono_field_tmp[i,:] = self.hankel_kernel_all[i,:,:].dot(lyot_field[i,:])
        
        return self.lam0/self.lam_t[:,None]*np.pi*corono_field_tmp*self.R/self.nPup
    
#%% 
"""
APLC 2d Coronagraph subclass        
"""
class APLC2d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Apodized Pupil Lyot Coronagraph
    for two-dimension geometry.
    """
    default_params = get_default_params_APLC2d()

    def __init__(self, **kwargs):
        """
        __init__ : method
            Constructor for the APLC2d class.
        
        Attributes
        ----------        
        mB_t : array_like
            Spatial frequency range within the focal plane mask 
            in plane B in lam/D at all the wavelengths
           
        mD_t : array_like
            Spatial frequency range in the final image plane D in lam/D at 
            all the wavelengths
                
        """      
        super().__init__(**kwargs)
        
        # mask size at a given wavelength for SFT
        self.mB_t  = 2.*self.rMask*(self.lam0/self.lam_t)
        self.mD_t  = self.Fmax2d*(self.lam0/self.lam_t)
        
#%% direct propagation (no focal plane mask)
    def compute_direct_field_2d(self,Apod2d):
        """ 
        Computes the coronagraph electric field for a classical Lyot coronagraph
        with four planes (A: entrance pupil, B: intermediate focal plane, 
        C: relayed pupil before stop, L: relayed pupil after stop, 
        D: final image plane).
        Resolution element are given in lam0/D where lam0 and D denote 
        the central and the telescope diameter.
    
        Parameters
        ----------     
        Apod2d : array_like 
            Entrance pupil apodization
            
        Returns    
        ----------    
        field_Dtmp : array_like
            Direct electric field in the final image plane at 
            all the wavelengths
    
        """    
        field_A    = Apod2d*self.Pupil2d
        field_L    = field_A*self.LyotStop2d 
        field_Dtmp = np.zeros((self.nlam,self.nImg2d,self.nImg2d), 
                              dtype='complex128')
        for i in range(self.nlam):
            field_Dtmp[i] = sft(field_L, self.nImg2d, self.mD_t[i], 
                      ctr=self.ctr2)         
    
        return field_Dtmp
 
#%%
    def compute_corono_field_2d(self,Apod2d):
        """
        Computes the coronagraph electric field for a classical Lyot coronagraph
        with four planes (A: entrance pupil, B: intermediate focal plane, 
        C: relayed pupil before stop, L: relayed pupil after stop, 
        D: final image plane).
        Resolution element are given in lam0/D where lam0 and D denote 
        the central and the telescope diameter.
    
        Parameters
        ---------- 
        Apod2d : array_like 
            Entrance pupil apodization
            
        Returns    
        ----------    
        field_Dtmp : array_like
            Coronagraphic electric field in the final image plane at 
            all the wavelengths
            
        """        
        field_A    = Apod2d*self.Pupil2d    
        field_Dtmp = np.zeros((self.nlam,self.nImg2d,self.nImg2d), 
                              dtype='complex128')
        for i in range(self.nlam):
            field_B       = self.mask2d*sft(field_A, self.nFPM, self.mB_t[i], 
                                            ctr=self.ctr)
            field_C       = field_A - isft(field_B, self.nPup, self.mB_t[i], 
                                           ctr=self.ctr)
            field_L       = field_C*self.LyotStop2d
            field_Dtmp[i] = sft(field_L, self.nImg2d, self.mD_t[i], 
                      ctr=self.ctr2)

        return field_Dtmp   

#%%
     