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
from .utils import besselJ0, sft, isft, uniform_disk, radius_disk, sft_even, isft_even

import json

#%% 
"""
Default parameters
"""
#%%
def get_default_params_Coronagraph():
    r"""
    Gets the default parameters for the Coronagraph class.
        
    Parameters
    ---------- 
    PupilObs : float (default=0.14)
        Pupil central obstruction size :math:`d` in fraction of the pupil 
        diameter :math:`D`
        
    LyotStopObs : float (default=0.28)
        Lyot stop central obstruction size :math:`d_S` in fraction of the pupil
        diameter :math:`D`
        
    rho0 : float (default=5)
        Inner edge :math:`\rho_0` of the search area in the coronagraphic image
        
    rho1 : float (default=10)
        Outer edge :math:`\rho_1` of the search area in the coronagraphic image
        
    nPup : int (default=200)
        Sampling across the pupil diameter :math:`D`
        
    nImg : int (default=200)
        Sampling across the final image plane along one axis for 1D simulations
        
    Fmax : float (default=25)
        Maximum spatial frequency in the final image plane for 1D simulations
        
    nImg2d : int (default=44)
        Final image plane sampling for 2D simulations
        
    Fmax2d : float (default=22)
        Maximum spatial frequency in the final image plane for 2D simulations

    lam0 : float (default=1.0)
        Central wavelength :math:`\lambda_0` 

    bw : float (default=0.2)
        Spectral bandwidth :math:`\Delta\lambda/\lambda_0`            
    
    nlam : int (default=5)
        Spectral sampling
  
    R : float (default=1)
        Unitary radius of the pupil
        
    fdir : string (default='')
        Directory
        
    CtrBtwnPix : boolean (default=True)
        Keyword to work with pupil arrays that are centered between four pixels
        from pupil to the coronagraph mask focal plane if True
        
    CtrBtwnPix2 : boolean (default=False)
        Keyword to work with image arrays that are centered between four pixels
        from pupil to the final image plane to if True           

    Pupil2dSym : boolean (default=False)
        Keyword to use faster computation for symmetric pupils
            
    Returns    
    ----------
    tmp : dict
        Dictionary with parameters and all their default values.

    References
    ----------
    .. [1] M. N'Diaye, L. Pueyo, and R. Soummer, Apodized Pupil Lyot Coronagraphs for 
        Arbitrary Apertures. IV. Reduced Inner Working Angle and Increased 
        Robustness to Low-order Aberrations, ApJ 799, 2, 225 (2015).
        
        http://iopscience.iop.org/article/10.1088/0004-637X/799/2/225/meta.
        
    .. [2] M. N'Diaye, R. Soummer, L. Pueyo, A. Carlotti, C. Stark, M. Perrin,
        Apodized Pupil Lyot Coronagraphs for Arbitrary Apertures. V. Hybrid
        Shaped Pupil Designs for Imaging Earth-like planets with Future 
        Space Observatories, ApJ 818, 2, 163 (2016). 
        
        http://iopscience.iop.org/article/10.3847/0004-637X/818/2/163/meta

            
    """
    tmp = {'PupilObs':0.14,'LyotStopObs':0.28,'LyotStopIns':1.0,    
           'rho0':5,'rho1':10,
           'rho0direct':0.7, 'rho1direct':6.0,
           'nPup':200,'nImg':200,'Fmax':25,
           'nImg2d':44, 'Fmax2d':22,
           'bw':0.2,'lam0':1.0,'nlam':5, 'wv':1.593e-6,
           'R':1.0,
           'fdir':'',
           'CtrBtwnPix':True, 'CtrBtwnPix2':False, 
           'Pupil2dSym':False
           }
            
    return tmp

#%%
def get_default_params_APLC1d():
    r"""
    Gets the default parameters for the APLC1d Coronagraph subclass.
        
    Parameters
    ---------- 
    rMask : float
        Focal plane mask (FPM) radius :math:`m/2` in :math:`\lambda_0/D`
        
    nFPM : int
        Mask sampling
            
    Returns    
    ----------
    tmp : dict
        Dictionary that contains all the default values of the 
        Coronagraph class and the APLC1d subclass
    
    """
    tmp = get_default_params_Coronagraph()  
    tmp.update({'rMask':2.8,'nFPM':50, 'corono_name':'APLC'})

    # Pupil radial coordinate
    r   = np.arange(tmp['nPup'])*tmp['R']/tmp['nPup']\
                +tmp['R']/(2*tmp['nPup'])
    # Telescope aperture
    Pupil1d      = (r>tmp['PupilObs'])*1.0
    # Lyot stop 
    LyotStop1d   = (r>tmp['LyotStopObs'])*(r<tmp['LyotStopIns'])*1.0
    
    tmp.update({'r':r, 'Pupil1d':Pupil1d, 'LyotStop1d':LyotStop1d})
    return tmp

#%%
def get_default_params_SP1d():
    r"""
    Gets the default parameters for the SP1d Coronagraph subclass.
        
    Parameters
    ---------- 
    rMask : float
        Focal plane mask (FPM) radius :math:`m/2` in :math:`\lambda_0/D`
        
    nFPM : int
        Mask sampling
            
    Returns    
    ----------
    tmp : dict
        Dictionary that contains all the default values of the 
        Coronagraph class and the APLC1d subclass
    
    """
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask':5.0,'nFPM':50, 'corono_name':'SP'})

    # Pupil radial coordinate    
    r   = np.arange(tmp['nPup'])*tmp['R']/tmp['nPup']\
                +tmp['R']/(2*tmp['nPup'])
    # Telescope aperture
    Pupil1d      = (r>tmp['PupilObs'])*1.0    
    tmp.update({'r':r, 'Pupil1d':Pupil1d})
    
    return tmp


#%%
def get_default_params_DZPM1d():
    r"""
    Gets the default parameters for the DZPM1d Coronagraph subclass.
        
    Parameters
    ---------- 
    rMask1 : float (default=0.875/2)
        Inner part of the focal plane mask radius :math:`m_1/2` in 
        :math:`\lambda_0/D`
        
    rMask2 : float (default=1.453/2)
        Outer part of the focal plane mask radius :math:`m_2/2` in 
        :math:`\lambda_0/D`    
        
    OPDx1 : float (default=0.309)
        Optical path difference :math:`\delta_1/2` in :math:`\lambda_0` 
        for the inner part of the FPM
        
    OPDx2 : float (default=0.672)
        Optical path difference :math:`\delta_2/2` in :math:`\lambda_0` 
        for the outer part of the FPM
        
    ome1 : float (default=-2.340)
        Second order term :math:`\omega_1` for an apodization with amplitude transmission 
        polynomial function
        
    ome2 : float (default=2.051)
        Forth order term :math:`\omega_2` for an apodization with amplitude transmission 
        polynomial function
        
    beta : float (default=-0.236)
        Coefficient :math:`\beta` in :math:`\lambda_0` related to a defocus shift that is applied to the 
        focal plane mask. Equivalent to a phase entrance pupil apodization. 
        
    nFPM : float (default=68.82)
        Mask sampling            
            
    Returns    
    ----------
    tmp : dict
        Dictionary with all the default values of the 
        Coronagraph class and the DZPM1d subclass.

    References
    ----------
    .. [1] R. Soummer, K. Dohlen, and C. Aime, Achromatic dual-zone phase mask 
        stellar coronagraph, A&A 403, 1 (2003).
        
        https://www.aanda.org/articles/aa/abs/2003/19/aa3246/aa3246.html
        
    .. [2] M. N'Diaye, K. Dohlen, S. Cuevas, R. Soummer, C. Sánchez-Pérez,
        F. Zamkotsian, Improved achromatization of phase mask coronagraphs 
        using colored apodization, A&A 538, A55 (2012). 
        
        https://www.aanda.org/articles/aa/abs/2012/02/aa17661-11/aa17661-11.html
        
    .. [3] J. R. Delorme, M. N'Diaye, R. Galicher, K. Dohlen, P. Baudoz, 
        A. Caillat, G. Rousset, R. Soummer, O. Dupuis, Laboratory validation of
        the dual-zone phase mask coronagraph in broadband light at the 
        high-contrast imaging THD testbed, A&A 592, A119 (2016). 
        
        https://www.aanda.org/articles/aa/abs/2016/08/aa28587-16/aa28587-16.html        

    """   
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask1':0.875/2, 'rMask2':1.453/2.,
           'OPDx1':0.309, 'OPDx2':0.672,
           'ome1':-2.340, 'ome2':2.051, 'beta':-0.236,
           'nFPM':68.82312456985547,
           'corono_name':'DZPM'})

    # Pupil radial coordinate        
    r   = np.arange(tmp['nPup'])*tmp['R']/tmp['nPup']\
                +tmp['R']/(2*tmp['nPup'])
    # Telescope aperture
    Pupil1d      = (r>tmp['PupilObs'])*1.0
    # Lyot stop 
    LyotStop1d   = (r>tmp['LyotStopObs'])*(r<tmp['LyotStopIns'])*1.0
    
    tmp.update({'r':r, 'Pupil1d':Pupil1d, 'LyotStop1d':LyotStop1d})
    return tmp

#%%
def get_default_params_HDZPM1d():
    r"""
    Gets the default parameters for the DZPM1d Coronagraph subclass.
    The central part of the FPM is opaque.
        
    Parameters
    ---------- 
    rMask1 : float (default=0.875/2)
        Inner part of the focal plane mask radius :math:`m_1/2` in 
        :math:`\lambda_0/D`
        
    rMask2 : float (default=1.453/2)
        Outer part of the focal plane mask radius :math:`m_2/2` in 
        :math:`\lambda_0/D`            
        
    OPDx2 : float (default=0.672)
        Optical path difference :math:`\delta_2/2` in :math:`\lambda_0` 
        for the outer part of the FPM
        
    ome1 : float (default=-2.340)
        Second order term :math:`\omega_1` for an apodization with amplitude transmission 
        polynomial function
        
    ome2 : float (default=2.051)
        Forth order term :math:`\omega_2` for an apodization with amplitude transmission 
        polynomial function
        
    beta : float (default=-0.236)
        Coefficient :math:`\beta` in :math:`\lambda_0` related to a defocus shift that is applied to the 
        focal plane mask. Equivalent to a phase entrance pupil apodization. 
        
    nFPM : float (default=68.82)
        Mask sampling            
            
    Returns    
    ----------
    tmp : dict
        Dictionary with all the default values of the 
        Coronagraph class and the DZPM1d subclass.

    References
    ----------
    .. [1] R. Soummer, K. Dohlen, and C. Aime, Achromatic dual-zone phase mask 
        stellar coronagraph, A&A 403, 1 (2003).
        
        https://www.aanda.org/articles/aa/abs/2003/19/aa3246/aa3246.html
        
    .. [2] M. N'Diaye, K. Dohlen, S. Cuevas, R. Soummer, C. Sánchez-Pérez,
        F. Zamkotsian, Improved achromatization of phase mask coronagraphs 
        using colored apodization, A&A 538, A55 (2012). 
        
        https://www.aanda.org/articles/aa/abs/2012/02/aa17661-11/aa17661-11.html
        
    .. [3] J. R. Delorme, M. N'Diaye, R. Galicher, K. Dohlen, P. Baudoz, 
        A. Caillat, G. Rousset, R. Soummer, O. Dupuis, Laboratory validation of
        the dual-zone phase mask coronagraph in broadband light at the 
        high-contrast imaging THD testbed, A&A 592, A119 (2016). 
        
        https://www.aanda.org/articles/aa/abs/2016/08/aa28587-16/aa28587-16.html        

    """   
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask1':0.875/2, 'rMask2':1.453/2.,
           'OPDx2':0.672,
           'nFPM':68.82312456985547,
           'corono_name':'HDZPM'})

    # Pupil radial coordinate        
    r   = np.arange(tmp['nPup'])*tmp['R']/tmp['nPup']\
                +tmp['R']/(2*tmp['nPup'])
    # Telescope aperture
    Pupil1d      = (r>tmp['PupilObs'])*1.0
    # Lyot stop 
    LyotStop1d   = (r>tmp['LyotStopObs'])*(r<tmp['LyotStopIns'])*1.0
    
    tmp.update({'r':r, 'Pupil1d':Pupil1d, 'LyotStop1d':LyotStop1d})
    return tmp

#%%
def get_default_params_HTZPM1d():
    r"""
    Gets the default parameters for the TZPM1d Coronagraph subclass.
    The central part of the FPM is opaque.
        
    Parameters
    ---------- 
    rMask1 : float (default=0.875/2)
        Inner part of the focal plane mask radius :math:`m_1/2` in 
        :math:`\lambda_0/D`
        
    rMask2 : float (default=1.453/2)
        Outer part of the focal plane mask radius :math:`m_2/2` in 
        :math:`\lambda_0/D`    

    rMask3 : float (default=1.453/2)
        Outer part of the focal plane mask radius :math:`m_3/2` in 
        :math:`\lambda_0/D`
                
    OPDx2 : float (default=0.672)
        Optical path difference :math:`\delta_2/2` in :math:`\lambda_0` 
        for the outer part of the FPM
        
    OPDx3 : float (default=0.672)
        Optical path difference :math:`\delta_2/2` in :math:`\lambda_0` 
        for the outer part of the FPM
        
    ome1 : float (default=-2.340)
        Second order term :math:`\omega_1` for an apodization with amplitude transmission 
        polynomial function
        
    ome2 : float (default=2.051)
        Forth order term :math:`\omega_2` for an apodization with amplitude transmission 
        polynomial function
        
    beta : float (default=-0.236)
        Coefficient :math:`\beta` in :math:`\lambda_0` related to a defocus shift that is applied to the 
        focal plane mask. Equivalent to a phase entrance pupil apodization. 
        
    nFPM : float (default=68.82)
        Mask sampling            
            
    Returns    
    ----------
    tmp : dict
        Dictionary with all the default values of the 
        Coronagraph class and the DZPM1d subclass.

    References
    ----------
    .. [1] R. Soummer, K. Dohlen, and C. Aime, Achromatic dual-zone phase mask 
        stellar coronagraph, A&A 403, 1 (2003).
        
        https://www.aanda.org/articles/aa/abs/2003/19/aa3246/aa3246.html
        
    .. [2] M. N'Diaye, K. Dohlen, S. Cuevas, R. Soummer, C. Sánchez-Pérez,
        F. Zamkotsian, Improved achromatization of phase mask coronagraphs 
        using colored apodization, A&A 538, A55 (2012). 
        
        https://www.aanda.org/articles/aa/abs/2012/02/aa17661-11/aa17661-11.html
        
    .. [3] J. R. Delorme, M. N'Diaye, R. Galicher, K. Dohlen, P. Baudoz, 
        A. Caillat, G. Rousset, R. Soummer, O. Dupuis, Laboratory validation of
        the dual-zone phase mask coronagraph in broadband light at the 
        high-contrast imaging THD testbed, A&A 592, A119 (2016). 
        
        https://www.aanda.org/articles/aa/abs/2016/08/aa28587-16/aa28587-16.html        

    """   
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask1':0.875/2, 'rMask2':1.453/2., 'rMask3':2.1/2,
           'OPDx2':0.672, 'OPDx3':0.8,
           'nFPM':68.82312456985547,
           'corono_name':'HTZPM'})

    # Pupil radial coordinate        
    r   = np.arange(tmp['nPup'])*tmp['R']/tmp['nPup']\
                +tmp['R']/(2*tmp['nPup'])
    # Telescope aperture
    Pupil1d      = (r>tmp['PupilObs'])*1.0
    # Lyot stop 
    LyotStop1d   = (r>tmp['LyotStopObs'])*(r<tmp['LyotStopIns'])*1.0
    
    tmp.update({'r':r, 'Pupil1d':Pupil1d, 'LyotStop1d':LyotStop1d})
    return tmp


#%%
def get_default_params_APLC2d():
    r"""
    Gets the default parameters for the APLC2d Coronagraph subclass.
        
    Parameters
    ---------- 
    rMask : float
        Focal plane mask (FPM) radius :math:`m/2` in :math:`\lambda_0/D`
    
    nPup : int
        Sampling across the pupil diameter :math:`D`
    
    nFPM : int
        Sampling across the mask diameter :math:`m`

    Pupil2d : array_like     
        2D entrance pupil :math:`P_0`

    LyotStop2d : array_like 
        2D Lyot stop :math:`L`
    
    OPDmap2d : array_like
        2D OPD phase map :math:`\delta`
        
    Returns    
    ----------
    tmp : dict
        Dictionary with all the default values of the 
        Coronagraph class and the APLC2d subclass
        
    """    
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask':2.8,
                'nPup':50, 'nFPM':25,
                'corono_name':'APLC'})
                
    # Telescope aperture
    Pupil2d      = uniform_disk(tmp['nPup'], tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])\
        - uniform_disk(tmp['nPup'], tmp['PupilObs']*tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])
    # Lyot stop 
    LyotStop2d   = uniform_disk(tmp['nPup'], tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])\
        - uniform_disk(tmp['nPup'], tmp['LyotStopObs']*tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])
    # OPD map
    OPDmap2d     = None
    Ampmap2d     = None        
        
    tmp.update({'Pupil2d':Pupil2d, 'LyotStop2d':LyotStop2d, 
                'OPDmap2d':OPDmap2d, 'Ampmap2d': Ampmap2d})
    return tmp

#%%
def get_default_params_SP2d():
    r"""
    Gets the default parameters for the SP2d Coronagraph subclass.
        
    Parameters
    ---------- 
    rMask : float
        Focal plane mask (FPM) radius :math:`m/2` in :math:`\lambda_0/D`
    
    nPup : int
        Sampling across the pupil diameter :math:`D`
    
    nFPM : int
        Sampling across the mask diameter :math:`m`
        
    Pupil2d : array_like     
        2D entrance pupil :math:`P_0`
            
    Returns    
    ----------
    tmp : dict
        Dictionary with all the default values of the 
        Coronagraph class and the APLC2d subclass
        
    """    
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask':2.8,
                'nPup':50, 'nFPM':25, 
                'corono_name':'SP'
                })
    # Telescope aperture
    Pupil2d      = uniform_disk(tmp['nPup'], tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])\
        - uniform_disk(tmp['nPup'], tmp['PupilObs']*tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])
    tmp.update({'Pupil2d':Pupil2d})
    return tmp

#%%
def get_default_params_DZPM2d():
    r"""
    Gets the default parameters for the DZPM1d Coronagraph subclass.
        
    Parameters
    ---------- 
    rMask1 : float (default=0.875/2)
        Inner part of the focal plane mask radius :math:`m_1/2` in 
        :math:`\lambda_0/D`
        
    rMask2 : float (default=1.453/2)
        Outer part of the focal plane mask radius :math:`m_2/2` in 
        :math:`\lambda_0/D`    
        
    OPDx1 : float (default=0.309)
        Optical path difference :math:`\delta_1/2` in :math:`\lambda_0` 
        for the inner part of the FPM
        
    OPDx2 : float (default=0.672)
        Optical path difference :math:`\delta_2/2` in :math:`\lambda_0` 
        for the outer part of the FPM
        
    ome1 : float (default=-2.340)
        Second order term :math:`\omega_1` for an apodization with amplitude transmission 
        polynomial function
        
    ome2 : float (default=2.051)
        Forth order term :math:`\omega_2` for an apodization with amplitude transmission 
        polynomial function
        
    beta : float (default=-0.236)
        Coefficient :math:`\beta` in :math:`\lambda_0` related to a defocus shift that is applied to the 
        focal plane mask. Equivalent to a phase entrance pupil apodization. 
        
    nFPM : float (default=68.82)
        Mask sampling            

    Pupil2d : array_like     
        2D entrance pupil :math:`P_0`

    LyotStop2d : array_like 
            2D Lyot stop :math:`L`
            
    Returns    
    ----------
    tmp : dict
        Dictionary with all the default values of the 
        Coronagraph class and the DZPM1d subclass.

    References
    ----------
    .. [1] R. Soummer, K. Dohlen, and C. Aime, Achromatic dual-zone phase mask 
        stellar coronagraph, A&A 403, 1 (2003).
        
        https://www.aanda.org/articles/aa/abs/2003/19/aa3246/aa3246.html
        
    .. [2] M. N'Diaye, K. Dohlen, S. Cuevas, R. Soummer, C. Sánchez-Pérez,
        F. Zamkotsian, Improved achromatization of phase mask coronagraphs 
        using colored apodization, A&A 538, A55 (2012). 
        
        https://www.aanda.org/articles/aa/abs/2012/02/aa17661-11/aa17661-11.html
        
    .. [3] J. R. Delorme, M. N'Diaye, R. Galicher, K. Dohlen, P. Baudoz, 
        A. Caillat, G. Rousset, R. Soummer, O. Dupuis, Laboratory validation of
        the dual-zone phase mask coronagraph in broadband light at the 
        high-contrast imaging THD testbed, A&A 592, A119 (2016). 
        
        https://www.aanda.org/articles/aa/abs/2016/08/aa28587-16/aa28587-16.html        

    """   
    tmp = get_default_params_Coronagraph()
    tmp.update({'rMask1':0.875/2, 'rMask2':1.453/2.,
           'OPDx1':0.309, 'OPDx2':0.672,
           'ome1':-2.340, 'ome2':2.051, 'beta':-0.236,
           'nFPM':25, 'nPup':50,
           'corono_name':'DZPM'})
        
    # Telescope aperture
    Pupil2d      = uniform_disk(tmp['nPup'], tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])\
        - uniform_disk(tmp['nPup'], tmp['PupilObs']*tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])
    # Lyot stop 
    LyotStop2d   = uniform_disk(tmp['nPup'], tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])\
        - uniform_disk(tmp['nPup'], tmp['LyotStopObs']*tmp['nPup']/2., CtrBtwnPix=tmp['CtrBtwnPix'])
    
    tmp.update({'Pupil2d':Pupil2d, 'LyotStop2d':LyotStop2d})
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
        r"""
        __init__ : method
            Builds the constructor for the Coronagraph class.
        
        Attributes
        ----------            
        params : dict
            Dictionary of parameters for the Coronagraph class
            
        lam0 : float
            Central wavelength :math:`\lambda_0`
        
        bw : float 
            Spectral bandwidth :math:`\Delta\lambda/\lambda_0`
        
        dlam : float 
            Spectral bandwidth :math:`\Delta\lambda`
            
        r : array_like 
            vector containing the Pupil radial coordinate :math:`r`
            
        nPup : float 
            Sampling across the pupil diameter :math:`D`
            
        R : float 
            Unitary radius of the pupil :math:`P_0`

        PupilObs : float
            Central obstruction size :math:`d` for the pupil :math:`P_0` 
            in pupil diameter :math:`D`
        
        LyotStopObs : float
            Central obstruction size :math:`d_S` for the Lyot stop :math:`L` 
            in pupil diameter :math:`D`
        
        ClearPupil1d : array_like 
            Clear pupil (no obstruction nor spiders)
        
        Pupil1d : array_like
            1D Entrance pupil :math:`P_0`
        
        LyotStop1d : array_like 
            1D Lyot Stop :math:`L`
        
        xi : array_like 
            Vector for the final image pupil plane coordinate :math:`\xi`
        
        xii : array_like 
            Vector for the final image pupil plane coordinate :math:`\xi` 
            that is weighted with wavelength
        
        Apod1d_t : array_like 
            Identity matrix for the computation of the response matrix for all
            the points in the Entrance pupil plane for 1D problem
        
        hankel_kernel : array_like
            Hankel kernel without wavelength variation
        
        hankel_kernel_all : 3D array
            Hankel kernel including wavelength variation
        
        ClearPupil2d : array_like 
            2D clear pupil
             
        mask2d : array_like 
            2D focal plane mask :math:`M`
                
        xi2d : array_like 
            Final image plane coordinate vector centered on a pixel
            for 2D problem
        
        xi2d_ctr : array_like
            Final image plane coordinate vector centered between 4 pixels
            for 2D problem
        


        References
        ----------
        .. [1] M. N'Diaye, L. Pueyo, and R. Soummer, Apodized Pupil Lyot Coronagraphs for 
            Arbitrary Apertures. IV. Reduced Inner Working Angle and Increased 
            Robustness to Low-order Aberrations, ApJ 799, 2, 225 (2015).
            
            http://iopscience.iop.org/article/10.1088/0004-637X/799/2/225/meta.
            
        .. [2] M. N'Diaye, R. Soummer, L. Pueyo, A. Carlotti, C. Stark, M. Perrin,
            Apodized Pupil Lyot Coronagraphs for Arbitrary Apertures. V. Hybrid
            Shaped Pupil Designs for Imaging Earth-like planets with Future 
            Space Observatories, ApJ 818, 2, 163 (2016). 
            
            http://iopscience.iop.org/article/10.3847/0004-637X/818/2/163/meta

        
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
        
        # Final image plane coordinate
        self.xi  = np.arange(self.nImg+1)*self.Fmax/self.nImg
        # final image plane coordinate weighted with wavelength
        self.xii = self.xi[None,:]*self.lam0/self.lam_t[:,None]

        # Hankel kernel (no wavelength variation)
        self.hankel_kernel     = besselJ0(
                np.pi/self.R*self.xi[:,None]*self.r[None,:])
        # Hankel kernel (including wavelength variation)
        self.hankel_kernel_all = besselJ0(
                np.pi/self.R*self.xii[:,:,None]*self.r[None,None,:])

        # clear Pupil
        self.ClearPupil2d = uniform_disk(self.nPup, self.nPup/2., CtrBtwnPix=self.CtrBtwnPix)
        # Focal plane mask
        self.mask2d       = uniform_disk(self.nFPM, self.nFPM/2., CtrBtwnPix=self.CtrBtwnPix)

        # Final image plane coordinate
        val = 0
        if self.nImg2d%2 == 0:
            val = 1/2        
        self.xi2d     = (np.arange(self.nImg2d//2+1))* self.Fmax2d/self.nImg2d
        self.xi2d_ctr = (np.arange(self.nImg2d//2)+val)* self.Fmax2d/self.nImg2d
        
        
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

#%% # direct propagation (no focal plane mask)
    def compute_nostop_field_1d(self):
        """
        Computes the electric field of the direct image with APLC
        for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Direct electric field :math:`\Psi_0` at all the wavelengths
            
        """
        return self.lam0/self.lam_t[:,None]*np.pi*self.hankel_kernel_all.dot(
                self.Pupil1d*self.r/self.R)*self.R/self.nPup


#%% direct signal in intensity
    def compute_nostop_intensity_1d(self,poly=True):
        r"""
        Computes the intensity of the direct image for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization :math:`\Phi`
        
        poly : boolean (default=True)
            parameter to compute broadband image or monochromatic images
            at all the wavelengths
                
        Returns    
        ----------
        res : array_like
            Intensity of the direct broadband image or monochromatic images. 
            
        """
        nostop_field = self.compute_nostop_field_1d()
        
        if poly:
            return np.sum(np.abs(nostop_field)**2,0)
        else:
            return np.abs(nostop_field)**2

#%% direct signal in intensity
    def compute_direct_intensity_1d(self,Apod,poly=True):
        r"""
        Computes the intensity of the direct image for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization :math:`\Phi`
        
        poly : boolean (default=True)
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
            Entrance pupil apodization :math:`\Phi`
            
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
            print('Warning: multi-wavelength normalization assuming flat spectrum!')
            return np.abs(corono_field)**2       

#%% direct signal in intensity
    def compute_direct_intensity_2d(self,Apod2d, poly=True):
        """
        Computes the intensity of the direct image for the 2D problem.
        
        Parameters
        ---------- 
        Apod2d : array_like
            Entrance pupil apodization :math:`\Phi`
        
        poly : thuth value (default=True)
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
    def compute_corono_intensity_2d(self,Apod2d, poly=True):
        """
        Computes the intensity of the coronagraphic image for the 2D problem.
        
        Parameters
        ---------- 
        Apod2d : array_like
            Entrance pupil apodization :math:`\Phi`
            
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
            print('Warning: multi-wavelength normalization assuming flat spectrum!')
            return np.abs(corono_field_2d)**2

#%%
    def compute_direct_field_2d_vec(self,Apod2d):
        """
        Computes the electric field of the direct image 
        for the vectorized 2D problem.
        
        Parameters
        ---------- 
        Apod2d : array_like
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like, array_like
            Real and imag parts of the direct electric field :math:`\Psi_0`
            
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
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like, array_like
            Real and imag parts of the coronagraphic electric field  :math:`\Psi_D`
            
        """ 
#        test = self.compute_corono_field_2d(Apod2d, Pupil2d, LyotStop2d)
        test = self.compute_corono_field_2d(Apod2d)
        test_re = np.reshape(test.real, (self.nlam, self.nImg2d**2))
        test_im = np.reshape(test.imag, (self.nlam, self.nImg2d**2))
        return test_re, test_im

#%%
    def generate_area(self,):
        """
        Computes the list of points for a given search area in the final image 
        plane of the coronagraph. The area is defined by an annulus with minimum
        and maximum angular separation from the star image.
    
        Parameters
        ----------     
        mD : float
            Spatial frequency range in the final image plane D in :math:`\lambda_0/D`
            
        nImg2d : int
            Linear number of points in the final image plane D
            
        sep_min : float
            Minimum angular separation from the star for the area
            in :math:`\lambda_0/D`
            
        sep_max : float
            Maximum angular separation from the star for the area
            in :math:`\lambda_0/D`
    
        Returns    
        ----------
        res : array_like
            2D array with 1 and 0 for the points inside and outside the area 
            in the coronagraphic image
    
        """
        val = 0
        if self.nImg2d%2 == 0:
            val = 1/2
        
        # array of angular distances in the final image plane
        xx,yy  = np.meshgrid(np.arange(self.nImg2d)-self.nImg2d//2+val, np.arange(self.nImg2d)-self.nImg2d//2+val)
        mydist = (self.Fmax2d/self.nImg2d)*np.hypot(yy,xx)        
        res = (mydist <= self.rho1)*(mydist >= self.rho0)
        if self.Pupil2dSym == True:
            res *= (xx >= 0)*(yy >= 0)
        return res, mydist[res]
    
#%% 
"""
APLC 1d Coronagraph subclass
"""
class APLC1d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Apodized Pupil Lyot Coronagraph
    (APLC) for 1D geometry.
    """
    default_params = get_default_params_APLC1d()
    
    def __init__(self, **kwargs):
        """
        __init__ : method
            Builds the constructor for the APLC1d class
        
        Attributes:
        ----------  
        rMask_t : array_like
            Focal plane mask (FPM) radius :math:`m` scaled with wavelength
            :math:`\lambda`
            
        nFPM_t : array_like
            Mask sampling at a given wavelength :math:`\lambda`
            
        nFPM_max : float
            Maximum mask sampling over all the wavelengths
            
        mask_lam : array_like
            Focal plane mask in :math:`\lambda/D` unit
            
        xi_FPM_lam : array_like
            Focal plane mask coordinate in :math:`\lambda/D`
            
        hankel_kernel_FPM_all : array_like
            Hankel kernel for the FPM at all the wavelengths
            
        hankel_kernel_iFPM_all : array_like
            Inverse hankel transform for the FPM at all the 
            wavelengths

        """      
        super(APLC1d,self).__init__(**kwargs)      
        
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
#        t0 = time.time()
        self.hankel_kernel_FPM_all  = besselJ0(
                np.pi/self.R*self.xi_FPM_lam[:,:,None]*self.r[None,None,:])
        self.hankel_kernel_iFPM_all = besselJ0(
                np.pi/self.R*self.xi_FPM_lam[:,None,:]*self.r[None,:,None])
        
#        self.hankel_kernel_FPM_all  = besselJ0(
#                np.pi/self.R*np.einsum('ij,k -> ijk', self.xi_FPM_lam, self.r))
#        self.hankel_kernel_iFPM_all = besselJ0(
#                np.pi/self.R*np.einsum('ik,j -> ijk', self.xi_FPM_lam, self.r))

        
#%% # direct propagation (no focal plane mask)
    def compute_direct_field_1d(self,Apod):
        """
        Computes the electric field of the direct image with APLC
        for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Direct electric field :math:`\Psi_0` at all the wavelengths
            
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
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Coronagraphic electric field :math:`\Psi_D` at all the wavelengths
            
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

#%% # propagation through coronagraph (with focal plane mask)
    def compute_corono_field_1d0(self,Apod):
        """
        Computes the electric field of the coronagraphic image with APLC
        for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Coronagraphic electric field :math:`\Psi_D` at all the wavelengths
            
        """
        FPM_field  = np.pi*self.hankel_kernel_FPM_all.dot(
                Apod*self.Pupil1d*self.r/self.R)\
                *(self.R/self.nPup)*self.xi_FPM_lam
                
        iFPM_field = np.pi*(1/self.nFPM)*np.einsum('ijk,ik -> ij', 
                           self.hankel_kernel_iFPM_all, FPM_field)
                
        lyot_field = (Apod[None,:]*self.Pupil1d[None,:]-iFPM_field)\
                *self.r[None,:]/self.R*self.LyotStop1d[None, :]
                
        corono_field_tmp = np.einsum('ij,ikj -> ik', lyot_field, self.hankel_kernel_all)
        
        return self.lam0/self.lam_t[:,None]*np.pi\
            *corono_field_tmp*self.R/self.nPup

            
#%% 
"""
SP 1d Coronagraph subclass
"""
class SP1d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Shaped Pupil Coronagraph
    (SP) for 1D geometry.
    """
    default_params = get_default_params_SP1d()
    
    def __init__(self, **kwargs):
        """
        __init__ : method
            Builds the constructor for the SP1d class
        
        Attributes:
        ----------  
        rMask_t : array_like
            Focal plane mask (FPM) radius :math:`m` scaled with wavelength
            :math:`\lambda`
            
        nFPM_t : array_like
            Mask sampling at a given wavelength :math:`\lambda`
            
        nFPM_max : float
            Maximum mask sampling over all the wavelengths
            
        mask_lam : array_like
            Focal plane mask in :math:`\lambda/D` unit
            
        xi_FPM_lam : array_like
            Focal plane mask coordinate in :math:`\lambda/D`
            
        hankel_kernel_FPM_all : array_like
            Hankel kernel for the FPM at all the wavelengths
            
        hankel_kernel_iFPM_all : array_like
            Inverse hankel transform for the FPM at all the 
            wavelengths

        """      
        super(SP1d,self).__init__(**kwargs)      
        
        # mask size at Apod given wavelength
        self.rMask_t    = (self.lam0/self.lam_t)*self.rMask
        
        # mask sampling at Apod given wavelength and max nFPM_max 
        self.nFPM_t     = self.rMask_t*self.nFPM
        self.nFPM_max   = int(np.max(self.nFPM_t))
        self.mask_lam   = (np.arange(self.nFPM_max+1)[None,:]\
                           <self.rMask_t[:,None]*self.nFPM)
        self.xi_FPM_lam = np.arange(self.nFPM_max+1)[None,:]\
                *self.mask_lam/self.nFPM
        
#%% # direct propagation (no focal plane mask)
    def compute_direct_field_1d(self,Apod):
        """
        Computes the electric field of the direct image with APLC
        for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Direct electric field :math:`\Psi_0` at all the wavelengths
            
        """
        return self.lam0/self.lam_t[:,None]*np.pi*self.hankel_kernel_all.dot(
                self.Pupil1d*Apod*self.r/self.R)*self.R/self.nPup

#%% # propagation through coronagraph (with focal plane mask)
    def compute_corono_field_1d(self,Apod):
        """
        Computes the electric field of the coronagraphic image with SP
        for the 1D problem.
        
        Parameters
        ---------- 
        Apod : array_like
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Coronagraphic electric field :math:`\Psi_D` at all the wavelengths
            
        """
        return self.lam0/self.lam_t[:,None]*np.pi*self.hankel_kernel_all.dot(
                self.Pupil1d*Apod*self.r/self.R)*self.R/self.nPup
            


#%% 
"""
DZPM 1d Coronagraph class     
"""
class DZPM1d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Dual Zone Phase Mask (DZPM) 
    Coronagraph for one-dimension geometry.
    """    
    default_params = get_default_params_DZPM1d()    

    def __init__(self, **kwargs):
        r"""
        __init__ : method
            Constructor for the DZPM1d class
        
        Attributes
        ----------      
        OPD1 : float
            Optical path difference :math:`\delta_1` 
            for the inner part of the focal plane mask (FPM)
            
        OPD2 : float
            Optical path difference :math:`\delta_2` 
            for the outer part of the mask
            
        phi1_t : array_like
            Phase shift :math:`\varphi_1` induced by the inner part of the mask
            at all the wavelengths

        phi2_t : array_like
            Phase shift :math:`\varphi_2` induced by the outer part of the mask
            at all the wavelengths
        
        eps1_t : array_like
            Phasor :math:`\varepsilon_1` induced by the inner part of the mask 
            at all the wavelengths

        eps2_t : array_like
            Phasor :math:`\varepsilon_2` induced by the outer part of the mask 
            at all the wavelengths
        
        rMask1_t : array_like
            FPM inner part radius :math:`m_1/2` scaled with 
            wavelength :math:`\lambda/D`
            
        rMask2_t : array_like
            FPM outer part radius :math:`m_2/2` scaled with 
            wavelength :math:`\lambda/D`
            
        nFPM1_t : array_like
            Inner mask sampling at a given wavelength :math:`\lambda/D`
        
        nFPM2_t : array_like
            Outer mask sampling at a given wavelength :math:`\lambda/D`
        
        nFPM1_max : float
            Maximum inner mask sampling over all the wavelengths
            
        nFPM2_max : float
            Maximum outer mask sampling over all the wavelengths
        
        mask1_lam : array_like
            Inner focal plane mask in :math:`\lambda/D` unit
        
        mask2_lam : array_like
            Outer focal plane mask in :math:`\lambda/D` unit
        
        xi_FPM1_lam : array_like
            Inner focal plane mask coordinate in :math:`\lambda/D`
        
        xi_FPM2_lam : array_like
            Outer focal plane mask coordinate in :math:`\lambda/D`       
 
        hankel_kernel_FPM1_all : array_like
            Hankel kernel for the inner FPM at all the wavelengths
        
        hankel_kernel_FPM2_all : array_like
            Hankel kernel for the outer FPM at all the wavelengths
        
        hankel_kernel_iFPM1_all : array_like
            Inverse hankel transform for the inner FPM at all the wavelengths
            
        hankel_kernel_iFPM2_all : array_like
            Inverse hankel transform for the outer FPM at all the wavelengths
            
        Apod_w : array_like
            Complex apodization :math:`\Phi_\omega` at all the wavelengths

        References
        ----------
        .. [1] R. Soummer, K. Dohlen, and C. Aime, Achromatic dual-zone phase mask 
            stellar coronagraph, A&A 403, 1 (2003).
            
            https://www.aanda.org/articles/aa/abs/2003/19/aa3246/aa3246.html
            
        .. [2] M. N'Diaye, K. Dohlen, S. Cuevas, R. Soummer, C. Sánchez-Pérez,
            F. Zamkotsian, Improved achromatization of phase mask coronagraphs 
            using colored apodization, A&A 538, A55 (2012). 
            
            https://www.aanda.org/articles/aa/abs/2012/02/aa17661-11/aa17661-11.html
            
        .. [3] J. R. Delorme, M. N'Diaye, R. Galicher, K. Dohlen, P. Baudoz, 
            A. Caillat, G. Rousset, R. Soummer, O. Dupuis, Laboratory validation of
            the dual-zone phase mask coronagraph in broadband light at the 
            high-contrast imaging THD testbed, A&A 592, A119 (2016). 
            
            https://www.aanda.org/articles/aa/abs/2016/08/aa28587-16/aa28587-16.html        


        """
        super(DZPM1d,self).__init__(**kwargs)
        
        # Optical path difference introduced by the mask
        self.OPD1 = self.OPDx1*self.lam0
        self.OPD2 = self.OPDx2*self.lam0
        
        # phase shift induced by the mask
        self.phi1_t = 2.*np.pi*self.OPD1/self.lam_t
        self.phi2_t = 2.*np.pi*self.OPD2/self.lam_t
        
        self.eps1_t   = 1j*np.sin(self.phi1_t) + np.cos(self.phi1_t)
        self.eps2_t   = 1j*np.sin(self.phi2_t) + np.cos(self.phi2_t)

        # mask size at a given wavelength
        self.rMask1_t = (self.lam0/self.lam_t)*self.rMask1
        self.rMask2_t = (self.lam0/self.lam_t)*self.rMask2
        
        # mask sampling at a given wavelength and max nFPM_max 
        self.nFPM1_t   = self.rMask1_t*self.nFPM
        self.nFPM1_max = int(np.max(self.nFPM1_t))     
        self.nFPM2_t   = self.rMask2_t*self.nFPM
        self.nFPM2_max = int(np.max(self.nFPM2_t))

        self.mask1_lam   = (np.arange(self.nFPM1_max+1)[None,:]\
                            <=self.rMask1_t[:,None]*self.nFPM)
        self.xi_FPM1_lam = np.arange(self.nFPM1_max+1)[None,:]\
        *self.mask1_lam/self.nFPM       
        self.mask2_lam   = (np.arange(self.nFPM2_max+1)[None,:]\
                            <=self.rMask2_t[:,None]*self.nFPM)
        self.xi_FPM2_lam = np.arange(self.nFPM2_max+1)[None,:]\
        *self.mask2_lam/self.nFPM

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
        r"""
        Computes the electric field of the direct image with DZPM
        for 1D problem.
        
        Parameters
        ---------- 
        Apod: array_like
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Direct electric field :math:`\Phi_0` at all the wavelengths
            
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
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Coronagraphic electric field :math:`\Phi_D` at all the wavelengths
            
        """
        FPM1_field = np.zeros((self.nlam, self.nFPM1_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM1_field[i,:] = np.pi*self.hankel_kernel_FPM1_all[i,:,:].dot(
                    Apod*self.Apod_w[i]*self.Pupil1d*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM1_lam[i,None]

        iFPM1_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM1_field[i,:]= np.pi*self.hankel_kernel_iFPM1_all[i,:,:].dot(
                    FPM1_field[i,:])*(1/self.nFPM)
    
        FPM2_field = np.zeros((self.nlam, self.nFPM2_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM2_field[i,:] = np.pi*self.hankel_kernel_FPM2_all[i,:,:].dot(
                    Apod*self.Apod_w[i]*self.Pupil1d*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM2_lam[i,None]

        iFPM2_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM2_field[i,:]=np.pi*self.hankel_kernel_iFPM2_all[i,:,:].dot(
                    FPM2_field[i,:])*(1/self.nFPM)
             
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
DZPM 1d Coronagraph class     
"""
class HDZPM1d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Dual Zone Phase Mask (DZPM) 
    Coronagraph for one-dimension geometry.
    """    
    default_params = get_default_params_HDZPM1d()    

    def __init__(self, **kwargs):
        r"""
        __init__ : method
            Constructor for the DZPM1d class
        
        Attributes
        ----------      
        OPD1 : float
            Optical path difference :math:`\delta_1` 
            for the inner part of the focal plane mask (FPM)
            
        OPD2 : float
            Optical path difference :math:`\delta_2` 
            for the outer part of the mask
            
        phi1_t : array_like
            Phase shift :math:`\varphi_1` induced by the inner part of the mask
            at all the wavelengths

        phi2_t : array_like
            Phase shift :math:`\varphi_2` induced by the outer part of the mask
            at all the wavelengths
        
        eps1_t : array_like
            Phasor :math:`\varepsilon_1` induced by the inner part of the mask 
            at all the wavelengths

        eps2_t : array_like
            Phasor :math:`\varepsilon_2` induced by the outer part of the mask 
            at all the wavelengths
        
        rMask1_t : array_like
            FPM inner part radius :math:`m_1/2` scaled with 
            wavelength :math:`\lambda/D`
            
        rMask2_t : array_like
            FPM outer part radius :math:`m_2/2` scaled with 
            wavelength :math:`\lambda/D`
            
        nFPM1_t : array_like
            Inner mask sampling at a given wavelength :math:`\lambda/D`
        
        nFPM2_t : array_like
            Outer mask sampling at a given wavelength :math:`\lambda/D`
        
        nFPM1_max : float
            Maximum inner mask sampling over all the wavelengths
            
        nFPM2_max : float
            Maximum outer mask sampling over all the wavelengths
        
        mask1_lam : array_like
            Inner focal plane mask in :math:`\lambda/D` unit
        
        mask2_lam : array_like
            Outer focal plane mask in :math:`\lambda/D` unit
        
        xi_FPM1_lam : array_like
            Inner focal plane mask coordinate in :math:`\lambda/D`
        
        xi_FPM2_lam : array_like
            Outer focal plane mask coordinate in :math:`\lambda/D`       
 
        hankel_kernel_FPM1_all : array_like
            Hankel kernel for the inner FPM at all the wavelengths
        
        hankel_kernel_FPM2_all : array_like
            Hankel kernel for the outer FPM at all the wavelengths
        
        hankel_kernel_iFPM1_all : array_like
            Inverse hankel transform for the inner FPM at all the wavelengths
            
        hankel_kernel_iFPM2_all : array_like
            Inverse hankel transform for the outer FPM at all the wavelengths
            

        References
        ----------
        .. [1] R. Soummer, K. Dohlen, and C. Aime, Achromatic dual-zone phase mask 
            stellar coronagraph, A&A 403, 1 (2003).
            
            https://www.aanda.org/articles/aa/abs/2003/19/aa3246/aa3246.html
            
        .. [2] M. N'Diaye, K. Dohlen, S. Cuevas, R. Soummer, C. Sánchez-Pérez,
            F. Zamkotsian, Improved achromatization of phase mask coronagraphs 
            using colored apodization, A&A 538, A55 (2012). 
            
            https://www.aanda.org/articles/aa/abs/2012/02/aa17661-11/aa17661-11.html
            
        .. [3] J. R. Delorme, M. N'Diaye, R. Galicher, K. Dohlen, P. Baudoz, 
            A. Caillat, G. Rousset, R. Soummer, O. Dupuis, Laboratory validation of
            the dual-zone phase mask coronagraph in broadband light at the 
            high-contrast imaging THD testbed, A&A 592, A119 (2016). 
            
            https://www.aanda.org/articles/aa/abs/2016/08/aa28587-16/aa28587-16.html        


        """
        super(HDZPM1d,self).__init__(**kwargs)
        
        # Optical path difference introduced by the mask
        self.OPD2 = self.OPDx2*self.lam0
        
        # phase shift induced by the mask
        self.phi2_t = 2.*np.pi*self.OPD2/self.lam_t
        
        self.eps2_t   = 1j*np.sin(self.phi2_t) + np.cos(self.phi2_t)

        # mask size at a given wavelength
        self.rMask1_t = (self.lam0/self.lam_t)*self.rMask1
        self.rMask2_t = (self.lam0/self.lam_t)*self.rMask2
        
        # mask sampling at a given wavelength and max nFPM_max 
        self.nFPM1_t   = self.rMask1_t*self.nFPM
        self.nFPM1_max = int(np.max(self.nFPM1_t))     
        self.nFPM2_t   = self.rMask2_t*self.nFPM
        self.nFPM2_max = int(np.max(self.nFPM2_t))

        self.mask1_lam   = (np.arange(self.nFPM1_max+1)[None,:]\
                            <=self.rMask1_t[:,None]*self.nFPM)
        self.xi_FPM1_lam = np.arange(self.nFPM1_max+1)[None,:]\
        *self.mask1_lam/self.nFPM       
        self.mask2_lam   = (np.arange(self.nFPM2_max+1)[None,:]\
                            <=self.rMask2_t[:,None]*self.nFPM)
        self.xi_FPM2_lam = np.arange(self.nFPM2_max+1)[None,:]\
        *self.mask2_lam/self.nFPM

        self.hankel_kernel_FPM1_all  = besselJ0(
                np.pi/self.R*self.xi_FPM1_lam[:,:,None]*self.r[None,None,:])
        self.hankel_kernel_iFPM1_all = besselJ0(
                np.pi/self.R*self.xi_FPM1_lam[:,None,:]*self.r[None,:,None])
        self.hankel_kernel_FPM2_all  = besselJ0(
                np.pi/self.R*self.xi_FPM2_lam[:,:,None]*self.r[None,None,:])
        self.hankel_kernel_iFPM2_all = besselJ0(
                np.pi/self.R*self.xi_FPM2_lam[:,None,:]*self.r[None,:,None])



#%% direct propagation (no focal plane mask)
    def compute_direct_field_1d(self,Apod):
        r"""
        Computes the electric field of the direct image with DZPM
        for 1D problem.
        
        Parameters
        ---------- 
        Apod: array_like
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Direct electric field :math:`\Phi_0` at all the wavelengths
            
        """
        field = np.zeros((self.nlam,self.nImg+1), dtype='complex128')
        for i in range(self.nlam):
            field[i,:] = \
            self.lam0/self.lam_t[i]*np.pi*self.hankel_kernel_all[i].dot(
                    self.Pupil1d*Apod*self.LyotStop1d*self.r/self.R)\
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
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Coronagraphic electric field :math:`\Phi_D` at all the wavelengths
            
        """
        FPM1_field = np.zeros((self.nlam, self.nFPM1_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM1_field[i,:] = np.pi*self.hankel_kernel_FPM1_all[i,:,:].dot(
                    Apod*self.Pupil1d*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM1_lam[i,None]

        iFPM1_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM1_field[i,:]= np.pi*self.hankel_kernel_iFPM1_all[i,:,:].dot(
                    FPM1_field[i,:])*(1/self.nFPM)
    
        FPM2_field = np.zeros((self.nlam, self.nFPM2_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM2_field[i,:] = np.pi*self.hankel_kernel_FPM2_all[i,:,:].dot(
                    Apod*self.Pupil1d*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM2_lam[i,None]

        iFPM2_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM2_field[i,:]=np.pi*self.hankel_kernel_iFPM2_all[i,:,:].dot(
                    FPM2_field[i,:])*(1/self.nFPM)
             
        nolyot_field = (Apod[None,:]*self.Pupil1d[None,:]\
                        -(self.eps2_t[:,None])*iFPM1_field \
                        -(1.-self.eps2_t[:, None])*iFPM2_field)*self.r[None,:]/self.R
    
        lyot_field   = nolyot_field*self.LyotStop1d[None,:]
        
        corono_field_tmp = np.zeros((self.nlam,self.nImg+1), dtype='complex128')
        for i in range(self.nlam):
            corono_field_tmp[i,:] = self.hankel_kernel_all[i,:,:].dot(lyot_field[i,:])
        
        return self.lam0/self.lam_t[:,None]*np.pi*corono_field_tmp*self.R/self.nPup

#%% 
"""
DZPM 1d Coronagraph class     
"""
class HTZPM1d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Dual Zone Phase Mask (DZPM) 
    Coronagraph for one-dimension geometry.
    """    
    default_params = get_default_params_HTZPM1d()    

    def __init__(self, **kwargs):
        r"""
        __init__ : method
            Constructor for the DZPM1d class
        
        Attributes
        ----------      
        OPD1 : float
            Optical path difference :math:`\delta_1` 
            for the inner part of the focal plane mask (FPM)
            
        OPD2 : float
            Optical path difference :math:`\delta_2` 
            for the outer part of the mask
            
        phi1_t : array_like
            Phase shift :math:`\varphi_1` induced by the inner part of the mask
            at all the wavelengths

        phi2_t : array_like
            Phase shift :math:`\varphi_2` induced by the outer part of the mask
            at all the wavelengths
        
        eps1_t : array_like
            Phasor :math:`\varepsilon_1` induced by the inner part of the mask 
            at all the wavelengths

        eps2_t : array_like
            Phasor :math:`\varepsilon_2` induced by the outer part of the mask 
            at all the wavelengths
        
        rMask1_t : array_like
            FPM inner part radius :math:`m_1/2` scaled with 
            wavelength :math:`\lambda/D`
            
        rMask2_t : array_like
            FPM outer part radius :math:`m_2/2` scaled with 
            wavelength :math:`\lambda/D`
            
        nFPM1_t : array_like
            Inner mask sampling at a given wavelength :math:`\lambda/D`
        
        nFPM2_t : array_like
            Outer mask sampling at a given wavelength :math:`\lambda/D`
        
        nFPM1_max : float
            Maximum inner mask sampling over all the wavelengths
            
        nFPM2_max : float
            Maximum outer mask sampling over all the wavelengths
        
        mask1_lam : array_like
            Inner focal plane mask in :math:`\lambda/D` unit
        
        mask2_lam : array_like
            Outer focal plane mask in :math:`\lambda/D` unit
        
        xi_FPM1_lam : array_like
            Inner focal plane mask coordinate in :math:`\lambda/D`
        
        xi_FPM2_lam : array_like
            Outer focal plane mask coordinate in :math:`\lambda/D`       
 
        hankel_kernel_FPM1_all : array_like
            Hankel kernel for the inner FPM at all the wavelengths
        
        hankel_kernel_FPM2_all : array_like
            Hankel kernel for the outer FPM at all the wavelengths
        
        hankel_kernel_iFPM1_all : array_like
            Inverse hankel transform for the inner FPM at all the wavelengths
            
        hankel_kernel_iFPM2_all : array_like
            Inverse hankel transform for the outer FPM at all the wavelengths
            

        References
        ----------
        .. [1] R. Soummer, K. Dohlen, and C. Aime, Achromatic dual-zone phase mask 
            stellar coronagraph, A&A 403, 1 (2003).
            
            https://www.aanda.org/articles/aa/abs/2003/19/aa3246/aa3246.html
            
        .. [2] M. N'Diaye, K. Dohlen, S. Cuevas, R. Soummer, C. Sánchez-Pérez,
            F. Zamkotsian, Improved achromatization of phase mask coronagraphs 
            using colored apodization, A&A 538, A55 (2012). 
            
            https://www.aanda.org/articles/aa/abs/2012/02/aa17661-11/aa17661-11.html
            
        .. [3] J. R. Delorme, M. N'Diaye, R. Galicher, K. Dohlen, P. Baudoz, 
            A. Caillat, G. Rousset, R. Soummer, O. Dupuis, Laboratory validation of
            the dual-zone phase mask coronagraph in broadband light at the 
            high-contrast imaging THD testbed, A&A 592, A119 (2016). 
            
            https://www.aanda.org/articles/aa/abs/2016/08/aa28587-16/aa28587-16.html        


        """
        super(HTZPM1d,self).__init__(**kwargs)
        
        # Optical path difference introduced by the mask
        self.OPD2 = self.OPDx2*self.lam0
        self.OPD3 = self.OPDx3*self.lam0
        
        # phase shift induced by the mask
        self.phi2_t = 2.*np.pi*self.OPD2/self.lam_t
        self.phi3_t = 2.*np.pi*self.OPD3/self.lam_t
        
        self.eps2_t   = 1j*np.sin(self.phi2_t) + np.cos(self.phi2_t)
        self.eps3_t   = 1j*np.sin(self.phi3_t) + np.cos(self.phi3_t)


        # mask size at a given wavelength
        self.rMask1_t = (self.lam0/self.lam_t)*self.rMask1
        self.rMask2_t = (self.lam0/self.lam_t)*self.rMask2
        self.rMask3_t = (self.lam0/self.lam_t)*self.rMask3
        
        # mask sampling at a given wavelength and max nFPM_max 
        self.nFPM1_t   = self.rMask1_t*self.nFPM
        self.nFPM1_max = int(np.max(self.nFPM1_t))     
        self.nFPM2_t   = self.rMask2_t*self.nFPM
        self.nFPM2_max = int(np.max(self.nFPM2_t))
        self.nFPM3_t   = self.rMask3_t*self.nFPM
        self.nFPM3_max = int(np.max(self.nFPM3_t))

        self.mask1_lam   = (np.arange(self.nFPM1_max+1)[None,:]\
                            <=self.rMask1_t[:,None]*self.nFPM)
        self.xi_FPM1_lam = np.arange(self.nFPM1_max+1)[None,:]\
        *self.mask1_lam/self.nFPM       
        self.mask2_lam   = (np.arange(self.nFPM2_max+1)[None,:]\
                            <=self.rMask2_t[:,None]*self.nFPM)
        self.xi_FPM2_lam = np.arange(self.nFPM2_max+1)[None,:]\
        *self.mask2_lam/self.nFPM
        self.mask3_lam   = (np.arange(self.nFPM3_max+1)[None,:]\
                            <=self.rMask3_t[:,None]*self.nFPM)
        self.xi_FPM3_lam = np.arange(self.nFPM3_max+1)[None,:]\
        *self.mask3_lam/self.nFPM


        self.hankel_kernel_FPM1_all  = besselJ0(
                np.pi/self.R*self.xi_FPM1_lam[:,:,None]*self.r[None,None,:])
        self.hankel_kernel_iFPM1_all = besselJ0(
                np.pi/self.R*self.xi_FPM1_lam[:,None,:]*self.r[None,:,None])
        self.hankel_kernel_FPM2_all  = besselJ0(
                np.pi/self.R*self.xi_FPM2_lam[:,:,None]*self.r[None,None,:])
        self.hankel_kernel_iFPM2_all = besselJ0(
                np.pi/self.R*self.xi_FPM2_lam[:,None,:]*self.r[None,:,None])
        self.hankel_kernel_FPM3_all  = besselJ0(
                np.pi/self.R*self.xi_FPM3_lam[:,:,None]*self.r[None,None,:])
        self.hankel_kernel_iFPM3_all = besselJ0(
                np.pi/self.R*self.xi_FPM3_lam[:,None,:]*self.r[None,:,None])



#%% direct propagation (no focal plane mask)
    def compute_direct_field_1d(self,Apod):
        r"""
        Computes the electric field of the direct image with DZPM
        for 1D problem.
        
        Parameters
        ---------- 
        Apod: array_like
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Direct electric field :math:`\Phi_0` at all the wavelengths
            
        """
        field = np.zeros((self.nlam,self.nImg+1), dtype='complex128')
        for i in range(self.nlam):
            field[i,:] = \
            self.lam0/self.lam_t[i]*np.pi*self.hankel_kernel_all[i].dot(
                    self.Pupil1d*Apod*self.LyotStop1d*self.r/self.R)\
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
            Entrance pupil apodization :math:`\Phi`
                
        Returns    
        ----------
        res : array_like
            Coronagraphic electric field :math:`\Phi_D` at all the wavelengths
            
        """
        FPM1_field = np.zeros((self.nlam, self.nFPM1_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM1_field[i,:] = np.pi*self.hankel_kernel_FPM1_all[i,:,:].dot(
                    Apod*self.Pupil1d*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM1_lam[i,None]

        iFPM1_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM1_field[i,:]= np.pi*self.hankel_kernel_iFPM1_all[i,:,:].dot(
                    FPM1_field[i,:])*(1/self.nFPM)
    
        FPM2_field = np.zeros((self.nlam, self.nFPM2_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM2_field[i,:] = np.pi*self.hankel_kernel_FPM2_all[i,:,:].dot(
                    Apod*self.Pupil1d*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM2_lam[i,None]

        iFPM2_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM2_field[i,:]=np.pi*self.hankel_kernel_iFPM2_all[i,:,:].dot(
                    FPM2_field[i,:])*(1/self.nFPM)

        FPM3_field = np.zeros((self.nlam, self.nFPM3_max+1), dtype='complex128')
        for i in range(self.nlam):
            FPM3_field[i,:] = np.pi*self.hankel_kernel_FPM3_all[i,:,:].dot(
                    Apod*self.Pupil1d*self.r/self.R)\
                    *(self.R/self.nPup)*self.xi_FPM3_lam[i,None]

        iFPM3_field = np.zeros((self.nlam,self.nPup), dtype='complex128')
        for i in range(self.nlam):
            iFPM3_field[i,:]=np.pi*self.hankel_kernel_iFPM3_all[i,:,:].dot(
                    FPM3_field[i,:])*(1/self.nFPM)
             
        nolyot_field = (Apod[None,:]*self.Pupil1d[None,:]\
                        -(self.eps2_t[:,None])*iFPM1_field \
                        -(self.eps3_t[:, None]-self.eps2_t[:, None])*iFPM2_field \
                        -(1.-self.eps3_t[:, None])*iFPM3_field )*self.r[None,:]/self.R
    
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
        r"""
        __init__ : method
            Constructor for the APLC2d class.
        
        Attributes
        ----------        
        mB_t : array_like
            Spatial frequency range within the focal plane mask (FPM) 
            in plane B in :math:`\lambda/D` at all the wavelengths
           
        mD_t : array_like
            Spatial frequency range in the final image plane D 
            in :math:`\lambda/D` at all the wavelengths
                
        """      
        super(APLC2d,self).__init__(**kwargs)
        
        # mask size at a given wavelength for SFT
        self.mB_t  = 2.*self.rMask*(self.lam0/self.lam_t)
        self.mD_t  = self.Fmax2d*(self.lam0/self.lam_t)
        
#%% direct propagation (no focal plane mask)
    def compute_direct_field_2d(self,Apod2d):
        r""" 
        Computes the coronagraph electric field for a classical Lyot coronagraph
        with four planes (A: entrance pupil, B: intermediate focal plane, 
        C: relayed pupil before stop, L: relayed pupil after stop, 
        D: final image plane).
        Resolution element are given in :math:`\lambda_0/D` where 
        :math:`\lambda_0` and :math:`D` denote the central and the telescope 
        diameter.
    
        Parameters
        ----------     
        Apod2d : array_like 
            Entrance pupil apodization :math:`\Phi`
                        
        Returns    
        ----------    
        field_Dtmp : array_like
            Direct electric field :math:`\Psi_0` in the final image plane at 
            all the wavelengths
    
        """

        field_A    = Apod2d*self.Pupil2d
        if self.OPDmap2d is not None:
            phasor_t  = 2.*np.pi*self.OPDmap2d[None, :, :]/(self.wv*self.lam_t[:, None,None])             
            field_A   = Apod2d*self.Pupil2d*(1j*np.sin(phasor_t)+np.cos(phasor_t))

        if self.Ampmap2d is not None:
            field_A   *= self.Ampmap2d

        field_Dtmp = np.zeros((self.nlam,self.nImg2d,self.nImg2d), 
                                  dtype='complex128')
 
        field_L    = field_A*self.LyotStop2d
        for i in range(self.nlam):
            if self.OPDmap2d is not None:
                field_L   = field_A[i]*self.LyotStop2d                
            if self.Pupil2dSym == False:
                field_Dtmp[i] = sft(field_L, self.nImg2d, self.mD_t[i], 
                          CtrBtwnPix=self.CtrBtwnPix2)
            else:
                field_Dtmp[i] = sft_even(field_L, self.nImg2d, self.mD_t[i], 
                          CtrBtwnPix=self.CtrBtwnPix2) 
                                
        return field_Dtmp
 
#%%
    def compute_corono_field_2d(self,Apod2d):
        """
        Computes the coronagraph electric field for a classical Lyot coronagraph
        with four planes (A: entrance pupil, B: intermediate focal plane, 
        C: relayed pupil before stop, L: relayed pupil after stop, 
        D: final image plane).
        Resolution element are given in :math:`\lambda_0/D` where 
        :math:`\lambda_0` and :math:`D` denote the central and the telescope 
        diameter.
    
        Parameters
        ---------- 
        Apod2d : array_like 
            Entrance pupil apodization :math:`\Phi`
            
        Returns    
        ----------    
        field_Dtmp : array_like
            Coronagraphic electric field :math:`\Psi_D` in the final image plane
            at all the wavelengths
            
        """        

        field_A    = Apod2d*self.Pupil2d
        if self.OPDmap2d is not None:
            phasor_t   = 2.*np.pi*self.OPDmap2d[None, :, :]/(self.wv*self.lam_t[:, None,None])             
            field_A    = Apod2d*self.Pupil2d*(1j*np.sin(phasor_t)+np.cos(phasor_t))

        if self.Ampmap2d is not None:
            field_A *= self.Ampmap2d
            
        field_Dtmp = np.zeros((self.nlam,self.nImg2d,self.nImg2d), 
                                  dtype='complex128')

        for i in range(self.nlam):
            if self.OPDmap2d is None:
                field = field_A
            else:
                field = field_A[i]
            if self.Pupil2dSym == False:
                field_B       = self.mask2d*sft(field, self.nFPM, self.mB_t[i], 
                                                CtrBtwnPix=self.CtrBtwnPix)
                field_C       = field - isft(field_B, self.nPup, self.mB_t[i], 
                                               CtrBtwnPix=self.CtrBtwnPix)
                field_L       = field_C*self.LyotStop2d
                field_Dtmp[i] = sft(field_L, self.nImg2d, self.mD_t[i], 
                          CtrBtwnPix=self.CtrBtwnPix2)
            else:
                field_B       = self.mask2d*sft_even(field, self.nFPM, self.mB_t[i], 
                                                CtrBtwnPix=self.CtrBtwnPix)
                field_C       = field - isft_even(field_B, self.nPup, self.mB_t[i], 
                                               CtrBtwnPix=self.CtrBtwnPix)
                field_L       = field_C*self.LyotStop2d
                field_Dtmp[i] = sft_even(field_L, self.nImg2d, self.mD_t[i], 
                          CtrBtwnPix=self.CtrBtwnPix2)
                                   
        return field_Dtmp   

#%% direct propagation (no focal plane mask)
    def compute_direct_lyot_field_2d(self,Apod2d):
        r""" 
        Computes the coronagraph electric field for a classical Lyot coronagraph
        in the Lyot plane (A: entrance pupil, B: intermediate focal plane, 
        C: relayed pupil before stop, L: relayed pupil after stop, 
        D: final image plane).
        Resolution element are given in :math:`\lambda_0/D` where 
        :math:`\lambda_0` and :math:`D` denote the central and the telescope 
        diameter.
    
        Parameters
        ----------     
        Apod2d : array_like 
            Entrance pupil apodization :math:`\Phi`
            
        Returns    
        ----------    
        field_L : array_like
            Direct electric field :math:`\Psi_0` in the Lyot plane before stop 
            at all the wavelengths if OPDmap is not None
    
        """

        field_A    = Apod2d*self.Pupil2d
        if self.OPDmap2d is not None:
            phasor_t   = 2.*np.pi*self.OPDmap2d[None, :, :]/(self.wv*self.lam_t[:, None,None])             
            field_A    = Apod2d*self.Pupil2d*(1j*np.sin(phasor_t)+np.cos(phasor_t))

        if self.Ampmap2d is not None:
            field_A   *= self.Ampmap2d
            
        return field_A

#%%
    def compute_corono_lyot_field_2d(self,Apod2d):
        """
        Computes the coronagraph electric field for a classical Lyot coronagraph
        with four planes (A: entrance pupil, B: intermediate focal plane, 
        C: relayed pupil before stop, L: relayed pupil after stop, 
        D: final image plane).
        Resolution element are given in :math:`\lambda_0/D` where 
        :math:`\lambda_0` and :math:`D` denote the central and the telescope 
        diameter.
    
        Parameters
        ---------- 
        Apod2d : array_like 
            Entrance pupil apodization :math:`\Phi`
            
        Returns    
        ----------    
        field_L : array_like
            Direct electric field :math:`\Psi_0` in the Lyot plane at 
            all the wavelengths if OPDmap is not None
            
        """        

        field_A    = Apod2d*self.Pupil2d
        if self.OPDmap2d is not None:
            phasor_t   = 2.*np.pi*self.OPDmap2d[None, :, :]/(self.wv*self.lam_t[:, None,None])             
            field_A    = Apod2d*self.Pupil2d*(1j*np.sin(phasor_t)+np.cos(phasor_t))

        if self.Ampmap2d is not None:
            field_A *= self.Ampmap2d            

        field_C    = np.zeros((self.nlam,self.nPup,self.nPup), 
                                  dtype='complex128')
        for i in range(self.nlam):
            if self.OPDmap2d is None:
                field = field_A
            else:
                field = field_A[i]                
            if self.Pupil2dSym == False:
                field_B       = self.mask2d*sft(field, self.nFPM, self.mB_t[i], 
                                                CtrBtwnPix=self.CtrBtwnPix)
                field_C[i]       = field - isft(field_B, self.nPup, self.mB_t[i], 
                                               CtrBtwnPix=self.CtrBtwnPix)

            else:
                field_B       = self.mask2d*sft_even(field, self.nFPM, self.mB_t[i], 
                                                CtrBtwnPix=self.CtrBtwnPix)
                field_C[i]       = field - isft_even(field_B, self.nPup, self.mB_t[i], 
                                               CtrBtwnPix=self.CtrBtwnPix)
                                   
        return field_C   

#%% 
"""
SP 2d Coronagraph subclass        
"""
class SP2d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Apodized Pupil Lyot Coronagraph
    for two-dimension geometry.
    """
    default_params = get_default_params_SP2d()

    def __init__(self, **kwargs):
        r"""
        __init__ : method
            Constructor for the APLC2d class.
        
        Attributes
        ----------        
        mB_t : array_like
            Spatial frequency range within the focal plane mask (FPM) 
            in plane B in :math:`\lambda/D` at all the wavelengths
           
        mD_t : array_like
            Spatial frequency range in the final image plane D 
            in :math:`\lambda/D` at all the wavelengths
                
        """      
        super(SP2d,self).__init__(**kwargs)
        
        # mask size at a given wavelength for SFT
        self.mD_t  = self.Fmax2d*(self.lam0/self.lam_t)
        
#%% direct propagation (no focal plane mask)
    def compute_direct_field_2d(self,Apod2d):
        r""" 
        Computes the coronagraph electric field for a classical Lyot coronagraph
        with four planes (A: entrance pupil, D: final image plane).
        Resolution element are given in :math:`\lambda_0/D` where 
        :math:`\lambda_0` and :math:`D` denote the central and the telescope 
        diameter.
    
        Parameters
        ----------     
        Apod2d : array_like 
            Entrance pupil apodization :math:`\Phi`
            
        Returns    
        ----------    
        field_Dtmp : array_like
            Direct electric field :math:`\Psi_0` in the final image plane at 
            all the wavelengths
    
        """    
        field_A    = Apod2d*self.Pupil2d
        field_Dtmp = np.zeros((self.nlam,self.nImg2d,self.nImg2d), 
                              dtype='complex128')

        if self.Pupil2dSym == False:
            for i in range(self.nlam):
                field_Dtmp[i] = sft(field_A, self.nImg2d, self.mD_t[i], 
                      CtrBtwnPix=self.CtrBtwnPix2)
        else:            
            for i in range(self.nlam):
                field_Dtmp[i] = sft_even(field_A, self.nImg2d, self.mD_t[i], 
                      CtrBtwnPix=self.CtrBtwnPix2)    
                            
        return field_Dtmp
 
#%%
    def compute_corono_field_2d(self,Apod2d):
        """
        Computes the coronagraph electric field for a classical Lyot coronagraph
        with two planes (A: entrance pupil, D: final image plane).
        Resolution element are given in :math:`\lambda_0/D` where 
        :math:`\lambda_0` and :math:`D` denote the central and the telescope 
        diameter.
    
        Parameters
        ---------- 
        Apod2d : array_like 
            Entrance pupil apodization :math:`\Phi`
            
        Returns    
        ----------    
        field_Dtmp : array_like
            Coronagraphic electric field :math:`\Psi_D` in the final image plane
            at all the wavelengths
            
        """        
        field_A    = Apod2d*self.Pupil2d
        field_Dtmp = np.zeros((self.nlam,self.nImg2d,self.nImg2d), 
                              dtype='complex128')

        if self.Pupil2dSym == False:
            for i in range(self.nlam):
                field_Dtmp[i] = sft(field_A, self.nImg2d, self.mD_t[i], 
                      CtrBtwnPix=self.CtrBtwnPix2)
        else:            
            for i in range(self.nlam):
                field_Dtmp[i] = sft_even(field_A, self.nImg2d, self.mD_t[i], 
                      CtrBtwnPix=self.CtrBtwnPix2)    
                             
        return field_Dtmp   

#%% 
"""
APLC 2d Coronagraph subclass        
"""
class DZPM2d(Coronagraph):
    """
    Defines the Coronagraph subclass for the Dual-Zone Phase Mask Coronagraph
    for two-dimension geometry.
    """
    default_params = get_default_params_DZPM2d()

    def __init__(self, **kwargs):
        r"""
        __init__ : method
            Constructor for the APLC2d class.
        
        Attributes
        ----------        
        mB_t : array_like
            Spatial frequency range within the focal plane mask (FPM) 
            in plane B in :math:`\lambda/D` at all the wavelengths
           
        mD_t : array_like
            Spatial frequency range in the final image plane D 
            in :math:`\lambda/D` at all the wavelengths
                
        """      
        super(DZPM2d,self).__init__(**kwargs)

        # Optical path difference introduced by the mask
        self.OPD1 = self.OPDx1*self.lam0
        self.OPD2 = self.OPDx2*self.lam0
        
        # phase shift induced by the mask
        self.phi1_t = 2.*np.pi*self.OPD1/self.lam_t
        self.phi2_t = 2.*np.pi*self.OPD2/self.lam_t
        
        self.eps1_t = 1j*np.sin(self.phi1_t) + np.cos(self.phi1_t)
        self.eps2_t = 1j*np.sin(self.phi2_t) + np.cos(self.phi2_t)
        
        # mask size at a given wavelength for SFT
        self.mB1_t  = 2.*self.rMask1*(self.lam0/self.lam_t)
        self.mB2_t  = 2.*self.rMask2*(self.lam0/self.lam_t)
        self.mD_t   = self.Fmax2d*(self.lam0/self.lam_t)
        
        self.rr = radius_disk(self.nPup, self.nPup/2, CtrBtwnPix=self.CtrBtwnPix)

        self.Apod2d_w = 1j*np.sin(2.*np.pi*(self.rr)**2*self.beta*self.lam0/self.lam_t[:,None,None])\
        + np.cos(2.*np.pi*(self.rr)**2*self.beta*self.lam0/self.lam_t[:,None,None])

        
#%% direct propagation (no focal plane mask)
    def compute_direct_field_2d(self,Apod2d):
        r""" 
        Computes the coronagraph electric field for a classical Lyot coronagraph
        with four planes (A: entrance pupil, B: intermediate focal plane, 
        C: relayed pupil before stop, L: relayed pupil after stop, 
        D: final image plane).
        Resolution element are given in :math:`\lambda_0/D` where 
        :math:`\lambda_0` and :math:`D` denote the central and the telescope 
        diameter.
    
        Parameters
        ----------     
        Apod2d : array_like 
            Entrance pupil apodization :math:`\Phi`
            
        Returns    
        ----------    
        field_Dtmp : array_like
            Direct electric field :math:`\Psi_0` in the final image plane at 
            all the wavelengths
    
        """    
 
        field_Dtmp = np.zeros((self.nlam,self.nImg2d,self.nImg2d), 
                              dtype='complex128')
        
        if self.Pupil2dSym == False:
            for i in range(self.nlam):
                field_A    = Apod2d*self.Pupil2d*self.Apod2d_w[i]
                field_L    = field_A*self.LyotStop2d
                field_Dtmp[i] = sft(field_L, self.nImg2d, self.mD_t[i], 
                          CtrBtwnPix=self.CtrBtwnPix2)
        else:
            for i in range(self.nlam):
                field_A    = Apod2d*self.Pupil2d*self.Apod2d_w[i]
                field_L    = field_A*self.LyotStop2d
                field_Dtmp[i] = sft_even(field_L, self.nImg2d, self.mD_t[i], 
                          CtrBtwnPix=self.CtrBtwnPix2)            
    
        return field_Dtmp
 
#%%
    def compute_corono_field_2d(self,Apod2d):
        """
        Computes the coronagraph electric field for a classical Lyot coronagraph
        with four planes (A: entrance pupil, B: intermediate focal plane, 
        C: relayed pupil before stop, L: relayed pupil after stop, 
        D: final image plane).
        Resolution element are given in :math:`\lambda_0/D` where 
        :math:`\lambda_0` and :math:`D` denote the central and the telescope 
        diameter.
    
        Parameters
        ---------- 
        Apod2d : array_like 
            Entrance pupil apodization :math:`\Phi`
            
        Returns    
        ----------    
        field_Dtmp : array_like
            Coronagraphic electric field :math:`\Psi_D` in the final image plane
            at all the wavelengths
            
        """        
   
        field_Dtmp = np.zeros((self.nlam,self.nImg2d,self.nImg2d), 
                              dtype='complex128')
        
        if self.Pupil2dSym == False: 
            for i in range(self.nlam):
                field_A       = Apod2d*self.Pupil2d*self.Apod2d_w[i] 
                field_B1      = self.mask2d*sft(field_A, self.nFPM, self.mB1_t[i], 
                                                CtrBtwnPix=self.CtrBtwnPix)
                field_B2      = self.mask2d*sft(field_A, self.nFPM, self.mB2_t[i], 
                                                CtrBtwnPix=self.CtrBtwnPix)
                
                field_C       = field_A - (self.eps2_t[i] - self.eps1_t[i])\
                                *isft(field_B1, self.nPup, self.mB1_t[i], 
                                               CtrBtwnPix=self.CtrBtwnPix)\
                                        - (1.-self.eps2_t[i])\
                                *isft(field_B2, self.nPup, self.mB2_t[i], 
                                               CtrBtwnPix=self.CtrBtwnPix)
                                
                field_L       = field_C*self.LyotStop2d
                field_Dtmp[i] = sft(field_L, self.nImg2d, self.mD_t[i], 
                          CtrBtwnPix=self.CtrBtwnPix2)
        else:
            for i in range(self.nlam):
                field_A       = Apod2d*self.Pupil2d*self.Apod2d_w[i]
                field_B1      = self.mask2d*sft_even(field_A, self.nFPM, self.mB1_t[i], 
                                                CtrBtwnPix=self.CtrBtwnPix)
                field_B2      = self.mask2d*sft_even(field_A, self.nFPM, self.mB2_t[i], 
                                                CtrBtwnPix=self.CtrBtwnPix)
                
                field_C       = field_A - (self.eps2_t[i] - self.eps1_t[i])\
                                *isft_even(field_B1, self.nPup, self.mB1_t[i], 
                                               CtrBtwnPix=self.CtrBtwnPix)\
                                        - (1.-self.eps2_t[i])\
                                *isft_even(field_B2, self.nPup, self.mB2_t[i], 
                                               CtrBtwnPix=self.CtrBtwnPix)
                                           
                field_L       = field_C*self.LyotStop2d
                field_Dtmp[i] = sft_even(field_L, self.nImg2d, self.mD_t[i], 
                          CtrBtwnPix=self.CtrBtwnPix2)            

        return field_Dtmp   
     