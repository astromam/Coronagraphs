#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:49:07 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

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
    PupilID : float (default=0.14)
        Pupil central obstruction size :math:`d` in fraction of the pupil 
        diameter :math:`D`
        
    LyotStopID : float (default=0.28)
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
    tmp = {'PupilID':0.14,'LyotStopID':0.28,'LyotStopOD':1.0,    
           'rho0':9,'rho1':10,
           'rho0direct':0.7, 'rho1direct':6.0,
           'nPup':50,'nImg':50,'Fmax':25,'nFPM':25,
           'nImg2d':44, 'Fmax2d':22,
           'bw':0.2,'lam0':1.0,'nlam':1, 'wv':1.593e-6,
           'R':1.0,
           'fdir':'',
           'CtrBtwnPix':True, 'CtrBtwnPix2':False, 
           'Pupil2dSym':True, 
           'Pupil1d':None, 'LyotStop1d':None,
           'Pupil2d':None, 'LyotStop2d':None,
           'ImPart':True, 'lam_t':None}
            
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
    tmp.update({'rMask':2.8,'nFPM':50, 'corono_name':'APLC',
                'Pupil1d':None, 'LyotStop1d':None})

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
    tmp.update({'rMask':5.0,'nFPM':50, 'corono_name':'SP',
                'Pupil1d':None, 'LyotStop1d':None})
    
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
           'corono_name':'DZPM',
           'Pupil1d':None, 'LyotStop1d':None})

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
           'corono_name':'HDZPM',
           'Pupil1d':None, 'LyotStop1d':None})

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
           'corono_name':'HTZPM',
           'Pupil1d':None, 'LyotStop1d':None})

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
                'nPup':30, 'nFPM':25,
                'corono_name':'APLC',
                'Pupil2d':None, 'LyotStop2d':None,
                'OPDmap2d':None, 'Ampmap2d':None})
                
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
                'nPup':30, 'nFPM':25, 
                'corono_name':'SP',
                'Pupil2d':None, 'LyotStop2d':None,
                'OPDmap2d':None, 'Ampmap2d':None
                })
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
           'nFPM':25, 'nPup':30,
           'corono_name':'DZPM',
           'Pupil2d':None, 'LyotStop2d':None,
           'OPDmap2d':None, 'Ampmap2d':None})
        
    return tmp

#%%
def get_default_params_1d_ProblemMatrix():
    r"""
    Gets the default parameters for the optimization problem matrix.
       
    Parameters
    ----------
    cDarkHole : float (default=8)
        Contrast goal :math:`C` in log scale inside the search area in the 
        coronagraphic image.
    
    tau : float (default=0.2)
        Integrated amplitude transmission :math:`\tau` of the apodizer 
        :math:`\Phi` in fraction of the integrated amplitude transmission of 
        the pupil :math:`P_0`.
    
    solver : string (default='stdgrb')
        solver for the programming problem (linear for the moment). 
        The user can choose between:
            
            - 'gurobipy'     : python implementation of the gurobi solver.
            Code from gurobi: http://www.gurobi.com/documentation/
        
            - 'stdgrb'       : cython wrapper that calls gurobi through its C 
            interface.        
            Code by R. Flamary: https://github.com/rflamary/stdgrb
        
            - 'scipy.optimize.linprog': linear programming solver from scipy 
            package.
            Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    
    slvCrossover : integer (default=0)
        gurobi solver parameter for barrier crossover strategy. 
        See details: http://www.gurobi.com/documentation/8.0/refman/crossover.html
        
    slvLogToConsole : integer (default=0)
        gurobi solver parameter for control console logging.
        See details: http://www.gurobi.com/documentation/8.0/refman/logtoconsole.html 
    
    slvMethod : integer (default=2)
        gurobi solver parameter to select the used algorithm to solve problem.
        See details: http://www.gurobi.com/documentation/8.0/refman/method.html
        
        The user can choose between:
    
        - -1 : Automatic
        
        -  1 : Dual simplex method
        
        -  2 : Barrier

    slvSparse : integer (default=0)
        stdgrb parameter to transform A into a sparse matrix and give it to 
        the gurobi solver.
        
    allLogToConsole : integer (default=0)
        control console logging for output from this class
        
    FirstDer : bool (default=False)
        introduce constraints on the first derivative of the apodizer 
        transmission in the optimization problem.
    
    FirstDerLim : float (default=0.01)
        Upper limit on the absolute first derivative of the apodizer 
        transmission.   
    
    SecondDer : bool (default=False)
        introduce constraints on the second derivative of the apodizer 
        transmission in the optimization problem.
        
    SecondDerLim : float (default=0.0001)
        Upper limit on the absolute second derivative of the apodizer 
        transmission.
        
    MinIsland : bool (default=False)
        introduce constraints on the first derivative of the apodizer 
        transmission in the optimization problem to minimize the number of 
        islands in the apodization.
        
    FirstDerGlobalLim : float (default=0.01)
        Upper limit on the integral of the absolute first derivative of the 
        apodizer transmission.
            
    Returns    
    ----------
    tmp : dict
        Dictionary of parameters with their default values.
        
    References
    ----------        
    .. [1] Gurobi Optimization, LLC, Gurobi Optimizer Reference Manual (2018).
    
           http://www.gurobi.com
        
    """
    tmp = {'cDarkHole':6, 'tau':0.1, 'solver':'stdgrb', 
           'problem_name':'MaxTau',
           'slvCrossover':0, 'slvLogToConsole':0, 'slvMethod':2, 'slvSparse':0,
           'allLogToConsole':0, 
           'FirstDer':False, 'FirstDerLim':0.01, 
           'SecondDer': False, 'SecondDerLim':0.0001,
           'MinIsland':False, 'FirstDerGlobalLim':0.01}
    
    return tmp

#%%
def get_default_params_1d_MaxTauProblemMatrix():
    r"""
    Gets the default parameters for the Max contrast optimization problem.
    
    Parameters
    ---------- 
    tmp : dict
        Dictionary from the get_default_matrix_pb
        
    Lnorm : string (default= 'L1')
        L-norm type for the optimization problem 
        ('Linf' : :math:`L_{\infty}` norm, 'L1' : :math:`L_1`-norm)
    
    problem_name : string (default='MaxTau')
        name of the optimization problem.
        The user can choose between:
            
            - 'MaxTau': maximization of the apodizer transmission for a given 
            contrast
            
            - 'MaxContrastL1': maximization of the contrast for a given 
            apodizer transmission under L1-norm constraints
            
            - 'MaxContrastLinf': maximization of the contrast for a given 
            apodizer transmission under Linf-norm constraints
                        
    Returns    
    ----------
    tmp : dict
        Updated dictionary
        
    """    
    tmp = get_default_params_1d_ProblemMatrix()
    tmp.update({'problem_name':'MaxTau'})
    
    return tmp

#%%
def get_default_params_1d_MaxContrastProblemMatrix():
    r"""
    Gets the default parameters for the Max contrast optimization problem.
    
    Parameters
    ---------- 
    tmp : dict
        Dictionary from the get_default_matrix_pb
        
    Lnorm : string (default= 'L1')
        L-norm type for the optimization problem 
        ('Linf' : :math:`L_{\infty}` norm, 'L1' : :math:`L_1`-norm)

    problem_name : string (default='MaxTau')
        name of the optimization problem.
        The user can choose between:
            
            - 'MaxTau': maximization of the apodizer transmission for a given 
            contrast
            
            - 'MaxContrastL1': maximization of the contrast for a given 
            apodizer transmission under L1-norm constraints
            
            - 'MaxContrastLinf': maximization of the contrast for a given 
            apodizer transmission under Linf-norm constraints
                            
    Returns    
    ----------
    tmp : dict
        Updated dictionary
        
    """    
    tmp = get_default_params_1d_ProblemMatrix()
    tmp.update({'Lnorm':'L1', 'problem_name':'MaxContrastL1'})
    
    return tmp

#%%
def get_default_params_1d_MaxSNR():
    r"""
    Gets the default parameters for the MaxSNR optimization problem.
    
    Parameters
    ---------- 
    tmp : dict
        Dictionary from the get_default_matrix_pb
        
    initialisation : string (default= 'L2')
        Type of initialization for the Frank-Wolfe algorithme.
        The user can choose between :
            
            - 'Unif' which initializes the algorithm with a uniform apodizer
            
            - 'Random' which initializes the algorithm with a random apodizer
            
            - L-norm type for the optimization problem that initialize the Frank-Wolfe
        algorithm
        ('Linf' : :math:`L_{\infty}` norm, 'L1' : :math:`L_1`-norm)

    problem_name : string (default='MaxSNR')
        name of the optimization problem.                
            - 'MaxSNR': maximization of the SNR for a given 
            apodizer transmission through the Frank-Wolfe algorithm
            
    nmax  : Int (default=10000)
        Maximum number of iterations for the Frank-Wolfe algorithm
        
    gradmin :  Float (default = 1e-7)
        Minimal gradient value of the cost function for which the Frank-Wolfe algorithm
        consider it has converged
    
    Returns    
    ----------
    tmp : dict
        Updated dictionary
        
    """    
    tmp = get_default_params_1d_ProblemMatrix()
    tmp.update({ 'problem_name':'MaxSNR','nmax' :10000,'gradmin':1e-7, 'initialisation':'Unif'})
    
    return tmp

#%%
def get_default_params_2d_ProblemMatrix():
    r"""
    Gets the default parameters for the optimization problem matrix.
       
    Parameters
    ----------
    cDarkHole : float (default=8)
        Contrast goal :math:`C` in log scale inside the search area in the 
        coronagraphic image.
    
    tau : float (default=0.2)
        Integrated amplitude transmission :math:`\tau` of the apodizer 
        :math:`\Phi` in fraction of the integrated amplitude transmission of 
        the pupil :math:`P_0`.
    
    solver : string (default='stdgrb')
        solver for the programming problem (linear for the moment). 
        The user can choose between:
            
            - 'gurobipy'     : python implementation of the gurobi solver.
            Code from gurobi: http://www.gurobi.com/documentation/
        
            - 'stdgrb'       : cython wrapper that calls gurobi through its C 
            interface.        
            Code by R. Flamary: https://github.com/rflamary/stdgrb
        
            - 'scipy.optimize.linprog': linear programming solver from scipy 
            package.
            Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    
    pupil_name : string (default='sbr')
        name of the pupil.
    
    problem_name : string (default='MaxTau')
        name of the optimization problem.
        The user can choose between:
            
            - 'MaxTau': maximization of the apodizer transmission for a given 
            contrast
            
            - 'MaxContrastL1': maximization of the contrast for a given 
            apodizer transmission under L1-norm constraints
            
            - 'MaxContrastLinf': maximization of the contrast for a given 
            apodizer transmission under Linf-norm constraints
    
    slvCrossover : integer (default=0)
        gurobi solver parameter for barrier crossover strategy. 
        See details: http://www.gurobi.com/documentation/8.0/refman/crossover.html
        
    slvLogToConsole : integer (default=0)
        gurobi solver parameter for control console logging.
        See details: http://www.gurobi.com/documentation/8.0/refman/logtoconsole.html 
    
    slvMethod : integer (default=2)
        gurobi solver parameter to select the used algorithm to solve problem.
        See details: http://www.gurobi.com/documentation/8.0/refman/method.html
        
        The user can choose between:
    
        - -1 : Automatic
        
        -  1 : Dual simplex method
        
        -  2 : Barrier
        
    allLogToConsole : integer (default=0)
        control console logging for output from this class
        
    MinIsland : bool (default=False)
        introduce constraints on the first derivative of the apodizer 
        transmission in the optimization problem to minimize the number of 
        islands in the apodization.
        
    FirstDerGlobalLim : float (default=0.01)
        Upper limit on the integral of the absolute first derivative of the 
        apodizer transmission.
            
    Returns    
    ----------
    tmp : dict
        Dictionary of parameters with their default values.
        
    References
    ----------        
    .. [1] Gurobi Optimization, LLC, Gurobi Optimizer Reference Manual (2018).
    
           http://www.gurobi.com
        
    """
    tmp = {'cDarkHole':8, 'tau':0.8, 'solver':'stdgrb', 
           'pupil_name':'sbr', 
           'problem_name':'MaxTau',
           'slvCrossover':0, 'slvLogToConsole':1, 'slvMethod':2,
           'allLogToConsole':0,
           'MinIsland':False, 'FirstDerGlobalLim':0.01,
           'ImPart':True, 'LSRobustness':False}
    return tmp

#%%
def get_default_params_2d_MaxTauProblemMatrix():
    r"""
    Gets the default parameters for the Max contrast optimization problem.
    
    Parameters
    ---------- 
    tmp : dict
        Dictionary from the get_default_matrix_pb
        
    Lnorm : string (default= 'L1')
        L-norm type for the optimization problem 
        ('Linf' : :math:`L_{\infty}` norm, 'L1' : :math:`L_1`-norm)
    
    problem_name : string (default='MaxTau')
        name of the optimization problem.
        The user can choose between:
            
            - 'MaxTau': maximization of the apodizer transmission for a given 
            contrast
            
            - 'MaxContrastL1': maximization of the contrast for a given 
            apodizer transmission under L1-norm constraints
            
            - 'MaxContrastLinf': maximization of the contrast for a given 
            apodizer transmission under Linf-norm constraints
        
    Returns    
    ----------
    tmp : dict
        Updated dictionary
        
    """    
    tmp = get_default_params_2d_ProblemMatrix()
    tmp.update({'problem_name':'MaxTau'})
    
    return tmp

#%%
def get_default_params_2d_MaxContrastProblemMatrix():
    r"""
    Gets the default parameters for the Max contrast optimization problem.
    
    Parameters
    ---------- 
    tmp : dict
        Dictionary from the get_default_matrix_pb
        
    Lnorm : string (default= 'L1')
        L-norm type for the optimization problem 
        ('Linf' : :math:`L_{\infty}` norm, 'L1' : :math:`L_1`-norm)

    problem_name : string (default='MaxTau')
        name of the optimization problem.
        The user can choose between:
            
            - 'MaxTau': maximization of the apodizer transmission for a given 
            contrast
            
            - 'MaxContrastL1': maximization of the contrast for a given 
            apodizer transmission under L1-norm constraints
            
            - 'MaxContrastLinf': maximization of the contrast for a given 
            apodizer transmission under Linf-norm constraints
            
    Returns    
    ----------
    tmp : dict
        Updated dictionary
        
    """    
    tmp = get_default_params_2d_ProblemMatrix()
    tmp.update({'Lnorm':'L1', 'problem_name':'MaxContrastL1'})
    
    return tmp

