# -*- coding: utf-8 -*-
"""
#!/usr/bin/env python3
Created on Fri Mar  9 11:36:39 2018

@author: mndiaye
"""

#%% Initialization problem
import numpy as np
import json

import sys    
sys.modules['gurobipy'] = None
sys.modules['stdgrb']   = None    
#del sys.modules['gurobipy']
#del sys.modules['stdgrb']

try:
    import stdgrb
except ModuleNotFoundError:
    try:
        import gurobipy as gb
    except ImportError:
        import scipy.optimize
    
from corono import corono_design as cd

#%%
"""
Default parameters
"""
#%%
def get_default_params_ProblemMatrix():
    r"""
    Gets the default parameters for the optimization problem matrix.
       
    Parameters
    ----------
    cDarkHole : float (default=8)
        Contrast goal :math:`C` in log scale inside the search area in the 
        coronagraphic image.
    
    tau : float (default=0.2)
        Integrated amplitude transmission :math:`\tau` of the apodizer :math:`\Phi` in 
        fraction of the integrated amplitude transmissio of the pupil 
        :math:`P_0`.
    
    Returns    
    ----------
    tmp : dict
        Dictionnary of parameters with their default values.
        
    """
    tmp = {'cDarkHole':8,'tau':0.2}
    return tmp

#%%
def get_default_params_MaxContrastProblemMatrix():
    r"""
    Gets the default parameters for the Max contrast optimization problem.
    
    Parameters
    ---------- 
    tmp : dict
        Dictionary from the get_default_matrix_pb
        
    Lnorm : string (default= 'L1')
        L-norm type for the optimization problem 
        ('Linf' : :math:`L_{\infty}` norm, 'L1' : :math:`L_1`-norm)
            
    Returns    
    ----------
    tmp : dict
        Updated dictionary
        
    """
    
    tmp = get_default_params_ProblemMatrix()
    tmp.update({'Lnorm':'L1'})
    return tmp

#%%
"""
Problem Matrix class
"""
class ProblemMatrix(object):
    r"""
    Defines the class for Matrix of optimization problem
    """
    default_params = get_default_params_ProblemMatrix()
  
    def __init__(self,corono=cd.APLC1d(),**kwargs):
        r"""
        __init__ : method
            Constructor for the ProblemMatrix class
        
        Attributes
        ----------
        params : dict
            Dictionary of parameters for the Coronagraph class
        
        corono : class
            Object of the Coronagraph class
        
        dz : array_like
            Dark zone points in the coronagraphic image
            
        aaa : array_like
            Vector indexing the points in the coronagraphic image
            
        idz_dz : array_like
            Vector indexing the points of the dark zone in the coronagraphic 
            image
            
        ndz : int
            Number of points of the dark zone in the coronagraphic image
            
        pup : array_like
            Index of non zero points in the pupil :math:`P_0`
            
        bbb : array_like
            Vector indexing the points in the pupil :math:`P_0`
            
        idx_pup : array_like
            Vector indexing the non zero points in the pupil  :math:`P_0`
            
        npp : int
            Number of non zero points in the pupil :math:`P_0`
            
        lys : array_like
            Index of non zero points in the Lyot stop  :math:`L`
            
        idx_lys : array_like
            Vector indexing the non zero points of the pupil in the Lyot stop
            :math:`L`
            
        direct_field_t_re, direct_field_t_im : array_like, array_like
            Real and imaginary part of the non coronagraphic response matrix 
            for all the points in the pupil :math:`P_0` and at all the wavelengths
            
        corono_field_t_re, corono_field_t_im : array_like, array_like
            Real and imaginary part of the coronagraphic response matrix 
            for all the points in the pupil :math:`P_0` and at all the wavelengths
         
        corono_field_t_re2 : array_like
            Real part of the coronagraphic response matrix 
            for all the non zero points in the pupil and at all the wavelengths
                                
        A, b, c : array_like, array_like, array_like
            Matrices for the optimization problem that writes as
            
            .. math:: \max_{\tau} c^{T}.x,    
            under the constraint :math:`A.x \leq b`.
        
        TR : float
            Integrated amplitude transmission of the pupil :math:`P_0` 
            with respect to that of the clear pupil
            
        m : gurobi model
            Gurobi model of the problem to solve
            
        Apod : array_like
            Apodizer :math:`\Phi` to be generated
        
        """
        self.params  = kwargs
        self.check_params()
        
        self.corono  = corono
        
        self.dz      = (self.corono.xi >= self.corono.rho0) \
                & (self.corono.xi <= self.corono.rho1)
        self.aaa     = np.arange(self.corono.nImg+1) 
        self.idx_dz  = list(self.aaa[self.dz])
        self.ndz     = len(self.idx_dz)
    
        self.pup     = (self.corono.Pupil1d > 0.)
        self.bbb     = np.arange(self.corono.nPup)
        self.idx_pup = list(self.bbb[self.pup])
        self.npp     = len(self.idx_pup)
    
        self.lys     = (self.corono.LyotStop1d > 0.)
        self.idx_lys = list(self.bbb[self.lys]) 
    
        direct_field_t = np.zeros((self.corono.nPup, self.corono.nlam, 
                                   self.corono.nImg+1), dtype='complex128')
        #print('generating direct response matrices for 1D problem')
        Apod1d    = np.zeros((self.corono.nPup))
        for i in np.arange(self.corono.nPup):
            Apod1d[i] = 1
            direct_field_t[i] = self.corono.compute_direct_field_1d(Apod1d)
            Apod1d[i] = 0            
        self.direct_field_t_re = direct_field_t.real
        self.direct_field_t_im = direct_field_t.imag    

        corono_field_t = np.zeros((self.corono.nPup, self.corono.nlam, 
                                   self.corono.nImg+1), dtype='complex128')
        #print('generating corono response matrices for 1D problem')
        Apod1d    = np.zeros((self.corono.nPup))
        for i in np.arange(self.corono.nPup):
            Apod1d[i] = 1
            corono_field_t[i] = self.corono.compute_corono_field_1d(Apod1d)
            Apod1d[i] = 0
        self.corono_field_t_re = corono_field_t.real
        self.corono_field_t_im = corono_field_t.imag


        self.corono_field_t2_re = np.reshape(
                self.corono_field_t_re[:,:,self.idx_dz], 
                (self.corono.nPup, self.corono.nlam*self.ndz))[self.idx_pup,:]

        self.corono_field_t2_im = np.reshape(
                self.corono_field_t_im[:,:,self.idx_dz], 
                (self.corono.nPup, self.corono.nlam*self.ndz))[self.idx_pup,:]

        self.corono_field_t2    = np.concatenate((self.corono_field_t2_re,
                                                  self.corono_field_t2_im), 
                                                 axis=1)
        
        self.A       = None
        self.b       = None
        self.c       = None
        
        self.m       = None
        
        self.Apod    = np.zeros((self.corono.nPup))

        self.TR      = np.sum(2.*np.pi*self.corono.Pupil1d *np.linspace(
                0.5,self.corono.nPup+0.5,num=self.corono.nPup)\
                /(2.*self.corono.nPup)**2)
               

#%%    
    def compute_matrices(self):
        """
        Virtual function for the matrix computation.
        
        Raises
        --------
        res
            Display of a warning
        
        """
        print('Warning: virtual fct - no A, b and c matrices will be computed')

#%%
    def __contains__(self, item):
        """
        Checks params for a given item.
        
        Parameters
        ----------
        item : string 
            Key in params dictionary
        
        Returns
        ----------
        res
            Value for the item in the dictionary  
            
        """
        return item in self.params
    
#%%     
    def __getattr__(self, name):
        """
        Checks the attribute for the params.
        
        Parameters
        ----------
        name : string
            Name of the variable to be retrieved
        
        Returns
        ----------
        res
            Value for name in params
        
        """
        return self.params[name]

#%%        
    def check_params(self):
        """
        Check the params.
        
        Returns
        ----------
        res
            Values for all the keys in the params
        
        """        
        for key in self.default_params:
            if not key in self.params:
                self.params[key] = self.default_params[key]
  
#%%    
    def load_params(self, fname):
        """
        Loads the params from a given filename 
        using JavaScript Object Notation (JSON).
        
        Parameters
        ----------
        fname : string
            Filename to load    
        
        """
        
        f=open(fname,'r')
        params=json.loads(f.read())
        self.__init__(**params)
        f.close()              

#%%        
    def solve_model(self):
        """
        Solves the optimization problem for the model using the gurobi solver.
        
        Returns
        -------
        Apod
            Solution for the optimization problem
        
        """

        try:
            print('solving with stdgrb package')
            Apodtmp, val = stdgrb.lp_solve(self.c, A=(self.A).T, b=self.b, 
                                           crossover=1, logtoconsole=1, method=2)
            self.Apod[self.idx_pup] = Apodtmp
            return self.Apod

        except NameError:
            print('stdgrb   package not found -> solving with gurobipy')                
            if self.m is not None:            

                try:
                    self.m.Params.Method       = 2
                    self.m.Params.LogToConsole = 1
                    self.m.Params.Crossover    = 0
                    
                    self.m.optimize()
        
                    Apodtmp = np.zeros((self.npp))
                    for i in range(self.npp):
                        Apodtmp[i] = self.m.getVars()[i].x
                    self.Apod[self.idx_pup] = Apodtmp
                                    
                    return self.Apod
        
                except gb.GurobiError as e:
                    print('Error code ' + str(e.errno) + ": " + str(e))
        
                except AttributeError:
                    print('Encountered an attribute error')
                        
            else:
                print('gurobipy package not found -> solving with scipy.optimize')
                bds = np.zeros((self.npp+self.neps, 2))
                bds[:,1] = 1.
                sol=scipy.optimize.linprog(self.c,self.A.T,self.b,
                                           method='interior-point',
                                           bounds=bds)
                self.Apod[self.idx_pup]=sol.x
                return self.Apod
        
#%%
"""
MaxTau ProblemMatrix subclass
"""
class MaxTau(ProblemMatrix):
    r"""
    Defines the ProblemMatrix subclass for the optimization problem that 
    maximizes the integrated apodizer transmission for a given contrast 
    :math:`C` in the search area inside the coronagraphic image.
    """
    def __init__(self, **kwargs):
        """
        Constructor for the Matrix problem with the coronagraph object
        
        """
        super(MaxTau, self).__init__(**kwargs)
        self.neps = 0

#%%        
    def compute_matrices(self):
        r"""
        Computes the matrices for the optimization problem that consists in 
        maximizing the amplitude transmission of the apodizer :math:`\Phi` 
        for a set contrast :math:`C` in a given search area in the coronagraphic 
        image. In terms of matrices, the optimization problem writes as
            
        .. math:: \max_{C} c^{T}.x,
            
        under the constraint :math:`A.x \leq b`.
                
        The variable :math:`x` represents the apodizer transmission 
        function :math:`\Phi`. The variables follow the notations of [1]_ and [2]_.       
        
        Parameters
        -----------        
        A0, b0 : array_like, array_like
            Contrast constraint on the coronagraphic electric field :math:`\Psi_D`
            that is represented the following equation:
                
            :math:`\Psi_D(\xi,\lambda)-10^{-C/2}\Psi_0(\xi,\lambda) \leq 0`. 
            
            :math:`\xi` and :math:`\lambda` denote the image plane coordinate 
            and wavelength. The term :math:`\Psi_0` represents the coronagraphic 
            electric field in the absence of focal plane mask (FPM).
            
        A1, b1 : array_like, array_like
            Contrast constraints on the coronagraphic electric field Psi_D
            that is represented the following equations:
                
            :math:`-\Psi_D(\xi,\lambda)-10^{-C/2}\Psi_0(\xi,\lambda) \leq 0`.

        A2, b2 : array_like, array_like
            Constraint on the transmission of the amplitude apodization 
            :math:`\Phi`
            
            :math:`- \Phi(r) \leq 0`,
            in which :math:`r` represents the radial coordinate of the pupil.
            
        A3, b3 : array_like, array_like
            Constraint on the transmission of the amplitude apodization 
            :math:`\Phi`
            
            :math:`\Phi(r) \leq 1`.
        
        Returns 
        ----------
        A, b, c: array_like, array_like, array_like
            The matrices for the optimization problem.
            A and b are concatenations of the matrices for the constraints that 
            are described above.
            
            The cost function c to maximize is the transmission of the apodizer 
            inside the pupil :math:`P_0`.
            
            .. math:: \max_{C}[\int_{P_0} \Phi(r)dr].
            
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
        ED0tmp = np.zeros((self.corono.nPup, self.corono.nlam*self.ndz*2))
        for j in range(self.corono.nlam*self.ndz*2):
            ED0tmp[:,j] = \
            self.direct_field_t_re[:,(self.corono.nlam-1)//2,0]
        ED0 = ED0tmp[self.idx_pup,:]
        cst = 10.**(-self.cDarkHole/2.)/np.sqrt(2.)
        ED0 *= cst
       
        A0  =  self.corono_field_t2 - ED0
        A1  = -self.corono_field_t2 - ED0
    
        b0  = np.zeros((self.corono.nlam*self.ndz*2))
        b1  = np.zeros((self.corono.nlam*self.ndz*2))
        
        if sys.modules['gurobipy'] is None and sys.modules['stdgrb'] is None:
            print('reduced matrix shape')
            self.A = np.concatenate((A0,A1), axis=1)
            self.b = np.concatenate((b0,b1))            
        else:
            A2  = -np.identity(self.npp)
            A3  =  np.identity(self.npp)
            b2  = np.zeros(self.npp)
            b3  = np.ones(self.npp)
            self.A = np.concatenate((A0,A1,A2,A3), axis=1)
            self.b = np.concatenate((b0,b1,b2,b3))
        self.c = - 2.*np.pi*(np.arange(self.corono.nPup)[self.idx_pup]+0.5)\
                /(2.*self.corono.nPup)**2/self.TR
        
        return self.A, self.b, self.c

#%%
    def compute_gurobi_model(self):
        """
        Generates the gurobi solver model for the MaxTau problem.
        
        Parameters 
        -----------
        Apodtmp : array_like
            Vector of the apodizer :math:`\Phi` in the non zero points of 
            the pupil :math:`P_0`
        
        Returns
        -----------
        m : gurobi model
            Gurobi model of the MaxTau problem to solve
            
        """

        try:
            if self.A is None or self.b is None or self.c is None:
                print('computing matrices')
                self.compute_matrices()
                
            nA = np.shape(self.A)[1]
        
            # Create a new model               
            self.m = gb.Model("LP max tau new")
            # Create variables
            ApodTmp = self.m.addVars(self.npp, lb=0.0, ub=1.0, name="ApodTmp")
            # Set objective
            self.m.setObjective(gb.quicksum((self.c[i]*ApodTmp[i] 
                    for i in range(self.npp))), gb.GRB.MINIMIZE)
            # Add constraint:                
            self.m.addConstrs((gb.quicksum((ApodTmp[i]*self.A[i,j] 
                    for i in range(self.npp) if self.A[i,j])) <=  self.b[j] 
                    for j in range(nA)), "cpos")
            self.m.update()  
        
        except NameError:
            pass

#%%
"""
MaxContrast ProblemMatrix subclass
"""
class MaxContrast(ProblemMatrix):
    r"""
    Defines the ProblemMatrix subclass for the optimization problem that 
    maximizes the contrast in a given search area for a given integrated 
    apodizer transmission :math:`\tau`.
    """
    default_params = get_default_params_MaxContrastProblemMatrix()
    
    def __init__(self, **kwargs):
        """
        Constructor for the Matrix problem with 
        the coronagraph object.
        """
        super(MaxContrast,self).__init__(**kwargs)

        if self.Lnorm == 'Linf':
            self.neps = 1
        else:
            self.neps = self.ndz

#%%    
    def compute_matrices(self):
        r"""
        Computes the matrices for the optimization problem that consists in 
        maximizing the contrast in a given search area in the coronagraphic 
        image for a set integrated apodizer transmission :math:`\tau`. In terms
        of matrices, the optimization problem writes as
            
        .. math:: \max_{\tau} c^{T}.x,
            
        under the constraint :math:`A.x \leq b`.
                
        The variable :math:`x` is a concatenation of the apodizer transmission 
        function :math:`\Phi` and an auxiliary variable :math:`\epsilon`.
        The variable :math:`\epsilon` represents the contrast to maximize in 
        the search area ranging between :math:`\rho_0` and :math:`\rho_1` in 
        the coronagraphic image. It can either depend on the position 
        :math:`\xi` in the coronagraphic image or not (:math:`L_1`-norm or 
        :math:`L_\infty`-norm problem). 
        The variables follow the notations of [1]_ and [2]_.

        Parameters
        -----------
        Lnorm : string
            Type of L-norm for the optimization problem
            
        I0, I1, N0, Z0, c1 : array_like
            Intermediate matrices for the generation of the matrices A0 to A5.
            They depend on the type of the norm (:math:`L_1` or :math:`L_\infty`) 
            for the problem.

        A0, b0 : array_like, array_like
            Contrast constraint on the coronagraphic electric field :math:`\Psi_D`
            that is represented the following equation:
                
            :math:`\Psi_D(\xi,\lambda)-\epsilon(\xi) \leq 0` 
            if :math:`L_1`-norm constraint,
            
            :math:`\Psi_D(\xi,\lambda)-\epsilon    \leq 0`  
            if :math:`L_\infty`-norm constraint,
            
            :math:`\xi` and :math:`\lambda` denote the image plane coordinate 
            and wavelength.

        A1, b1 : array_like, array_like
            Contrast constraint on the coronagraphic electric field :math:`\Psi_D`
            that is represented the following equation:
                
            :math:`-\Psi_D(\xi, \lambda) - \epsilon(\xi) \leq 0`
            if :math:`L_1`-norm constraint,
            
            :math:`-\Psi_D(\xi,\lambda) - \epsilon    \leq 0`
            if :math:`L_\infty`-norm constraint.
            

        A2, b2 : array_like, array_like
            Constraint on the transmission of the amplitude apodization 
            :math:`\Phi`
            
            :math:`- \Phi(r) \leq 0`,
            in which :math:`r` represents the radial coordinate of the pupil.
            
        A3, b3 : array_like, array_like
            Constraint on the transmission of the amplitude apodization 
            :math:`\Phi`
            
            :math:`\Phi(r) \leq 1`.
            
        A4, b4 : array_like, array_like
            Constraint on the variable epsilon that is related to contrast
            and represented by the following equation:
                
            :math:`-\epsilon(\xi) \leq 0` if :math:`L_1`-norm constraint,
            
            :math:`-\epsilon     \leq 0` if :math:`L_\infty`-norm constraint.            
                  
        A5, b5 : array_like
            Constraint on the integral of the apodization amplitude transmission, 
            used as a proxy of the apodizer throughput. It is normalized to the
            integral of the transmission of the pupil :math:`P_0` and is larger 
            than a parameter :math:`\tau` set by the user.
            
            .. math:: - \frac{\int_{P} \Phi(r)dr}{\int_{P_0} P(r)dr} \leq \tau.
                  
        
        Returns
        -----------
        A, b, c: array_like, array_like, array_like
            The matrices for the optimization problem.
            A and b are concatenations of the matrices for the constraints that 
            are described above.
            
            The cost function c to maximize represents the contrast 
            in the search area ranging between :math:`\rho_0` and :math:`\rho_1` 
            inside the coronagraphic image.
            
            .. math:: \max_{\tau}[ - \int_{\rho_0}^{\rho_1} W(\xi)\epsilon(\xi)d\xi],
            
            with
            
            :math:`W(\xi)= \xi` if :math:`L_1`-norm constraint
            
            :math:`W(\xi) = 1` if :math:`L_\infty`-norm constraint

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
        if self.Lnorm == 'Linf':
            I1 = np.ones(self.ndz*self.corono.nlam*2)
            I1 = I1[None,:]
            I0 = np.ones(self.ndz)
            I0 = I0[None,:]
            N0 = np.zeros((1, self.npp))
            Z0 = np.zeros(1)
            c1 = [1]
        else:
            I0 = np.identity(self.ndz)
            I1 = np.hstack([I0 for k in range(self.corono.nlam*2)])            
            N0 = np.zeros((self.ndz, self.npp))
            Z0 = np.zeros(self.ndz)
            c1 = 2.*np.pi*np.array(self.idx_dz)*(self.corono.Fmax\
                                  /self.corono.nImg)**2
        
        A0  = np.concatenate(( self.corono_field_t2, -I1), axis=0)
        A1  = np.concatenate((-self.corono_field_t2, -I1), axis=0)
        A4  = np.concatenate((np.zeros((self.npp, self.ndz)), -I0), axis=0)
        A5  = np.concatenate((- 2.*np.pi*(
                np.arange(self.corono.nPup)[self.idx_pup]+0.5)\
            *self.corono.Pupil1d[self.idx_pup]/(2.*self.corono.nPup)**2/self.TR, 
                                  Z0))
        
        b0  = np.zeros((self.corono.nlam*self.ndz*2))
        b1  = np.zeros((self.corono.nlam*self.ndz*2))
        b4  = np.zeros(self.ndz)
        b5  = [-self.tau]
        
        if sys.modules['gurobipy'] is None and sys.modules['stdgrb'] is None:
            self.A = np.concatenate((A0,A1,A4,A5[:,None]), axis=1)
            self.b = np.concatenate((b0,b1,b4,b5))            
        else:
            A2  = np.concatenate((-np.identity(self.npp), N0), axis=0)
            A3  = np.concatenate(( np.identity(self.npp), N0), axis=0)
            b2  = np.zeros(self.npp)
            b3  = np.ones(self.npp)
            self.A = np.concatenate((A0,A1,A2,A3,A4,A5[:,None]), axis=1)
            self.b = np.concatenate((b0,b1,b2,b3,b4,b5))        
        self.c = np.concatenate((np.zeros(self.npp), c1), axis=0)
        
        return self.A, self.b, self.c

#%%
    def compute_gurobi_model(self):
        r"""
        Generates the gurobi solver model for the MaxContrast problem.
        
        Parameters 
        -----------
        ApodEpstmp : array_like
            Vector of the apodizer in the non zero points of the pupil 
            :math:`P_0` and neps points for the coronagraphic image
        
        Returns
        -----------
        m : gurobi model
            Gurobi model of the MaxContrast problem to solve
            
        """        

        try:
            if self.A is None or self.b is None or self.c is None:
                print('computing matrices')
                self.compute_matrices()
                           
            nn = np.shape(self.A)[1]
            # Create a new model  
            self.m = gb.Model("LP max C new")
            

            # Create variables
            ApodEpsTmp = self.m.addVars(self.npp + self.neps, lb=0.0, name="ApodEpsTmp")        
            # Set objective
            self.m.setObjective(gb.quicksum((self.c[i+self.npp]*ApodEpsTmp[i+self.npp] 
                    for i in range(self.neps))), gb.GRB.MINIMIZE)
            # Add constraint:
            self.m.addConstrs((gb.quicksum((ApodEpsTmp[i]*self.A[i,j] 
                    for i in range(self.npp + self.neps) if self.A[i,j])) <=  self.b[j] 
                    for j in np.arange(nn)), "cpos")
            
            self.m.update()
                
            return self.m
        
        except NameError:
            pass

#%%