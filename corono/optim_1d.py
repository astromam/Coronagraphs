# -*- coding: utf-8 -*-
"""
#!/usr/bin/env python3
Created on Fri Mar  9 11:36:39 2018

@author: mndiaye
"""

#%% Initialization problem
import numpy as np
import json
import time

try:
    import stdgrb
except ModuleNotFoundError:
    stdgrb = False

try:
    import gurobipy as gb
except ModuleNotFoundError:
    gb = False

import scipy.optimize
from .utils import update_params
from . import design        

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
    tmp = {'cDarkHole':8,'tau':0.2,
           'cDarkHoledirect':2, 'solver':'stdgrb', 
           'slvCrossover':0, 'slvLogToConsole':1, 'slvMethod':2,
           'allLogToConsole':0, 
           'FirstDer':False, 'SecondDer': False,
           'FirstDerLim':0.01, 'SecondDerLim':0.0001,
           'MinIsland':False,
           'FirstDerGlobalLim':0.01}
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
def get_default_params_MaxContrastDirectProblemMatrix():
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
  
    def __init__(self,corono=design.APLC1d(),**kwargs):
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
            
        direct_field_re_t_tmp, direct_field_im_t_tmp : array_like, array_like
            Real and imaginary part of the non coronagraphic response matrix 
            for all the points in the pupil :math:`P_0` and at all the wavelengths
            
        corono_field_re_t_tmp, corono_field_im_t_tmp : array_like, array_like
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

        self.dzdirect      = (self.corono.xi >= self.corono.rho0direct) \
                & (self.corono.xi <= self.corono.rho1direct) 
        self.idx_dzdirect  = list(self.aaa[self.dzdirect])
        self.ndzdirect     = len(self.idx_dzdirect)

    
        self.pup     = (self.corono.Pupil1d > 0.)
        self.bbb     = np.arange(self.corono.nPup)
        self.idx_pup = list(self.bbb[self.pup])
        self.npp     = len(self.idx_pup)
    
        self.lys     = (self.corono.LyotStop1d > 0.)
        self.idx_lys = list(self.bbb[self.lys]) 
        
        self.corono_field_t = None
        self.direct_field_re_t_tmp = None
        
        self.A       = None
        self.b       = None
        self.c       = None
        
        self.m       = None
        
        self.Apod    = np.zeros((self.corono.nPup))

        self.TR      = np.sum(2.*np.pi*self.corono.Pupil1d *np.linspace(
                0.5,self.corono.nPup+0.5,num=self.corono.nPup)\
                /(2.*self.corono.nPup)**2)
               
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
    def compute_response_matrices(self):

        if self.problem_name == 'MaxTau' or self.problem_name == 'MaxTauMinIsland':
            direct_field_t_tmp = np.zeros((self.npp, self.corono.nlam, 
                                       self.corono.nImg+1), dtype='complex128')
            self.print_log('generating direct response matrices for 1D problem')
            t0 = time.time()
            Apod1d    = np.zeros((self.corono.nPup))
            for i, val in enumerate(self.idx_pup):
                Apod1d[val] = 1
                direct_field_t_tmp[i] = self.corono.compute_direct_field_1d(Apod1d)
                Apod1d[val] = 0            
            self.direct_field_re_t_tmp = direct_field_t_tmp.real
    #        self.direct_field_im_t_tmp = direct_field_t_tmp.imag    
            t1 = time.time()
            self.print_log('direct matrix computation time: {0:.2f}s'.format(t1-t0))


        corono_field_t_tmp = np.zeros((self.npp, self.corono.nlam, 
                                   self.corono.nImg+1), dtype='complex128')

        t0 = time.time()
        Apod1d    = np.zeros((self.corono.nPup))
        for i, val in enumerate(self.idx_pup):
            Apod1d[val] = 1
            corono_field_t_tmp[i] = self.corono.compute_corono_field_1d(Apod1d)
            Apod1d[val] = 0
        corono_field_re_t_tmp = corono_field_t_tmp.real
        corono_field_im_t_tmp = corono_field_t_tmp.imag
        t1 = time.time()
        self.print_log('corono matrix computation time: {0:.2f}s'.format(t1-t0))
        
        corono_field_re_t = np.reshape(
                corono_field_re_t_tmp[:,:,self.idx_dz], 
                (self.npp, self.corono.nlam*self.ndz))

        corono_field_im_t = np.reshape(
                corono_field_im_t_tmp[:,:,self.idx_dz], 
                (self.npp, self.corono.nlam*self.ndz))

        self.corono_field_t    = np.concatenate((corono_field_re_t,
                                                  corono_field_im_t), 
                                                  axis=1)


#%%        
    def solve_model(self):
        """
        Solves the optimization problem for the model using the gurobi solver.
        
        Returns
        -------
        Apod
            Solution for the optimization problem
        
        """
        self.compute_matrices()

        t0 = time.time()
        
        if stdgrb and self.solver == 'stdgrb':
            self.print_log('solving problem with stdgrb package')
            Apodtmp, val = stdgrb.lp_solve(self.c, A=(self.A).T, b=self.b, 
                                           crossover=self.slvCrossover, 
                                           logtoconsole=self.slvLogToConsole, method=self.slvMethod)
            self.Apod[self.idx_pup] = Apodtmp[:self.npp]

        elif gb and self.solver == 'gurobipy':
            self.print_log('generating gurobi model')
            self.compute_gurobi_model()
            
            self.print_log('solving problem with gurobipy package')                
            try:
                self.m.Params.Method       = self.slvMethod
                self.m.Params.LogToConsole = self.slvLogToConsole
                self.m.Params.Crossover    = self.slvCrossover
                
                self.m.optimize()
    
                for i, val in enumerate(self.idx_pup):
                    self.Apod[val] = self.m.getVars()[i].x
                    
            except gb.GurobiError as e:
                print('Error code ' + str(e.errno) + ": " + str(e))
    
            except AttributeError:
                print('Encountered an attribute error')
                        
        else:
            self.print_log('solving problem with scipy.optimize')
            bds = np.zeros((self.npp+self.neps+self.nvv, 2))
            bds[:,1] = 1.
            sol=scipy.optimize.linprog(self.c,self.A.T,self.b,
                                       method='interior-point',
                                       bounds=bds)
            self.Apod[self.idx_pup]=sol.x
            
        t1 = time.time()
        self.print_log('solving time: {0:.2f}s\n'.format(t1-t0))
        return self.Apod


#%%
    def get_filename(self, **kwargs):
        """
        Generate a string of characters to define a filename with all the 
        parameters
        
        Parameters
        --------
        kwargs : dict
            parameters given by the user for the keys with the values to update
        
        Returns
        --------
        fname_gen : str
            generic string of characters for a filename
            
        
        """
        params = self.params.copy()
        params = update_params(params, **kwargs)

        if self.problem_name == 'MaxTau' or self.problem_name == 'MaxTauMinIsland' :
            str_opt = '_C={cDarkHole:.1f}'
        elif self.problem_name == 'MaxContrastL1' or self.problem_name == 'MaxContrastLinf' or\
        self.problem_name == 'MaxContrastL1MinIsland' or self.problem_name == 'MaxContrastLinfMinIsland':
            str_opt = '_tau={tau:.3f}'
        else:
            raise NameError('{0}: Not an existing optimization problem!'.format(self.problem_name))

        if self.corono.corono_name == 'APLC' or self.corono.corono_name == 'SP': 
            str_cor = '_rMask={rMask:.3f}'
        elif self.corono.corono_name == 'HDZPM':
            str_cor = '_rMask1={rMask1:.3f}_rMask2={rMask2:.3f}'
        elif self.corono.corono_name == 'HTZPM':
            str_cor = '_rMask1={rMask1:.3f}_rMask2={rMask2:.3f}_rMask3={rMask3:.3f}'
        else:
            raise NameError('{0}: Not an existing coronagraph!'.format(self.corono.corono_name))
        
        if self.FirstDer is True:
            str_FirstDer = '_1stder={FirstDerLim}'
        else:
            str_FirstDer = ''

        if self.SecondDer is True:
            str_SecondDer = '_2ndder={SecondDerLim}'
        else:
            str_SecondDer = ''

        if (self.problem_name == 'MaxTauMinIsland' or self.problem_name == 'MaxContrastL1MinIsland' \
            or  self.problem_name == 'MaxContrastLinfMinIsland') and self.MinIsland is True:
            str_FirstDerGlobal = '_1stderglo={FirstDerGlobalLim}'
        else:
            str_FirstDerGlobal = ''
                    
        fname_gen   = '{corono_name}_obs={PupilObs:.2f}' + \
        '_lsid={LyotStopObs:.2f}_lsod={LyotStopIns:.2f}' + \
        '_IWA={rho0}_OWA={rho1}_BW={bw:.2f}_nlam={nlam:02d}' + \
        '_1D_N={nPup:04d}_nFPM={nFPM:03d}'+ str_cor + '_{problem_name}' + str_opt + \
        str_FirstDer + str_SecondDer + str_FirstDerGlobal + '_{solver}'
        
        return fname_gen.format(**params)
    
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

                         
        Returns
        -----------


        References
        ----------

                                
        """
        
        if self.corono_field_t is None:
            t00 = time.time()            
            self.print_log('computing response matrices for 1D problem')   
            self.compute_response_matrices()
            t11 = time.time()
            self.print_log('computing time (response matrices): {0:.2f}s\n'.format(t11-t00))

        if self.A is None or self.b is None or self.c is None:
            t00 = time.time()
            self.print_log('computing A, b, and c matrices')
            self.compute_problem_matrices()
            t11 = time.time()
            self.print_log('computing time (Abc matrices): {0:.2f}s\n'.format(t11-t00))
#        else:
#            if self.problem_name == 'MaxTau':
#                print('updating A matrix')
#                self.update_cDarkHole()
#            else:
#                print('updating b matrix')
#                self.update_tau()                

#%%
    def print_log(self, string):
        r"""
        Print a given log on the console depending on the value of 
        allLogtoConsole parameter
        
        Parameters:
        --------
        string: string
            string to be displayed on the console
            
        Returns:
        --------
        string: string
            string given by the user
        
        """
        if self.allLogToConsole == 1:
            return print(string)

        
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
        self.nvv  = 0

#%%        
    def compute_problem_matrices(self):
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
        cst = 10.**(-self.cDarkHole/2.)/np.sqrt(2.)

        A0  =  self.corono_field_t \
        - cst*self.direct_field_re_t_tmp[:,(self.corono.nlam-1)//2,0, None]
        A1  = -self.corono_field_t \
        - cst*self.direct_field_re_t_tmp[:,(self.corono.nlam-1)//2,0, None]
    
        b0  = np.zeros((self.corono.nlam*self.ndz*2))
        b1  = np.zeros((self.corono.nlam*self.ndz*2))
                                    
        self.A = np.concatenate((A0,A1), axis=1)
        self.b = np.concatenate((b0,b1))

        if (stdgrb and self.solver == 'stdgrb') or (gb and self.solver == 'gurobipy'):
            self.compute_problem_matrices_gurobi()
#            A2  = -np.identity(self.npp)
#            A3  =  np.identity(self.npp)
#            b2  = np.zeros(self.npp)
#            b3  = np.ones(self.npp)
#            self.A = np.concatenate((self.A,A2,A3), axis=1)
#            self.b = np.concatenate((self.b,b2,b3))

        if self.FirstDer is True:
            self.compute_problem_matrices_1stDer()
            
        if self.SecondDer is True:
            self.compute_problem_matrices_2ndDer()
               
        self.c = - 2.*np.pi*(np.asarray(self.idx_pup)+0.5)\
                /(2.*self.corono.nPup)**2/self.TR                 
        
        return self.A, self.b, self.c

#%%
    def compute_problem_matrices_gurobi(self):
        A2  = -np.identity(self.npp)
        A3  =  np.identity(self.npp)
        b2  = np.zeros(self.npp)
        b3  = np.ones(self.npp)
        self.A = np.concatenate((self.A,A2,A3), axis=1)
        self.b = np.concatenate((self.b,b2,b3))
        
#%%        
    def compute_problem_matrices_1stDer(self):
        A4  = np.diff(np.identity(self.npp), axis=1)
        b4  = self.FirstDerLim*np.ones(self.npp)
        self.A = np.concatenate((self.A, A4, -A4), axis=1)
        self.b = np.concatenate((self.b, b4, b4))

#%%        
    def compute_problem_matrices_2ndDer(self):
        A5  = np.diff(np.diff(np.identity(self.npp), axis=1), axis=1)
        b5  = self.SecondDerLim*np.ones(self.npp)
        self.A = np.concatenate((self.A, A5, -A5), axis=1)
        self.b = np.concatenate((self.b, b5, b5))        

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
            
        self.nvv   = 0

#%%    
    def compute_problem_matrices(self):
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
            self.N0 = np.zeros((1, self.npp))
            Z0 = np.zeros(1)
            c1 = [1]
        else:
            I0 = np.identity(self.ndz)
            I1 = np.hstack([I0 for k in range(self.corono.nlam*2)])            
            self.N0 = np.zeros((self.ndz, self.npp))
            Z0 = np.zeros(self.ndz)
            c1 = 2.*np.pi*np.asarray(self.idx_dz)*(self.corono.Fmax\
                                  /self.corono.nImg)**2
        
        A0  = np.concatenate(( self.corono_field_t, -I1), axis=0)
        A1  = np.concatenate((-self.corono_field_t, -I1), axis=0)
        A6  = np.concatenate((np.zeros((self.npp, self.ndz)), -I0), axis=0)
        A7  = np.concatenate((- 2.*np.pi*(np.asarray(self.idx_pup)+0.5)\
            *self.corono.Pupil1d[self.idx_pup]/(2.*self.corono.nPup)**2/self.TR, 
                                  Z0))
        
        b0  = np.zeros((self.corono.nlam*self.ndz*2))
        b1  = np.zeros((self.corono.nlam*self.ndz*2))
        b6  = np.zeros(self.ndz)
        b7  = [-self.tau]
                        
        self.A = np.concatenate((A0,A1,A6,A7[:,None]), axis=1)
        self.b = np.concatenate((b0,b1,b6,b7))

        if (stdgrb and self.solver == 'stdgrb') or (gb and self.solver == 'gurobipy'):
            self.compute_problem_matrices_gurobi()
            
        if self.FirstDer is True:
            self.compute_problem_matrices_1stDer()

        if self.SecondDer is True:
            self.compute_problem_matrices_2ndDer()     
            
        self.c = np.concatenate((np.zeros(self.npp), c1), axis=0)        
        
        return self.A, self.b, self.c

#%%
    def compute_problem_matrices_gurobi(self):
        A2  = np.concatenate((-np.identity(self.npp), self.N0), axis=0)
        A3  = np.concatenate(( np.identity(self.npp), self.N0), axis=0)
        b2  = np.zeros(self.npp)
        b3  = np.ones(self.npp)
        self.A = np.concatenate((self.A,A2,A3), axis=1)
        self.b = np.concatenate((self.b,b2,b3))        

#%%        
    def compute_problem_matrices_1stDer(self):
        A4  = np.diff(np.identity(self.npp), axis=1)
        A4  = np.concatenate((A4, self.N0[:, :self.npp-1]), axis=0)
        b4  = self.FirstDerLim*np.ones(self.npp)
        self.A = np.concatenate((self.A, A4, -A4), axis=1)
        self.b = np.concatenate((self.b, b4, b4))        

#%%        
    def compute_problem_matrices_2ndDer(self):
        A5  = np.diff(np.diff(np.identity(self.npp), axis=1), axis=1)
        A5  = np.concatenate((A5, self.N0[:, :self.npp-2]), axis=0)
        b5  = self.SecondDerLim*np.ones(self.npp)
        self.A = np.concatenate((self.A, A5, -A5), axis=1)
        self.b = np.concatenate((self.b, b5, b5))

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

#%%
"""
MaxTau ProblemMatrix subclass
"""
class MaxTauMinIsland(ProblemMatrix):
    r"""
    Defines the ProblemMatrix subclass for the optimization problem that 
    maximizes the integrated apodizer transmission for a given contrast 
    :math:`C` in the search area inside the coronagraphic image.
    """
    def __init__(self, **kwargs):
        """
        Constructor for the Matrix problem with the coronagraph object
        
        """
        super(MaxTauMinIsland, self).__init__(**kwargs)
        self.neps = 0
        self.nvv  = 2*(self.npp-1)

#%%        
    def compute_problem_matrices(self):
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
        cst = 10.**(-self.cDarkHole/2.)/np.sqrt(2.)

        A0  =  self.corono_field_t \
        - cst*self.direct_field_re_t_tmp[:,(self.corono.nlam-1)//2,0, None]
        A1  = -self.corono_field_t \
        - cst*self.direct_field_re_t_tmp[:,(self.corono.nlam-1)//2,0, None]
    
        b0  = np.zeros((self.corono.nlam*self.ndz*2))
        b1  = np.zeros((self.corono.nlam*self.ndz*2))
                                    
        self.A = np.concatenate((A0,A1), axis=1)
        self.b = np.concatenate((b0,b1))

        if (stdgrb and self.solver == 'stdgrb') or (gb and self.solver == 'gurobipy'):
            A2  = -np.identity(self.npp)
            A3  =  np.identity(self.npp)
            b2  = np.zeros(self.npp)
            b3  = np.ones(self.npp)
            self.A = np.concatenate((self.A,A2,A3), axis=1)
            self.b = np.concatenate((self.b,b2,b3))

        if self.MinIsland is True:
            A00    = np.zeros((self.nvv, np.shape(self.A)[1]))            
            self.A = np.concatenate((self.A, A00))

            AD  = np.diff(np.identity(self.npp), axis=1)
            AI  = np.identity((self.npp-1))
            
            A4   = np.concatenate(( AD, -AI,  AI))
            A5   = np.concatenate((-AD,  AI, -AI))
            
            AZ1 = np.zeros((self.npp, self.npp-1))                       
            AZ2 = np.zeros((self.npp-1, self.npp-1))

            A6  = np.concatenate((AZ1, -AI, AZ2))
            A7  = np.concatenate((AZ1, AZ2, -AI))

            bZ  = np.zeros((self.npp-1))

            A8Z = np.zeros((self.npp))
            A81 = np.ones(self.npp-1)
            A8  = np.concatenate((A8Z, A81, A81))
            b8  = [self.FirstDerGlobalLim]
            
            self.A = np.concatenate((self.A, A4, A5, A6, A7, A8[:, None]), axis=1)
            self.b = np.concatenate((self.b, bZ, bZ, bZ, bZ, b8))

        ctmp = np.zeros((self.npp + self.nvv))
        ctmp[:self.npp] = np.asarray(self.idx_pup)+0.5
        self.c = - 2.*np.pi*ctmp/(2.*self.corono.nPup)**2/self.TR
                                 
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
         
        nA = np.shape(self.A)[1]
    
        # Create a new model               
        self.m = gb.Model("LP max tau new")
        # Create variables
        ApodTmp = self.m.addVars(self.npp+self.nvv, lb=0.0, ub=1.0, name="ApodTmp")
        # Set objective
        self.m.setObjective(gb.quicksum((self.c[i]*ApodTmp[i] 
                for i in range(self.npp))), gb.GRB.MINIMIZE)
        # Add constraint:                
        self.m.addConstrs((gb.quicksum((ApodTmp[i]*self.A[i,j] 
                for i in range(self.npp+self.nvv) if self.A[i,j])) <=  self.b[j] 
                for j in range(nA)), "cpos")
        self.m.update()          


#%%
"""
MaxContrast ProblemMatrix subclass
"""
class MaxContrastMinIsland(ProblemMatrix):
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
        super(MaxContrastMinIsland,self).__init__(**kwargs)

        if self.Lnorm == 'Linf':
            self.neps = 1
        else:
            self.neps = self.ndz
            
        self.nvv   = 2*(self.npp-1)

#%%    
    def compute_problem_matrices(self):
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
            c1 = 2.*np.pi*np.asarray(self.idx_dz)*(self.corono.Fmax\
                                  /self.corono.nImg)**2
        
        A0  = np.concatenate(( self.corono_field_t, -I1), axis=0)
        A1  = np.concatenate((-self.corono_field_t, -I1), axis=0)
        A6  = np.concatenate((np.zeros((self.npp, self.ndz)), -I0), axis=0)
        A7  = np.concatenate((- 2.*np.pi*(np.asarray(self.idx_pup)+0.5)\
            *self.corono.Pupil1d[self.idx_pup]/(2.*self.corono.nPup)**2/self.TR, 
                                  Z0))
        
        b0  = np.zeros((self.corono.nlam*self.ndz*2))
        b1  = np.zeros((self.corono.nlam*self.ndz*2))
        b6  = np.zeros(self.ndz)
        b7  = [-self.tau]
                        
        self.A = np.concatenate((A0,A1,A6,A7[:,None]), axis=1)
        self.b = np.concatenate((b0,b1,b6,b7))

        if (stdgrb and self.solver == 'stdgrb') or (gb and self.solver == 'gurobipy'):
            A2  = np.concatenate((-np.identity(self.npp), N0), axis=0)
            A3  = np.concatenate(( np.identity(self.npp), N0), axis=0)
            b2  = np.zeros(self.npp)
            b3  = np.ones(self.npp)
            self.A = np.concatenate((self.A,A2,A3), axis=1)
            self.b = np.concatenate((self.b,b2,b3))
            
        if self.MinIsland is True:
            A00    = np.zeros((self.nvv, np.shape(self.A)[1]))            
            self.A = np.concatenate((self.A, A00))

            AD  = np.diff(np.identity(self.npp), axis=1)
            AI  = np.identity((self.npp-1))
            
            AZ0 = np.zeros((self.neps, self.npp-1))
            
            A8   = np.concatenate(( AD, AZ0, -AI,  AI))
            A9   = np.concatenate((-AD, AZ0,  AI, -AI))
            
            AZ1 = np.zeros((self.npp, self.npp-1))                       
            AZ2 = np.zeros((self.npp-1, self.npp-1))

            A10  = np.concatenate((AZ1, AZ0, -AI, AZ2))
            A11  = np.concatenate((AZ1, AZ0, AZ2, -AI))

            bZ  = np.zeros((self.npp-1))

            A12Z = np.zeros((self.npp))
            A121 = np.ones(self.npp-1)
            A12  = np.concatenate((A12Z, np.zeros((self.neps)), A121, A121))
            b12  = [self.FirstDerGlobalLim]
                        
            self.A = np.concatenate((self.A, A8, A9, A10, A11, A12[:, None]), axis=1)
            self.b = np.concatenate((self.b, bZ, bZ, bZ, bZ, b12))
                   
        self.c = np.concatenate((np.zeros(self.npp), c1, np.zeros(self.nvv)), axis=0)
        
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
                          
        nn = np.shape(self.A)[1]
        # Create a new model  
        self.m = gb.Model("LP max C new")
        
        # Create variables
        ApodEpsTmp = self.m.addVars(self.npp + self.nvv +self.neps, lb=0.0, 
                                    name="ApodEpsTmp")        
        # Set objective
        self.m.setObjective(gb.quicksum((self.c[i+self.npp]*ApodEpsTmp[i+self.npp] 
                for i in range(self.neps))), gb.GRB.MINIMIZE)
        # Add constraint:
        self.m.addConstrs((gb.quicksum((ApodEpsTmp[i]*self.A[i,j] 
                for i in range(self.npp + self.nvv + self.neps) if self.A[i,j])) <=  self.b[j] 
                for j in np.arange(nn)), "cpos")
        
        self.m.update()
            
        return self.m

