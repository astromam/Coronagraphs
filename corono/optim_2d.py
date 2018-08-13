#!/usr/bin/env python3
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
        
    Binarity : bool (default=False)
        introduce constraints of the apodizer transmission in the optimization 
        problem to maximize the number of binary points in the apodization.
    
    BinarityReg : float (default=0.01)
        Regularization term on the binarity of the apodizer transmission.
    
    Returns    
    ----------
    tmp : dict
        Dictionary of parameters with their default values.
        
    References
    ----------        
    .. [1] Gurobi Optimization, LLC, Gurobi Optimizer Reference Manual (2018).
    
           http://www.gurobi.com
        
    """
    tmp = {'cDarkHole':8, 'tau':0.2, 'solver':'stdgrb', 
           'pupil_name':'sbr', 
           'problem_name':'MaxTau',
           'slvCrossover':0, 'slvLogToConsole':1, 'slvMethod':2,
           'allLogToConsole':0,
           'MinIsland':False, 'FirstDerGlobalLim':0.01,
           'Binarity':False, 'BinarityReg':0.1}
    return tmp

#%%
def get_default_params_MaxTauProblemMatrix():
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
    tmp = get_default_params_ProblemMatrix()
    tmp.update({'problem_name':'MaxTau'})
    
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
    tmp = get_default_params_ProblemMatrix()
    tmp.update({'Lnorm':'L1', 'problem_name':'MaxContrastL1'})
    
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
  
    def __init__(self,corono=design.APLC2d(), **kwargs):
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
                                            
        A, b, c : array_like, array_like, array_like
            Matrices for the optimization problem that writes as
            
            .. math:: \max_{\tau} c^{T}.x,    
            under the constraint :math:`A.x \leq b`.

        m : gurobi model
            Gurobi model of the problem to solve
                    
        TR : float
            Integrated amplitude transmission of the pupil :math:`P_0` 
            with respect to that of the clear pupil
            
        Apod : array_like
            Apodizer :math:`\Phi` to be generated
        
        """
        self.params  = kwargs
        self.check_params()
        
        if isinstance(corono, list) == True:
            self.corono   = corono[0]
            self.corono_t = corono
            self.ncorono  = len(self.corono_t)
            print('number of corono: {0}'.format(self.ncorono))
            self.LyotStop_vec_t = np.zeros((self.ncorono, (self.corono.nPup**2)))
            for i in range(self.ncorono):
                self.LyotStop_vec_t[i] = np.reshape(self.corono_t[i].LyotStop2d, (self.corono.nPup**2))                
        else:
            self.corono  = corono
            self.corono_t = [corono]
            self.ncorono = 1
            self.LyotStop_vec = np.reshape(self.corono.LyotStop2d, (self.corono.nPup**2))
            self.LyotStop_vec_t = [self.LyotStop_vec]

        
        if self.corono.Pupil2dSym == False:
            self.Pupil_vec = np.reshape(self.corono.Pupil2d, (self.corono.nPup**2))
        else:
            Pupil2dquarter = np.zeros_like(self.corono.Pupil2d)
            Pupil2dquarter[self.corono.nPup//2:, self.corono.nPup//2:] = 1.
            self.Pupil_vec = np.reshape(self.corono.Pupil2d*Pupil2dquarter, (self.corono.nPup**2))
         
        self.pup     = (self.Pupil_vec > 0.)
        self.bbb     = np.arange(self.corono.nPup**2)
        self.idx_pup = list(self.bbb[self.pup])
        self.npp     = len(self.idx_pup) 
        
        self.dz2d, self.rad2d = self.corono.generate_area()

        if self.corono.Pupil2dSym == False:        
            self.dz      = np.reshape(self.dz2d, (self.corono.nImg2d**2))
        else:
            Image2dquarter = np.zeros_like(self.dz2d)
            Image2dquarter[self.corono.nImg2d//2:, self.corono.nImg2d//2:] = 1.
            self.dz = np.reshape(self.dz2d*Image2dquarter, (self.corono.nImg2d**2))           
            
        self.aaa     = np.arange(self.corono.nImg2d**2)
        self.idx_dz  = list(self.aaa[self.dz])  
        self.ndz     = len(self.idx_dz)

#        self.LyotStop_vec = np.reshape(self.corono.LyotStop2d, (self.corono.nPup**2))
#        self.lys     = (self.LyotStop_vec > 0.)
#        self.idx_lys = list(self.bbb[self.lys]) 
       
        self.corono_field_t    = None

        self.A       = None
        self.b       = None
        self.c       = None
        
        self.m       = None

        self.TR      = np.sum(self.Pupil_vec)
                
        self.Apod    = np.zeros((self.corono.nPup**2))        

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
        r"""
        Computes the response matrix for the coronagraph with and without 
        the focal plane mask.
        
        Notes
        -----        
        corono_field_re_t_tmp, corono_field_im_t_tmp : array_like, array_like
            Real and imaginary parts of the coronagraphic response matrix
            for all the points in the pupil :math:`P_0` and at all the wavelengths

        corono_field_re_t, corono_field_im_t : array_like, array_like
            Real and imaginary parts of the coronagraphic response matrix
            for all the points in the pupil :math:`P_0` and at all the wavelengths.
            These arrays are sliced from corono_field_re_t_tmp, corono_field_im_t_tmp 
            for the points inside the region of interest in the final image plane.

        corono_field_t : array_like
            Concatenation of the real and imaginary parts of the coronagraphic 
            response matrix for all the points in the pupil :math:`P_0` and 
            at all the wavelengths
                
        """
        corono_field_re_t_tmp = np.zeros((self.ncorono*self.npp, self.corono.nlam, 
                                           self.corono.nImg2d**2))
        corono_field_im_t_tmp = np.zeros((self.ncorono*self.npp, self.corono.nlam, 
                                           self.corono.nImg2d**2))

        Apod2d = np.zeros((self.corono.nPup, self.corono.nPup))

        for k, corono in enumerate(self.corono_t):
            for i, val in enumerate(self.idx_pup):
                (i0,j0) = np.unravel_index(val, (self.corono.nPup, self.corono.nPup))
                Apod2d[i0,j0] = 1            
                corono_field_re_t_tmp[k*self.npp+i], corono_field_im_t_tmp[k*self.npp+i] = \
                corono.compute_corono_field_2d_vec(Apod2d)
                Apod2d[i0,j0] = 0 
                
#        corono_field_re_t = np.reshape(
#                corono_field_re_t_tmp[:,:,self.idx_dz], 
#                (self.npp, self.corono.nlam*self.ndz*self.ncorono))
#
#        corono_field_im_t = np.reshape(
#                corono_field_im_t_tmp[:,:,self.idx_dz], 
#                (self.npp, self.corono.nlam*self.ndz*self.ncorono))


        corono_field_re_t_tmp2 = corono_field_re_t_tmp[:,:,self.idx_dz]

        corono_field_im_t_tmp2 = corono_field_im_t_tmp[:,:,self.idx_dz]
        
        
        corono_field_re_t_tmp3 = np.reshape(corono_field_re_t_tmp2, 
                (self.ncorono, self.npp, self.corono.nlam*self.ndz))

        corono_field_im_t_tmp3 = np.reshape(corono_field_im_t_tmp2, 
                (self.ncorono, self.npp, self.corono.nlam*self.ndz))
        
#        for k in range(self.npp-1):
#            print(np.max(np.abs(corono_field_re_t_tmp3[1, k]-corono_field_re_t_tmp3[0, k])))
#        stop

        corono_field_re_t_tmp4 = np.swapaxes(corono_field_re_t_tmp3, 0,1)
        corono_field_im_t_tmp4 = np.swapaxes(corono_field_im_t_tmp3, 0,1)

#        for k in range(self.npp):
#            print(np.max(np.abs(corono_field_re_t_tmp4[k, 1]-corono_field_re_t_tmp4[k, 0])))
#        stop
        
        corono_field_re_t = np.reshape(
                corono_field_re_t_tmp4, 
                (self.npp, self.corono.nlam*self.ndz*self.ncorono))

        corono_field_im_t = np.reshape(
                corono_field_im_t_tmp4, 
                (self.npp, self.corono.nlam*self.ndz*self.ncorono))        

        self.corono_field_t    = np.concatenate((corono_field_re_t,
                                                 corono_field_im_t), 
                                                 axis=1)       

#%%        
    def solve_model(self):
        """
        Solves the optimization problem model for the model with the selected 
        solver.
               
        Returns
        -------
        Apod
            Apodizer solution :math:`\Phi` for the optimization problem.
        
        """
        self.compute_matrices()

        t0 = time.time()

        if stdgrb and self.solver == 'stdgrb':
            self.print_log('solving problem with stdgrb package')
            Apodtmp, val = stdgrb.lp_solve(self.c, A=(self.A).T, b=self.b, 
                                           ub = np.ones(self.npp+self.neps+self.nbb+self.nvv),                                           
                                           crossover=self.slvCrossover, 
                                           logtoconsole=self.slvLogToConsole, 
                                           method=self.slvMethod)
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
            bds = np.zeros((self.npp+self.neps+self.nbb+self.nvv, 2))
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
                
        if self.problem_name == 'MaxTau':
            str_opt = '_C={cDarkHole:.1f}'
        elif self.problem_name == 'MaxContrastL1' or self.problem_name == 'MaxContrastLinf':
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

        str_FirstDerGlobal = ''
        if self.MinIsland is True:
            str_FirstDerGlobal = '_1stderglo={FirstDerGlobalLim}'

        str_Binarity = ''
        if self.Binarity is True:
            str_Binarity = '_binreg={BinarityReg}'
            
        fname_gen  = '{pupil_name}_{corono_name}_IWA={rho0}' \
        + '_OWA={rho1}_BW={bw:.2f}_nlam={nlam:02d}' \
        + '_2D_nPup={nPup:04d}' + str_cor + '_{problem_name}' \
        + str_opt + str_FirstDerGlobal + str_Binarity + '_{solver}'        
        
        return fname_gen.format(**params)

#%%    
    def compute_matrices(self):
        r"""
        Computes the matrices for the optimization problem.
                                
        """ 
        if self.corono_field_t is None:
            t00 = time.time()
            self.print_log('computing corono response matrix for 2D problem')   
            self.compute_response_matrices()
            t11 = time.time()
            self.print_log('computing time (response matrices): {0:.2f}s\n'.format(t11-t00))

        if self.A is None or self.b is None or self.c is None:
            t00 = time.time()
            self.print_log('computing A, b, and c matrices')
            self.compute_problem_matrices()
            t11 = time.time()                
            self.print_log('computing time (Abc matrices): {0:.2f}s\n'.format(t11-t00))
        else:
            if self.problem_name == 'MaxTau':
                self.print_log('updating A matrix')
                self.update_cDarkHole()
            else:
                self.print_log('updating b matrix')
                self.update_tau()

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
    default_params = get_default_params_MaxTauProblemMatrix()

    def __init__(self, **kwargs):
        """
        Constructor for the Matrix problem with the coronagraph object

        Attributes
        ----------        
        eps : int (default=0)
            size of the variable :math:`\epsilon` for contrast.
            eps is null for MaxTau optimization problem
        
        nvv : int (default=0)
            size of the auxiliary variables for the apodizer derivative 
        
        nbb : int (default=0)
            size of the auxiliary variables for the apodizer binarity
        
        """
        super(MaxTau, self).__init__(**kwargs)
        
        self.neps = 0
        
        self.nvv  = 0
        if self.MinIsland is True:
            self.idx_pup_bis = list(set().union(list(np.asarray(self.idx_pup)-1),list(np.asarray(self.idx_pup)-self.nPup), self.idx_pup))
            self.npp_bis     = len(self.idx_pup_bis)
            self.nvv         = 4*self.npp_bis

        self.nbb  = 0
        if self.Binarity is True:
            self.nbb  = 2*self.npp
        
#%%        
    def compute_problem_matrices(self):
        r"""
        Computes the matrices for the optimization problem that consists in 
        maximizing the amplitude transmission of the apodizer :math:`\Phi` 
        for a set contrast :math:`C` in a given search area in the 
        coronagraphic image. In terms of matrices, the optimization problem 
        writes as
            
        .. math:: \max_{C} c^{T}.x,
            
        under the constraint :math:`A.x \leq b`.
                
        The variable :math:`x` represents the apodizer transmission 
        function :math:`\Phi`. The variables follow the notations of [1]_ and 
        [2]_.       
        
        Notes
        -----------        
        A0 : array_like
            Contrast constraint on the coronagraphic electric field 
            :math:`\Psi_D` that is represented the following equation:
                
            :math:`\Psi_D(\xi,\lambda)-10^{-C/2}\Psi_0(\xi,\lambda) \leq 0`. 
            
            :math:`\xi` and :math:`\lambda` denote the image plane coordinate 
            and wavelength. The term :math:`\Psi_0` represents the 
            coronagraphic electric field in the absence of focal plane mask 
            (FPM).
            
        A1 : array_like
            Contrast constraints on the coronagraphic electric field Psi_D
            that is represented the following equations:
                
            :math:`-\Psi_D(\xi,\lambda)-10^{-C/2}\Psi_0(\xi,\lambda) \leq 0`.
        
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
        .. [1] M. N'Diaye, L. Pueyo, and R. Soummer, "Apodized Pupil Lyot 
            Coronagraphs for Arbitrary Apertures. IV. Reduced Inner Working 
            Angle and Increased Robustness to Low-order Aberrations", ApJ 799, 
            2, 225 (2015).
            
            http://iopscience.iop.org/article/10.1088/0004-637X/799/2/225/meta.
            
        .. [2] M. N'Diaye, R. Soummer, L. Pueyo, A. Carlotti, C. Stark, 
            M. Perrin, "Apodized Pupil Lyot Coronagraphs for Arbitrary 
            Apertures. V. Hybrid Shaped Pupil Designs for Imaging Earth-like 
            planets with Future Space Observatories", ApJ 818, 2, 163 (2016). 
            
            http://iopscience.iop.org/article/10.3847/0004-637X/818/2/163/meta
            
        """                    
        # Compute constant term that includes contrast and normalization
        cst = (10.**(-self.cDarkHole/2.)/np.sqrt(2.))*self.corono.Fmax2d/(self.corono.nImg2d*self.corono.nPup)

        # Compute contrast constraints on the coronagraphic electric field
        A0  =  self.corono_field_t \
        - cst*self.Pupil_vec[self.idx_pup, None]*self.LyotStop_vec[self.idx_pup, None]           
        A1  = -self.corono_field_t \
        - cst*self.Pupil_vec[self.idx_pup, None]*self.LyotStop_vec[self.idx_pup, None]    

        # Yield the A and b matrices for the optimization problem                               
        self.A = np.concatenate((A0,A1), axis=1)
        self.b = np.zeros((2*len(self.corono_field_t.T)))

        # Add apodizer normalization contraints for gurobi solvers
        if (stdgrb and self.solver == 'stdgrb') \
        or (gb and self.solver == 'gurobipy'):
            self.compute_problem_matrices_gurobi()

        # Add apodizer binarity constraints
        if self.Binarity is True:
            self.compute_problem_matrices_Binarity()

        # Add apodizer minimal islands constraints    
        if self.MinIsland is True:
            self.compute_problem_matrices_MinIsland()
            
        # Compute the cost function
        self.c = np.concatenate((-self.Pupil_vec[self.idx_pup]/self.TR, 
                                 self.BinarityReg*np.ones(self.nbb),
                                 np.zeros(self.nvv)), axis=0)
        
        # Return the A, b, and c matrices
        return self.A, self.b, self.c

#%%
    def compute_problem_matrices_gurobi(self):
        r"""
        Computes constraints to range the amplitude transmission of the 
        apodizer :math:`\Phi` between 0 and 1. It writes as
        
        :math:`0 \leq \Phi(r) \leq 1`,
        
        in which :math:`r` represents the radial coordinate of the pupil.
        This computation is used to solve the optimization problem with gurobi
        
        Notes
        -----
        A2, b2 : array_like, array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`- \Phi(r) \leq 0`,
            
            in which :math:`r` represents the radial coordinate of the pupil.
            
        A3, b3 : array_like, array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`\Phi(r) \leq 1`.        
        
        """
        # Compute constraints on the apodizer transmission
        A2  = -np.identity(self.npp)
        A3  =  np.identity(self.npp)
        b2  = np.zeros(self.npp)
        b3  = np.ones(self.npp)

        # Update the A, b, and c matrices
        self.A = np.concatenate((self.A,A2,A3), axis=1)
        self.b = np.concatenate((self.b,b2,b3))        

#%%
    def compute_problem_matrices_Binarity(self):
        r"""
        Computes matrices to add constraints on the apodizer binarity.
        
        Notes
        -----
        A11, b11 : array_like, array_like
            Constraint on the apodization transmission such that
            
            :math:`\Phi(r) - w^{+}(r) + w^{-}(r) \leq 0.5`,
            
            where :math:`w^{+}` and :math:`w^{-}` are two auxiliary variables
            to constrain the absolute value of the derivative to remain 
            positive. 
            
        A12, b12 : array_like, array_like
            Constraint on the apodization transmission such that
            
            :math:`-\Phi(r) + w^{+}(r) - w^{-}(r) \leq -0.5`.
            
        A13 : array_like
            Constraint on the auxiliary variable :math:`w^{+}` to force it 
            to be positive with
            
            :math:`- w^{+}(r) \leq 0`.
        
        A14 : array_like
            Constraint on the auxiliary variable :math:`w^{-}` to force it 
            to be positive with
            
            :math:`- w^{-}(r) \leq 0`.        
        
        """
        # Reshape the matrice A to account for the auxialiary variables
        A00    = np.zeros((self.nbb, np.shape(self.A)[1]))            
        self.A = np.concatenate((self.A, A00))

        # Compute an intermediate matrix for further computation of A matrices
        AI     = np.identity(self.npp)
        
        # Add constraints on the apodizer transmission with auxiliary variables
        A11    = np.concatenate(( AI, -AI,  AI))
        A12    = np.concatenate((-AI,  AI, -AI))
        
        # Compute an intermediate matrix for further computation of A matrices        
        AZ     = np.zeros((self.npp, self.npp))                       

        # Add positivity constraints on the auxiliary variables
        A13    = np.concatenate((AZ, -AI,  AZ))
        A14    = np.concatenate((AZ,  AZ, -AI))
        
        # Compute the b terms corresponding to A11 and A12 matrices        
        b11    = 0.5*np.ones(self.npp)

        # Compute an intermediate matrix for b terms for A13 and A14 
        bZ     = np.zeros((self.npp))
       
        # Update the A, b, and c matrices
        self.A = np.concatenate((self.A, A11,  A12, A13, A14), axis=1)
        self.b = np.concatenate((self.b, b11, -b11,  bZ,  bZ))        
        
        # Warning on the code
        print('Constraint matrix for apodizer binarity is not yet validated!!!')
        
#%%
    def compute_problem_matrices_MinIsland(self):
        r"""
        Computes matrices to add constraints that minimizes the islands in the
        apodizer transmission.
        
        Notes
        -----
        A6, b6 : array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`\frac{d\Phi(r)}{dr} - v^{+}(r) + v^{-}(r) \leq 0`,
            
            where :math:`v^{+}` and :math:`v^{-}` are two auxiliary variables
            to constrain the absolute value of the derivative to remain 
            positive. 
            
        A7, b7 : array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`-\frac{d\Phi(r)}{dr} + v^{+}(r) - v^{-}(r) \leq 0`.
            
        A8 : array_like
            Constraint on the auxiliary variable :math:`v^{+}` to force it 
            to be positive with
            
            :math:`- v^{+}(r) \leq 0`.
        
        A9 : array_like
            Constraint on the auxiliary variable :math:`v^{-}` to force it 
            to be positive with
            
            :math:`- v^{-}(r) \leq 0`. 
            
        A10, b10: array_like, array_like
            Constraint on the apodizer first derivative through the auxiliary 
            variables with
            
            :math:`\int_{P_0} (v^{+}(r) + v^{-}(r))dr \leq \delta`.
        
        """
        # Reshape the matrice A to account for the auxialiary variables
        A00    = np.zeros((self.nvv, np.shape(self.A)[1]))            
        self.A = np.concatenate((self.A, A00))
        
        # Compute the derivative operator of the pupil along x and y axis
        dx0_op = np.zeros((self.nPup**2, self.npp_bis))
        dx1_op = np.zeros((self.nPup**2, self.npp_bis))

        for k, val in enumerate(self.idx_pup_bis):
            if val//self.nPup != self.nPup-1:
                dx0_op[val, k]      = -1
                dx0_op[val+self.nPup, k] = 1
                
        for k, val in enumerate(self.idx_pup_bis):
            if val % self.nPup != self.nPup-1:
                dx1_op[val, k]   = -1
                dx1_op[val+1, k] = 1

        ADx  = dx0_op[self.idx_pup]
        ADy  = dx1_op[self.idx_pup]
                
        # Compute an intermediate matrix for further computation of A matrices
        AI   = np.identity(self.npp_bis)
        
        # Compute intermediate matrices for further computation of A matrices                                
        AZ2  = np.zeros((self.npp_bis, self.npp_bis))

        # Add terms corresponding to the binarity auxiliary variables        
        AZ0 = np.zeros((self.nbb, self.npp_bis))
    
        # Add constraints on the apodizer first derivative with 
        # auxiliary variables        
        A6x  = np.concatenate(( ADx, AZ0, -AI,  AI, AZ2, AZ2))        
        A7x  = np.concatenate((-ADx, AZ0,  AI, -AI, AZ2, AZ2))
        
        A6y  = np.concatenate(( ADy, AZ0, AZ2, AZ2, -AI,  AI))        
        A7y  = np.concatenate((-ADy, AZ0, AZ2, AZ2,  AI, -AI))
        
        # Compute intermediate matrices for further computation of A matrices                                
        AZ1 = np.zeros((self.npp, self.npp_bis))
                
        # Add positivity constraints on the auxiliary variables
        A8x  = np.concatenate((AZ1, AZ0, -AI, AZ2, AZ2, AZ2))
        A9x  = np.concatenate((AZ1, AZ0, AZ2, -AI, AZ2, AZ2))
        
        A8y  = np.concatenate((AZ1, AZ0, AZ2, AZ2, -AI, AZ2))
        A9y  = np.concatenate((AZ1, AZ0, AZ2, AZ2, AZ2, -AI))

        # Compute an intermediate matrix for b terms for A6, A7, A8, and A9 
        bZ   = np.zeros((self.npp_bis))

        # Add boundary constraints on the apodizer first derivative
        A10Z = np.zeros((self.npp))
        A101 = np.ones(self.npp_bis)
        A10  = np.concatenate((A10Z, np.zeros((self.nbb)), A101, A101, A101, A101))

        # Compute b term to bound the integral of the apodizer derivative
        b10  = [self.FirstDerGlobalLim]
                
        # Update the A and b matrices
        self.A = np.concatenate((self.A, A6x, A7x, A6y, A7y, A8x, A9x, A8y, A9y, A10[:, None]), axis=1)
        self.b = np.concatenate((self.b,  bZ,  bZ,  bZ,  bZ,  bZ,  bZ,  bZ,  bZ, b10))       

#%%            
    def update_cDarkHole(self):
        cst = (10.**(-self.cDarkHole/2.)/np.sqrt(2.))*self.corono.Fmax2d/(self.corono.nImg2d*self.corono.nPup)

        A0  =  self.corono_field_t - cst*self.Pupil_vec[self.idx_pup, None]*self.LyotStop_vec[self.idx_pup, None]            
        A1  = -self.corono_field_t - cst*self.Pupil_vec[self.idx_pup, None]*self.LyotStop_vec[self.idx_pup, None]    

        self.A = np.concatenate((A0,A1), axis=1)
        if (stdgrb and self.solver == 'stdgrb') or (gb and self.solver == 'gurobipy'):
            A2  = -np.identity(self.npp)
            A3  =  np.identity(self.npp)
            self.A = np.concatenate((self.A,A2,A3), axis=1)
            

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
        # Compute the length of the A matrix along axis=1  
        nA = np.shape(self.A)[1]
    
        # Create a new model               
        self.m = gb.Model("LP max tau new")
        
        # Create variables
        ApodTmp = self.m.addVars(self.npp + self.nbb + self.nvv, lb=0.0, ub=1.0, 
                                 name="ApodTmp")
        
        # Set objective
        self.m.setObjective(gb.quicksum((self.c[i]*ApodTmp[i] 
                for i in range(self.npp + self.nbb))), gb.GRB.MINIMIZE)
        
        # Add constraint:                
        self.m.addConstrs((gb.quicksum((ApodTmp[i]*self.A[i,j] 
                for i in range(self.npp + self.nbb + self.nvv) if self.A[i,j])) <=  self.b[j] 
                for j in range(nA)), "cpos")
        
        # Update model        
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
        r"""
        Constructor for the MaxContrast problem with the coronagraph object
        
        Attributes
        ----------
        
        eps : int (default=0)
            size of the variable :math:`\epsilon` for contrast.
        
        nvv : int (default=0)
            size of the auxiliary variables for the apodizer derivative. 
        
        nbb : int (default=0)
            size of the auxiliary variables for the apodizer binarity.
        
        """
        super(MaxContrast,self).__init__(**kwargs)

        if self.Lnorm == 'Linf':
            self.neps = 1
        else:
            self.neps = self.ndz

        self.nvv   = 0
        if self.MinIsland is True:
            self.idx_pup_bis = list(set().union(list(np.asarray(self.idx_pup)-1),list(np.asarray(self.idx_pup)-self.nPup), self.idx_pup))
            self.npp_bis     = len(self.idx_pup_bis)
            self.nvv         = 4*self.npp_bis

        self.nbb  = 0
        if self.Binarity is True:
            self.nbb  = 2*self.npp

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

        A0, b01 : array_like, array_like
            Contrast constraint on the coronagraphic electric field :math:`\Psi_D`
            that is represented the following equation:
                
            :math:`\Psi_D(\xi,\lambda)-\epsilon(\xi) \leq 0` 
            if :math:`L_1`-norm constraint,
            
            :math:`\Psi_D(\xi,\lambda)-\epsilon    \leq 0`  
            if :math:`L_\infty`-norm constraint,
            
            :math:`\xi` and :math:`\lambda` denote the image plane coordinate 
            and wavelength.

        A1, b01 : array_like, array_like
            Contrast constraint on the coronagraphic electric field :math:`\Psi_D`
            that is represented the following equation:
                
            :math:`-\Psi_D(\xi, \lambda) - \epsilon(\xi) \leq 0`
            if :math:`L_1`-norm constraint,
            
            :math:`-\Psi_D(\xi,\lambda) - \epsilon    \leq 0`
            if :math:`L_\infty`-norm constraint.
                        
        A6, b6 : array_like, array_like
            Constraint on the variable epsilon that is related to contrast
            and represented by the following equation:
                
            :math:`-\epsilon(\xi) \leq 0` if :math:`L_1`-norm constraint,
            
            :math:`-\epsilon     \leq 0` if :math:`L_\infty`-norm constraint.            
                  
        A7, b7 : array_like
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
        # Compute intermediate variables for electric field constraints 
        if self.Lnorm == 'Linf':
            I1 = np.ones(len(self.corono_field_t.T))
            I1 = I1[None,:]
            I0 = np.ones(self.ndz)
            I0 = I0[None,:]
            self.N0 = np.zeros((1, self.npp))
            Z0 = np.zeros(1)
            c1 = [1]
        else:
            I0 = np.identity(self.ndz)
            I1 = np.hstack([I0 for k in range(len(self.corono_field_t.T)//self.ndz)])            
            self.N0 = np.zeros((self.ndz, self.npp))
            Z0 = np.zeros(self.ndz)
            c1 = np.array(self.rad2d)
        
        # Compute constraints on the coronagraphic electric field
        A0  = np.concatenate(( self.corono_field_t, -I1), axis=0)
        A1  = np.concatenate((-self.corono_field_t, -I1), axis=0)

        # Compute constraint on the auxiliary variable epsilon
        A6  = np.concatenate((np.zeros((self.npp, self.ndz)), -I0), axis=0)

        # Compute constraint on the integral of the apodizer transmission
        A7  = np.concatenate((-self.Pupil_vec[self.idx_pup]/self.TR, Z0))
        
        # Compute b term corresponding to A0 and A1
        b01 = np.zeros((2*len(self.corono_field_t.T)))

        # Compute b terms corresponding to A4 and A5
        b6  = np.zeros(self.ndz)
        b7  = [-self.tau]
        
        #  Yield the A and b matrices for the optimization problem       
        self.A = np.concatenate((A0,A1,A6,A7[:,None]), axis=1)
        self.b = np.concatenate((b01,  b6,b7))

        # Add apodizer normalization contraints for gurobi solvers
        if (stdgrb and self.solver == 'stdgrb') or (gb and self.solver == 'gurobipy'):        
            self.compute_problem_matrices_gurobi()

        # Add apodizer binarity constraints
        if self.Binarity is True:
            self.compute_problem_matrices_Binarity()

        # Add apodizer minimal islands constraints
        if self.MinIsland is True:
            self.compute_problem_matrices_MinIsland()
            
        # Compute the cost function            
        self.c = np.concatenate((np.zeros(self.npp), c1, 
                                 -self.BinarityReg*np.ones(self.nbb), 
                                 np.zeros(self.nvv)), axis=0)
        
        # Returns the A, b, and c matrices        
        return self.A, self.b, self.c

#%%
    def compute_problem_matrices_gurobi(self):
        r"""
        Computes constraints to range the amplitude transmission of the 
        apodizer :math:`\Phi` between 0 and 1. It writes as
        
        :math:`0 \leq \Phi(r) \leq 1`,
        
        in which :math:`r` represents the radial coordinate of the pupil.
        This computation is used to solve the optimization problem with gurobi
        
        Notes
        -----
        A2, b2 : array_like, array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`- \Phi(r) \leq 0`,
            
            in which :math:`r` represents the radial coordinate of the pupil.
            
        A3, b3 : array_like, array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`\Phi(r) \leq 1`.        
        
        """
        # Compute constraints on the apodizer transmission
        A2  = np.concatenate((-np.identity(self.npp), self.N0), axis=0)
        A3  = np.concatenate(( np.identity(self.npp), self.N0), axis=0)
        b2  = np.zeros(self.npp)
        b3  = np.ones(self.npp)
        
        # Update the A, b, and c matrices
        self.A = np.concatenate((self.A,A2,A3), axis=1)
        self.b = np.concatenate((self.b,b2,b3))        

#%%
    def compute_problem_matrices_Binarity(self):
        r"""
        Computes matrices to add constraints on the apodizer binarity.
        
        Notes
        -----
        A11, b11 : array_like, array_like
            Constraint on the apodization transmission such that
            
            :math:`\Phi(r) - w^{+}(r) + w^{-}(r) \leq 0.5`,
            
            where :math:`w^{+}` and :math:`w^{-}` are two auxiliary variables
            to constrain the absolute value of the derivative to remain 
            positive. 
            
        A12, b12 : array_like, array_like
            Constraint on the apodization transmission such that
            
            :math:`-\Phi(r) + w^{+}(r) - w^{-}(r) \leq -0.5`.
            
        A13 : array_like
            Constraint on the auxiliary variable :math:`w^{+}` to force it 
            to be positive with
            
            :math:`- w^{+}(r) \leq 0`.
        
        A14 : array_like
            Constraint on the auxiliary variable :math:`w^{-}` to force it 
            to be positive with
            
            :math:`- w^{-}(r) \leq 0`.        
        
        """
        # Reshape the matrice A to account for the auxialiary variables
        A00    = np.zeros((self.nbb, np.shape(self.A)[1]))            
        self.A = np.concatenate((self.A, A00))

        # Compute an intermediate matrix for further computation of A matrices
        AI     = np.identity(self.npp)

        # Add terms corresponding to the epsilon variable
        AZ0    = np.zeros((self.neps, self.npp))
        
        # Add constraints on the apodizer transmission with auxiliary variables
        A11    = np.concatenate(( AI, AZ0, -AI,  AI))
        A12    = np.concatenate((-AI, AZ0,  AI, -AI))
        
        # Compute an intermediate matrix for further computation of A matrices                        
        AZ     = np.zeros((self.npp, self.npp))                       

        # Add positivity constraints on the auxiliary variables
        A13    = np.concatenate((AZ, AZ0, -AI,  AZ))
        A14    = np.concatenate((AZ, AZ0,  AZ, -AI))
        
        # Compute the b terms corresponding to A11 and A12 matrices                
        b11    = 0.5*np.ones(self.npp)

        # Compute an intermediate matrix for b terms for A13 and A14 
        bZ     = np.zeros((self.npp))
                      
        # Update the A and b matrices              
        self.A = np.concatenate((self.A, A11, A12, A13, A14), axis=1)
        self.b = np.concatenate((self.b, b11,-b11,  bZ,  bZ))
        
        # Warning on the code        
        print('Constraint matrix for apodizer binarity is not yet validated!!!')


#%%
    def compute_problem_matrices_MinIsland(self):
        r"""
        Computes matrices to add constraints that minimizes the islands in the
        apodizer transmission.
        
        Notes
        -----
        A6, b6 : array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`\frac{d\Phi(r)}{dr} - v^{+}(r) + v^{-}(r) \leq 0`,
            
            where :math:`v^{+}` and :math:`v^{-}` are two auxiliary variables
            to constrain the absolute value of the derivative to remain 
            positive. 
            
        A7, b7 : array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`-\frac{d\Phi(r)}{dr} + v^{+}(r) - v^{-}(r) \leq 0`.
            
        A8 : array_like
            Constraint on the auxiliary variable :math:`v^{+}` to force it 
            to be positive with
            
            :math:`- v^{+}(r) \leq 0`.
        
        A9 : array_like
            Constraint on the auxiliary variable :math:`v^{-}` to force it 
            to be positive with
            
            :math:`- v^{-}(r) \leq 0`. 
            
        A10, b10: array_like, array_like
            Constraint on the apodizer first derivative through the auxiliary 
            variables with
            
            :math:`\int_{P_0} (v^{+}(r) + v^{-}(r))dr \leq \delta`.
        
        """       
        # Reshape the matrice A to account for the auxialiary variables
        A00    = np.zeros((self.nvv, np.shape(self.A)[1]))
        self.A = np.concatenate((self.A, A00))

        # Compute the derivative operator of the pupil
        dx0_op = np.zeros((self.nPup**2, self.npp_bis))
        dx1_op = np.zeros((self.nPup**2, self.npp_bis))

        for k, val in enumerate(self.idx_pup_bis):
            if val//self.nPup != self.nPup-1:
                dx0_op[val, k]      = -1
                dx0_op[val+self.nPup, k] = 1
                
        for k, val in enumerate(self.idx_pup_bis):
            if val % self.nPup != self.nPup-1:
                dx1_op[val, k]   = -1
                dx1_op[val+1, k] = 1

        ADx  = dx0_op[self.idx_pup]
        ADy  = dx1_op[self.idx_pup]
               
        # Compute an intermediate matrix for further computation of A matrices
        AI   = np.identity(self.npp_bis)
        
        # Add terms corresponding to eps and the binarity auxiliary variables            
        AZ0 = np.zeros((self.neps+self.nbb, self.npp_bis))

        # Compute intermediate matrices for further computation of A matrices                                
        AZ2 = np.zeros((self.npp_bis, self.npp_bis)) 

        # Add constraints on the apodizer first derivative with 
        # auxiliary variables        
        A6x  = np.concatenate(( ADx, AZ0, -AI,  AI, AZ2, AZ2))        
        A7x  = np.concatenate((-ADx, AZ0,  AI, -AI, AZ2, AZ2))
        
        A6y  = np.concatenate(( ADy, AZ0, AZ2, AZ2, -AI,  AI))        
        A7y  = np.concatenate((-ADy, AZ0, AZ2, AZ2,  AI, -AI))
        
        # Compute intermediate matrices for further computation of A matrices                                
        AZ1 = np.zeros((self.npp, self.npp_bis))
 
        # Add positivity constraints on the auxiliary variables
        A8x  = np.concatenate((AZ1, AZ0, -AI, AZ2, AZ2, AZ2))
        A9x  = np.concatenate((AZ1, AZ0, AZ2, -AI, AZ2, AZ2))
        
        A8y  = np.concatenate((AZ1, AZ0, AZ2, AZ2, -AI, AZ2))
        A9y  = np.concatenate((AZ1, AZ0, AZ2, AZ2, AZ2, -AI))
       
        # Compute an intermediate matrix for b terms for A13 and A14 
        bZ   = np.zeros((self.npp_bis))

        # Add boundary constraints on the apodizer first derivative
        A10Z = np.zeros((self.npp))
        A101 = np.ones(self.npp_bis)
        A10  = np.concatenate((A10Z, np.zeros((self.neps+self.nbb)), A101, A101, A101, A101))

        # Compute b term to bound the integral of the apodizer derivative
        b10  = [self.FirstDerGlobalLim]
                
        # Update the A, b, and c matrices
        self.A = np.concatenate((self.A, A6x, A7x, A6y, A7y, A8x, A9x, A8y, A9y, A10[:, None]), axis=1)
        self.b = np.concatenate((self.b,  bZ,  bZ,  bZ,  bZ,  bZ,  bZ,  bZ,  bZ, b10))       

#%%            
    def update_tau(self):
        b0  = np.zeros((self.corono.nlam*self.ndz*2))
        b1  = np.zeros((self.corono.nlam*self.ndz*2))        
        b4  = np.zeros(self.ndz)
        b5  = [-self.tau]

        self.b = np.concatenate((b0,b1,b4,b5))

        if (stdgrb and self.solver == 'stdgrb') or (gb and self.solver == 'gurobipy'):        
            b2  = np.zeros(self.npp)
            b3  = np.ones(self.npp)
            self.b = np.concatenate((self.b,b2,b3))
            
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
        # Compute the length of the A matrix along axis=1                          
        nA = np.shape(self.A)[1]

        # Create a new model  
        self.m = gb.Model("LP max C new")
        
        # Create variables
        ApodEpsTmp = self.m.addVars(self.npp + self.neps + self.nbb + self.nvv, 
                                    lb=0.0, name="ApodEpsTmp")        

        # Set objective
        self.m.setObjective(gb.quicksum((self.c[i+self.npp]*ApodEpsTmp[i+self.npp] 
                for i in range(self.neps + self.nbb))), gb.GRB.MINIMIZE)

        # Add constraint:
        self.m.addConstrs((gb.quicksum((ApodEpsTmp[i]*self.A[i,j] 
                for i in range(self.npp + self.neps + self.nbb + self.nvv) if self.A[i,j])) <=  self.b[j] 
                for j in np.arange(nA)), "cpos")
        
        # Update model
        self.m.update()

#%%