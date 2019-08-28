# -*- coding: utf-8 -*-
"""
#!/usr/bin/env python3
Created on Fri Mar  9 11:36:39 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

#%% Initialization problem
import numpy as np
import json
import time
import random
from qpsolvers import solve_qp

try:
    import stdgrb
except ImportError:
    stdgrb = False

try:
    import gurobipy as gb
except ImportError:
    gb = False

import scipy.optimize
from . import design, default  
from utils import update_params
from Optim_func import  line_search_armijo,line_search_ratio,fmin_cond,solve_closed_form,cost_function,gradient_function  
#import design, default,utils       

#%%
"""
Problem Matrix class
"""
class ProblemMatrix(object):
    r"""
    Defines the class for Matrix of optimization problem
    """
    default_params = default.get_default_params_1d_ProblemMatrix()
  
    def __init__(self,corono=None,**kwargs):
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
            
        idx_dz : array_like
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
            
        direct_field_re_t_tmp : array_like, array_like
            Real part of the non coronagraphic response matrix for all the 
            points in the pupil :math:`P_0` and at all the wavelengths
            
        corono_field_t : array_like
            Coronagraphic response matrix for all the points in the pupil 
            :math:`P_0` and at all the wavelengths
                                         
        A, b, c : array_like, array_like, array_like
            Matrices for the optimization problem that writes as
            
            .. math:: \max_{\tau} c^{T}.x,    
            under the constraint :math:`A.x \leq b`.

        m : gurobi model
            Gurobi model of the problem to solve when the used solver is 
            gurobipy
        
        TR : float
            Integrated amplitude transmission of the pupil :math:`P_0` 
            with respect to that of the clear pupil
                       
        Apod : array_like
            Apodizer :math:`\Phi` to be generated
        
        """
        self.params  = kwargs
        self.check_params()
        
        if corono is None:
            self.corono = design.APLC1d()
            print('Warning: default coronagraph')
        else:
            self.corono = corono
        
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

        self.direct_field_re_t_tmp = None        
        self.corono_field_t = None
      
        self.A       = None
        self.b       = None
        self.c       = None
        
        self.m       = None
    
        self.TR      = np.sum(2.*np.pi*self.corono.Pupil1d *np.linspace(
                0.5,self.corono.nPup+0.5,num=self.corono.nPup)\
                /(2.*self.corono.nPup)**2)
    
        self.Apod    = np.zeros((self.corono.nPup))   
               
#%%
    def __contains__(self, item):
        r"""
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
    def __getattr__(self, name):
        r"""
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
        r"""
        Checks the params.
        
        Returns
        ----------
        res
            Values for all the keys in the params
        
        """        
        for key in self.default_params:
            if not key in self.params:
                self.params[key] = self.default_params[key]

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
        r"""
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
        direct_field_re_t_tmp, direct_field_im_t_tmp : array_like, array_like
            Real and imaginary part of the non coronagraphic response matrix 
            for all the points in the pupil :math:`P_0` and at all the wavelengths.
            These terms are only computed for 'MaxTau' optimization problem
        
        corono_field_t_tmp : array_like
            Complex amplitude of the coronagraphic 
            response matrix for all the points in the pupil :math:`P_0` and 
            at all the wavelengths
            
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
        if self.problem_name == 'MaxTau' or self.problem_name == 'MaxSNR':
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
            if self.problem_name == 'MaxSNR':         
                self.direct_field_re_t_tmp = np.reshape(self.direct_field_re_t_tmp\
                [:,:,self.idx_dz], (self.npp, self.corono.nlam*self.ndz))
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
        Solves the optimization problem model for the model with the selected 
        solver.
               
        Returns
        -------
        Apod
            Apodizer solution :math:`\Phi` for the optimization problem.
        
        """
        t0 = time.time()

        if self.problem_name=='MaxSNR':
            self.solve_Frank_Wolfe()
        else :
        
            self.compute_matrices()
    
            if self.problem_name =='MaxContrastL2':
                
                self.print_log('solving problem with quadprog')
                
                Valp,Vecp=np.linalg.eigh(self.c)
                
                #Regularization of the Matrix in order to transform it into
                #a positive definite Matrix. It's done by diagonalizing the Matrix and
                #then changing the eigenvalues that are too low.
                
                self.c=np.diag(np.clip(Valp,10**-17,max(Valp)))
                Inv=np.linalg.inv(Vecp.T)
                self.c=np.dot(Inv,self.c)
                self.c=np.dot(self.c,Vecp.T)
                
                
                solve='quadprog'
                x=solve_qp(self.c,np.zeros(len(self.c)),self.A,self.b,None,None,solve)
                self.Apod[self.idx_pup]=x[:self.npp]
            else:
                if stdgrb and self.solver == 'stdgrb':
                    self.print_log('solving problem with stdgrb package')
                    Apodtmp, val = stdgrb.lp_solve(self.c, A=(self.A).T, b=self.b, 
                                                   ub = np.ones(self.npp+self.neps+self.nvv),
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
                    bds = np.zeros((self.npp+self.neps+self.nvv, 2))
                    bds[:,1] = 1.
                    sol=scipy.optimize.linprog(self.c,(self.A).T,self.b,
                                               method='interior-point',
                                               bounds=bds, options={'sparse':False})
                    self.Apod[self.idx_pup]=sol.x[:self.npp]
                
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
        elif self.problem_name == 'MaxContrastL1' or self.problem_name == 'MaxContrastLinf'\
        or self.problem_name == 'MaxContrastL2' or self.problem_name == 'MaxSNR' :
            str_opt = '_tau={tau:.3f}'
        else:
            raise NameError('{0}: Not an existing optimization problem!'.format(self.problem_name))

        str_FirstDer = ''        
        if self.FirstDer is True:
            str_FirstDer = '_1stder={FirstDerLim}'

        str_SecondDer = ''
        if self.SecondDer is True:
            str_SecondDer = '_2ndder={SecondDerLim}'

        str_FirstDerGlobal = ''
        if self.MinIsland is True:
            str_FirstDerGlobal = '_1stderglo={FirstDerGlobalLim}'
                                   
        fname_corono = self.corono.get_filename()  
        
        if self.problem_name == 'MaxSNR':
            str_initialisation = '_initialisation={initialisation}'
            str_nmax = '_nmax={nmax}'
            str_gradmin = '_gradmin={gradmin}'
            
            fname_gen_optim = '{problem_name}' \
            + str_opt + str_FirstDer + str_SecondDer + str_FirstDerGlobal \
            + str_initialisation + str_nmax + str_gradmin
        
        else :
            fname_gen_optim = '{problem_name}' \
            + str_opt + str_FirstDer + str_SecondDer + str_FirstDerGlobal \
            + '_{solver}'

        return fname_corono + '_' + fname_gen_optim.format(**params)

    
#%%    
    def compute_matrices(self):
        r"""
        Computes the matrices for the optimization problem.
                                
        """
#        if self.corono_field_t is None:
        t00 = time.time()            
        self.print_log('computing response matrices for 1D problem')   
        self.compute_response_matrices()
        t11 = time.time()
        self.print_log('computing time (response matrices): {0:.2f}s\n'.format(t11-t00))

#        if self.A is None or self.b is None or self.c is None:
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
        Prints a given log on the console depending on the value of 
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
            print(string)

        
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
    default_params = default.get_default_params_1d_MaxTauProblemMatrix()

    def __init__(self, **kwargs):
        r"""
        Constructor for the MaxTau problem with the coronagraph object
        
        Attributes
        ----------        
        eps : int (default=0)
            size of the variable :math:`\epsilon` for contrast.
            eps is null for MaxTau optimization problem
        
        nvv : int (default=0)
            size of the auxiliary variables for the apodizer derivative 
                
        """
        super(MaxTau, self).__init__(**kwargs)

        self.neps = 0
        self.npp_bis = 0
        self.nvv  = 0
        if self.MinIsland is True:
            self.npp_bis = self.npp-1
            self.nvv     = 2*(self.npp_bis)
            
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
        cst = 10.**(-self.cDarkHole/2.)/np.sqrt(2.)

        # Compute contrast constraints on the coronagraphic electric field
        if self.direct_field_re_t_tmp is None or self.corono_field_t is None:
            self.compute_response_matrices()

        A0tmp  =  self.corono_field_t \
        - cst*self.direct_field_re_t_tmp[:,(self.corono.nlam-1)//2,0, None]
        A1tmp  = -self.corono_field_t \
        - cst*self.direct_field_re_t_tmp[:,(self.corono.nlam-1)//2,0, None]
        
        # Add terms corresponding to the MinIsland auxiliary variables        
        AZ0vv = np.zeros((self.nvv, np.shape(A0tmp)[1]))
        
        A0 = np.concatenate((A0tmp, AZ0vv))
        A1 = np.concatenate((A1tmp, AZ0vv))
        
        # Yield the A and b matrices for the optimization problem                               
        self.A = np.concatenate((A0,A1), axis=1)
        self.b = np.zeros((2*self.corono.nlam*self.ndz*2))
        
        # Add apodizer normalization contraints for gurobi solvers
        if (stdgrb and self.solver == 'stdgrb') \
        or (gb and self.solver == 'gurobipy'):
            self.compute_problem_matrices_gurobi()
        
        # Add apodizer first derivative constraints
        if self.FirstDer is True:
            self.compute_problem_matrices_1stDer()
            
        # Add apodizer second derivative constraints            
        if self.SecondDer is True:
            self.compute_problem_matrices_2ndDer()
                        
        # Add apodizer minimal islands constraints    
        if self.MinIsland is True:
            self.compute_problem_matrices_MinIsland()
        
        # Compute the cost function
        ctmp = np.zeros((self.npp + self.nvv))
        ctmp[:self.npp] = np.asarray(self.idx_pup)+0.5
        self.c = - 2.*np.pi*ctmp/(2.*self.corono.nPup)**2/self.TR
        
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
        A2, b2, b3 : array_like, array_like, array_like 
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`- \Phi(r) \leq 0`,
            
            :math:`\Phi(r) \leq 1`,
            
            in which :math:`r` represents the radial coordinate of the pupil.
        
        """        
        # Compute constraints on the apodizer transmission
        A2tmp  = -np.identity(self.npp)
        
        # Add terms corresponding to the MinIsland auxiliary variables        
        AZ0vv = np.zeros((self.nvv, self.npp))

        A2 = np.concatenate((A2tmp, AZ0vv))

        # Add constraints on the apodizer first derivative with 
        # auxiliary variables                        
        b2  = np.zeros(self.npp)
        b3  = np.ones(self.npp)
        
        # Update the A, b, and c matrices
        if self.A is None:
            self.A = np.concatenate((A2, -A2), axis=1)
        else:
            self.A = np.concatenate((self.A, A2, -A2), axis=1)
        
        if self.b is None:
            self.b = np.concatenate((b2,  b3))
        else:
            self.b = np.concatenate((self.b, b2,  b3))
        
#%%        
    def compute_problem_matrices_1stDer(self):
        r"""
        Computes matrices to add constraints on the apodizer first derivative.

        Notes
        -----
        A4, b4 : array_like, array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`|\frac{d\Phi(r)}{dr}| \leq lim_1`,
            
            in which :math:`r` represents the radial coordinate of the pupil.
                            
        """
        if self.FirstDer is True:
            # Compute the apodizer first derivative constraints
            A4tmp= np.diff(np.identity(self.npp), axis=1)
            
            AZ0vv = np.zeros((self.nvv, self.npp-1))
            
            A4 = np.concatenate((A4tmp, AZ0vv))
            
            b4  = self.FirstDerLim*np.ones(self.npp-1)

            # Update the A, b, and c matrices
            self.A = np.concatenate((self.A, A4, -A4), axis=1)
            self.b = np.concatenate((self.b, b4,  b4))
            
        else:
            print('Warning: Set FirstDer keyword to True to add its constraints!')
                

#%%        
    def compute_problem_matrices_2ndDer(self):
        r"""
        Computes matrices to add constraints on the apodizer second derivative.

        Notes
        -----
        A5, b5 : array_like, array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`|\frac{d^2\Phi(r)}{dr^2}| \leq lim_2`,
            
            in which :math:`r` represents the radial coordinate of the pupil.
                            
        """
        if self.SecondDer is True:
            # Compute the apodizer second derivative constraints        
            A5tmp  = np.diff(np.diff(np.identity(self.npp), axis=1), axis=1)
            b5  = self.SecondDerLim*np.ones(self.npp-2)
            
            AZ0vv = np.zeros((self.nvv, self.npp-2))
            
            A5 = np.concatenate((A5tmp, AZ0vv))
            
            # Update the A, b, and c matrices
            self.A = np.concatenate((self.A, A5, -A5), axis=1)
            self.b = np.concatenate((self.b, b5,  b5)) 
                
        else:
            print('Warning: Set SecondDer keyword to True to add its constraints!')


#%%
    def compute_problem_matrices_MinIsland(self):
        r"""
        Computes matrices to add constraints that minimizes the islands in the
        apodizer transmission.
        
        Notes
        -----
        A10, b10 : array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`\frac{d\Phi(r)}{dr} - v^{+}(r) + v^{-}(r) \leq 0`,
            
            where :math:`v^{+}` and :math:`v^{-}` are two auxiliary variables
            to constrain the absolute value of the derivative to remain 
            positive. 
            
        A11, b11 : array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`-\frac{d\Phi(r)}{dr} + v^{+}(r) - v^{-}(r) \leq 0`.
            
        A12 : array_like
            Constraint on the auxiliary variable :math:`v^{+}` to force it 
            to be positive with
            
            :math:`- v^{+}(r) \leq 0`.
        
        A13 : array_like
            Constraint on the auxiliary variable :math:`v^{-}` to force it 
            to be positive with
            
            :math:`- v^{-}(r) \leq 0`. 
            
        A14, b14: array_like, array_like
            Constraint on the apodizer first derivative through the auxiliary 
            variables with
            
            :math:`\int_{P_0} (v^{+}(r) + v^{-}(r))dr \leq \delta`.
        
        """
        if self.MinIsland is True:
            # Compute the derivative operator of the pupil
            AD  = np.diff(np.identity(self.npp), axis=1)
            
            # Compute an intermediate matrix for further computation of A matrices
            AI  = np.identity(self.npp_bis)
    
            # Add constraints on the apodizer first derivative with 
            # auxiliary variables        
            A10   = np.concatenate(( AD, -AI,  AI))
            A11   = np.concatenate((-AD,  AI, -AI))
    
            # Compute intermediate matrices for further computation of A matrices                                
            AZ1 = np.zeros((self.npp, self.npp_bis))                       
            AZ2 = np.zeros((self.npp_bis, self.npp_bis))
    
            # Add positivity constraints on the auxiliary variables
            A12  = np.concatenate((AZ1, -AI, AZ2))
            A13  = np.concatenate((AZ1, AZ2, -AI))
    
            # Compute an intermediate matrix for b terms for A10, A11, A12, and A13 
            bZ  = np.zeros((4*self.npp_bis))
    
            # Add boundary constraints on the apodizer first derivative
            A14Z = np.zeros((self.npp))
            A141 = np.ones(2*self.npp_bis)
            A14  = np.concatenate((A14Z, A141))
            
            # Compute b term to bound the integral of the apodizer derivative
            b14  = [self.FirstDerGlobalLim]
            
            # Update the A and b matrices            
            self.A = np.concatenate((self.A, A10, A11, A12, A13, A14[:, None]), axis=1)            
            self.b = np.concatenate((self.b,  bZ, b14))
                
        else:
            print('Warning: Set MinIsland keyword to True to add MinIsland constraints!')

#%%
    def compute_gurobi_model(self):
        """
        Generates the gurobi solver model for the MaxTau problem.
        
        Parameters 
        -----------
        Apodtmp : array_like
            Vector of the apodizer :math:`\Phi` in the non zero points of 
            the pupil :math:`P_0`
        
        Notes
        -----------
        m : gurobi model
            Gurobi model of the MaxTau problem to solve
            
        """
#        if self.A is None or self.b is None or self.c is None:
#            self.compute_problem_matrices()
        
        if gb and self.solver == 'gurobipy':
            # Compute the length of the A matrix along axis=1  
            nA = np.shape(self.A)[1]
        
            # Create a new model               
            self.m = gb.Model("LP max tau new")
            
            # Create variables
            ApodTmp = self.m.addVars(self.npp + self.nvv, lb=0.0, ub=1.0,
                                     name="ApodTmp")
            
            # Set objective
            self.m.setObjective(gb.quicksum((self.c[i]*ApodTmp[i] 
                    for i in range(self.npp))), gb.GRB.MINIMIZE)
            
            # Add constraint:                
            self.m.addConstrs((gb.quicksum((ApodTmp[i]*self.A[i,j] 
                    for i in range(self.npp + self.nvv) if self.A[i,j])) <=  self.b[j] 
                    for j in range(nA)), "cpos")
            
            # Update model
            self.m.update()
            
        else:
            print('Warning: Set solver keyword to "gurobipy" to make model!')    

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
    default_params = default.get_default_params_1d_MaxContrastProblemMatrix()
    
    def __init__(self, **kwargs):
        r"""
        Constructor for the MaxContrast problem with the coronagraph object
        
        Attributes
        ----------
        
        eps : int (default=0)
            size of the variable :math:`\epsilon` for contrast.
        
        nvv : int (default=0)
            size of the auxiliary variables for the apodizer derivative. 
                
        """
        super(MaxContrast,self).__init__(**kwargs)

        if self.Lnorm == 'Linf':
            self.neps = 1
        elif self.Lnorm == 'L1':
            self.neps = self.ndz
        elif self.Lnorm == 'L2'  :
            self.neps = 0
        else:
            raise ValueError('{0}: Not an existing L-type norm!'.format(self.problem_name))

        self.npp_bis = 0        
        self.nvv   = 0
        if self.MinIsland is True:
            self.npp_bis = self.npp-1
            self.nvv     = 2*(self.npp-1)

        self.N0 = np.zeros((self.neps, self.npp))            

#%%    
    def compute_problem_matrices(self):
        r"""
        Computes the matrices for the optimization problem that consists in 
        maximizing the contrast in a given search area in the coronagraphic 
        image for a set integrated apodizer transmission :math:`\tau`.
        
        For the :math:`L_1`-norm or :math:`L_\infty`-norm problems, in terms
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
        
        For the :math:`L_2`-norm problem, in terms
        of matrices, the optimization problem writes as
            
        .. math:: \max_{\tau} x^{T}.c.x,
            
        under the constraint :math:`A.x \leq b`.
        Where x is the apodizer transmission function and x^{T}.c.x is the L2-norm
        of the residual in the search area
        The variables follow the notations of [1]_ and [2]_.

        Notes
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
                        
        A20, b20 : array_like, array_like
            Constraint on the variable epsilon that is related to contrast
            and represented by the following equation:
                
            :math:`-\epsilon(\xi) \leq 0` if :math:`L_1`-norm constraint,
            
            :math:`-\epsilon     \leq 0` if :math:`L_\infty`-norm constraint.            
                  
        A21, b21 : array_like
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
        
        if self.Lnorm =='L2':
            if self.corono_field_t is None:
                self.compute_response_matrices()
            self.c=np.dot(self.corono_field_t,self.corono_field_t.T)
            A1=-np.identity(self.npp)
            A2=np.identity(self.npp)
            ctmp = np.zeros(self.npp)
            ctmp[:self.npp] = np.asarray(self.idx_pup)
            ctmp=(-1/sum(ctmp)*(ctmp))
            self.A = np.concatenate((A1,A2,ctmp[None,:]),axis=0)
            b0  = np.zeros(self.npp)
            b1  = np.ones(self.npp)
            b2  = [-self.tau]  
            self.b = np.concatenate((b0,b1,b2))    
        else:
        # Compute intermediate variables for electric field constraints 
            if self.Lnorm == 'Linf':
                I0 = np.ones(self.ndz)
                I0 = I0[None,:]
                I1 = np.ones(self.ndz*self.corono.nlam*2)
                I1 = I1[None,:]
                c1 = [1]
            else:
                I0 = np.identity(self.ndz)
                I1 = np.hstack([I0 for k in range(self.corono.nlam*2)])            
                c1 = 2.*np.pi*np.asarray(self.idx_dz)*(self.corono.Fmax\
                                      /self.corono.nImg)**2
           
            Z0 = np.zeros(self.neps)
    
            # Compute contrast constraints on the coronagraphic electric field
            if self.corono_field_t is None:
                self.compute_response_matrices()
            
            # Compute constraints on the coronagraphic electric field
            A0tmp  = np.concatenate(( self.corono_field_t, -I1), axis=0)
            A1tmp  = np.concatenate((-self.corono_field_t, -I1), axis=0)
    
            # Add terms corresponding to the MinIaland auxiliary variables        
            AZ0vv = np.zeros((self.nvv,  np.shape(A0tmp)[1]))
            
            A0 = np.concatenate((A0tmp, AZ0vv))
            A1 = np.concatenate((A1tmp, AZ0vv))
            
            # Compute constraint on the auxiliary variable epsilon
            A20  = np.concatenate((np.zeros((self.npp, self.ndz)), 
                                   -I0,
                                   np.zeros((self.nvv, self.ndz))), axis=0)
            
            
            # Compute constraint on the integral of the apodizer transmission
            A21  = np.concatenate((-2.*np.pi*(np.asarray(self.idx_pup)+0.5)\
                *self.corono.Pupil1d[self.idx_pup]/(2.*self.corono.nPup)**2/self.TR, 
                                      Z0,
                                      np.zeros((self.nvv))), axis=0)
    
            # Compute b term corresponding to A0 and A1
            b01  = np.zeros((2*self.corono.nlam*self.ndz*2))
            
            # Compute b terms corresponding to A6 and A7
            b20  = np.zeros(self.ndz)
            b21  = [-self.tau]
            
            #  Yield the A and b matrices for the optimization problem       
            self.A = np.concatenate((A0,A1,A20,A21[:,None]), axis=1)
            self.b = np.concatenate((b01  ,b20,b21))
    
            # Add apodizer normalization contraints for gurobi solvers
            if (stdgrb and self.solver == 'stdgrb') \
            or (gb and self.solver == 'gurobipy'):
                self.compute_problem_matrices_gurobi()
    
            # Add apodizer first derivative constraints            
            if self.FirstDer is True:
                self.compute_problem_matrices_1stDer()
    
            # Add apodizer second derivative constraints
            if self.SecondDer is True:
                self.compute_problem_matrices_2ndDer()     
    
            # Add apodizer minimal islands constraints
            if self.MinIsland is True:
                self.compute_problem_matrices_MinIsland()
    
            # Compute the cost function            
            self.c = np.concatenate((np.zeros(self.npp), c1, 
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
        A2, b2, b3 : array_like, array_like, array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`- \Phi(r) \leq 0`,
            
            :math:`\Phi(r) \leq 1`,
            
            in which :math:`r` represents the radial coordinate of the pupil.
        
        """
        # Compute constraints on the apodizer transmission
        A2tmp  = np.concatenate((-np.identity(self.npp), self.N0), axis=0)

        # Add terms corresponding to the MinIsland auxiliary variables        
        AZ0vv = np.zeros((self.nvv, self.npp))

        A2 = np.concatenate((A2tmp, AZ0vv))

        b2  = np.zeros(self.npp)
        b3  = np.ones(self.npp)
        
        # Update the A, b, and c matrices
        if self.A is None:
            self.A = np.concatenate((A2, -A2), axis=1)
        else:
            self.A = np.concatenate((self.A, A2, -A2), axis=1)
        
        if self.b is None:
            self.b = np.concatenate((b2,  b3))
        else:
            self.b = np.concatenate((self.b, b2,  b3))        

#%%        
    def compute_problem_matrices_1stDer(self):
        r"""
        Computes matrices to add constraints on the apodizer first derivative.

        Notes
        -----
        A4, b4 : array_like, array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`|\frac{d\Phi(r)}{dr}| \leq lim_1`,
            
            in which :math:`r` represents the radial coordinate of the pupil.
                            
        """
        if self.FirstDer is True:
            # Compute the apodizer first derivative constraints
            A4tmp  = np.diff(np.identity(self.npp), axis=1)
            A4tmp  = np.concatenate((A4tmp, self.N0[:, :self.npp-1]), axis=0)
    
            AZ0vv = np.zeros((self.nvv, self.npp-1))
            
            A4 = np.concatenate((A4tmp, AZ0vv))
            
            b4  = self.FirstDerLim*np.ones(self.npp-1)
            
            # Update the A, b, and c matrices
            self.A = np.concatenate((self.A, A4, -A4), axis=1)
            self.b = np.concatenate((self.b, b4,  b4))
                
        else:
            print('Warning: Set FirstDer keyword to True to add its constraints!')

#%%        
    def compute_problem_matrices_2ndDer(self):
        r"""
        Computes matrices to add constraints on the apodizer second derivative.

        Notes
        -----
        A5, b5 : array_like, array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`|\frac{d^2\Phi(r)}{dr^2}| \leq lim_2`,
            
            in which :math:`r` represents the radial coordinate of the pupil.
                            
        """
        if self.SecondDer is True:
            # Compute the apodizer second derivative constraints
            A5tmp  = np.diff(np.diff(np.identity(self.npp), axis=1), axis=1)
            A5tmp  = np.concatenate((A5tmp, self.N0[:, :self.npp-2]), axis=0)
            
            AZ0vv = np.zeros((self.nvv, self.npp-2))
            
            A5 = np.concatenate((A5tmp, AZ0vv))        
            
            b5  = self.SecondDerLim*np.ones(self.npp-2)
            
            # Update the A, b, and c matrices
            self.A = np.concatenate((self.A, A5, -A5), axis=1)
            self.b = np.concatenate((self.b, b5,  b5))
                
        else:
            print('Warning: Set SecondDer keyword to True to add its constraints!')

#%%
    def compute_problem_matrices_MinIsland(self):
        r"""
        Computes matrices to add constraints that minimizes the islands in the
        apodizer transmission.
        
        Notes
        -----
        A10, b10 : array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`\frac{d\Phi(r)}{dr} - v^{+}(r) + v^{-}(r) \leq 0`,
            
            where :math:`v^{+}` and :math:`v^{-}` are two auxiliary variables
            to constrain the absolute value of the derivative to remain 
            positive. 
            
        A11, b11 : array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`-\frac{d\Phi(r)}{dr} + v^{+}(r) - v^{-}(r) \leq 0`.
            
        A12 : array_like
            Constraint on the auxiliary variable :math:`v^{+}` to force it 
            to be positive with
            
            :math:`- v^{+}(r) \leq 0`.
        
        A13 : array_like
            Constraint on the auxiliary variable :math:`v^{-}` to force it 
            to be positive with
            
            :math:`- v^{-}(r) \leq 0`. 
            
        A14, b14: array_like, array_like
            Constraint on the apodizer first derivative through the auxiliary 
            variables with
            
            :math:`\int_{P_0} (v^{+}(r) + v^{-}(r))dr \leq \delta`.
        
        """
        if self.MinIsland is True:
            # Compute the derivative operator of the pupil
            AD  = np.diff(np.identity(self.npp), axis=1)
    
            # Compute an intermediate matrix for further computation of A matrices
            AI  = np.identity((self.npp_bis))
            
            # Add terms corresponding to eps and the MinIsland auxiliary variables     
            AZ0 = np.zeros((self.neps, self.npp_bis))
            
            # Add constraints on the apodizer first derivative with 
            # auxiliary variables        
            A10 = np.concatenate(( AD, AZ0, -AI,  AI))
            
            # Compute intermediate matrices for further computation of A matrices                                
            AZ1 = np.zeros((self.npp, self.npp_bis))                       
            AZ2 = np.zeros((self.npp_bis, self.npp_bis))
    
            # Add positivity constraints on the auxiliary variables
            A12  = np.concatenate((AZ1, AZ0, -AI, AZ2))
            A13  = np.concatenate((AZ1, AZ0, AZ2, -AI))
    
            # Compute an intermediate matrix for b terms for A13 and A14 
            bZ  = np.zeros((4*self.npp_bis))
    
            # Add boundary constraints on the apodizer first derivative
            A14Z = np.zeros((self.npp))
            A141 = np.ones(2*self.npp_bis)
            A14  = np.concatenate((A14Z, np.zeros((self.neps)), A141))
    
            # Compute b term to bound the integral of the apodizer derivative
            b14  = [self.FirstDerGlobalLim]
    
            # Update the A, b, and c matrices
            self.A = np.concatenate((self.A, A10, -A10, A12, A13, A14[:, None]), axis=1)
            self.b = np.concatenate((self.b,  bZ,  b14))
                
        else:
            print('Warning: Set MinIsland keyword to True to add its constraints!')
      
#%%
    def compute_gurobi_model(self):
        r"""
        Generates the gurobi solver model for the MaxContrast problem.
        
        Parameters 
        -----------
        ApodEpstmp : array_like
            Vector of the apodizer in the non zero points of the pupil 
            :math:`P_0` and neps points for the coronagraphic image
        
        Notes
        -----------
        m : gurobi model
            Gurobi model of the MaxContrast problem to solve
            
        """        
        if gb and self.solver == 'gurobipy':
            # Compute the length of the A matrix along axis=1                          
            nA = np.shape(self.A)[1]
            
            # Create a new model  
            self.m = gb.Model("LP max C new")
            
            # Create variables
            ApodEpsTmp = self.m.addVars(self.npp + self.neps + self.nvv, 
                                        lb=0.0, name="ApodEpsTmp")        
            # Set objective
            self.m.setObjective(gb.quicksum((self.c[i+self.npp]*ApodEpsTmp[i+self.npp] 
                    for i in range(self.neps))), gb.GRB.MINIMIZE)
            
            # Add constraint:
            self.m.addConstrs((gb.quicksum((ApodEpsTmp[i]*self.A[i,j] 
                    for i in range(self.npp + self.neps + self.nvv) if self.A[i,j])) <=  self.b[j] 
                    for j in np.arange(nA)), "cpos")
            
            # Update model
            self.m.update()
            
        else:
            print('Warning: Set solver keyword to "gurobipy" to make model!')             

#%%
"""
MaxContrast ProblemMatrix subclass
"""
class MaxSNR(ProblemMatrix):
    r"""
    Defines the ProblemMatrix subclass for the optimization problem that 
    maximizes the SNR in a given search area for a given integrated 
    apodizer transmission :math:`\tau`.
    """
    default_params = default.get_default_params_1d_MaxSNR()
    
    def __init__(self, **kwargs):
        r"""
        Constructor for the MaxSNR problem with the coronagraph object
        
        Attributes
        ----------
        
        pb : ProblemMatrix  object (default=None)
            define the optimization problem that has to be solved to initialize 
            the Frank-Wolfe algorithm for :math:`L_p' initializations 
            (:math:`L_1`-norm, `L_2`-norm or :math:`L_\infty`-norm problem)
        

        params :  Dict 
            Dictionnary that contains the parameters of pb, the initialization problem 
        for :math:`L_p' initializations       
        

        nmax  : Int (default=10000)
            Maximum number of iterations forthe Frank-Wolfe algorithm
            
        
        gradmin :  Float (default = 1e-7)
            Minimal gradient value of the cost function for which the Frank-Wolfe algorithm
        consider it has converged
        
        """
        super(MaxSNR,self).__init__(**kwargs)
        if self.initialisation == 'Linf':
            params=kwargs.copy()
            params['Lnorm'] = params.pop('initialisation')
            del params['nmax']
            del params['gradmin']
            params['problem_name']='MaxContrastLinf'
            self.pb=MaxContrast(**params)
            
        elif self.initialisation == 'L1':
            params=kwargs.copy()
            params['Lnorm'] = params.pop('initialisation')
            del params['nmax']
            del params['gradmin']
            params['problem_name']='MaxContrastL1'
            self.pb=MaxContrast(**params)        

        elif self.initialisation == 'L2'  :
            params=kwargs.copy()
            params['Lnorm'] = params.pop('initialisation')
            del params['nmax']
            del params['gradmin']
            params['problem_name']='MaxContrastL2'
            self.pb=MaxContrast(**params)        
        
        elif self.initialisation == 'Unif'  :
            pass
        
        elif self.initialisation == 'Random'  :
            pass
        
        else:
            raise ValueError('{0}: Not an existing initialization!'.format(self.initialisation))
            
        if self.nmax == None:
            self.nmax=10000
        if self.gradmin == None:
            self.gradmin=1e-7

#%%
        
    def solve_Frank_Wolfe(self):
        r"""
        Computes the matrices for the optimization problem that consists in 
        maximizing the SNR in a given search area in the coronagraphic 
        image for a set integrated apodizer transmission :math:`\tau` via a
        condionnal gradient descent method (Frank-Wolfe algorithm). The algorithm
        can be initialized  with the solution of the :math:`L_1`-norm or 
        :math:`L_\infty`-norm or :math:`L_2` problems. It can also be initialized with 
        a random apodizer or a uniform apodizer.
        
        Notes
        initialisation : string
            Type of initialization for the optimization problem
            
        x0 : array_like
        Contains the Apodizer solution of the optimisation problem specified
        
        c_transmission :array_like
        Vector whose scalar product with an apodiser returns the transmission
        :math: \tau of this apodizer
            

        -----------

        
        """
        t0=time.time()
        x0=np.zeros_like(self.Apod)
        
        # Computes the vector to estimate the transmission \tau of an apodizer

        c_transmission = np.zeros(self.npp)
        c_transmission[:self.npp] = np.asarray(self.idx_pup)
        c_transmission=(1/sum(c_transmission)*(c_transmission)) 
        
        
        # Computes the apodizer that initializes the Frank-Wolfe algorithm 

        self.compute_response_matrices()
        
        
        #If the initializer apodizer is uniform
        if self.initialisation =='Unif':
           x0[self.idx_pup]=(self.tau)*(np.ones(self.npp))
           x0=x0[self.idx_pup]
        
        #If the initializer apodizer is randomly choosen among all the acceptable apodizers

        elif self.initialisation =='Random':
            l=[i for i in range (len(self.idx_pup))]
            l=np.random.permutation(l)
            x0=x0[self.idx_pup]
            x1=np.ones_like(x0)
            k=-1
            while np.dot(c_transmission,x1)>self.tau and k<len(x0)-1:
                k+=1
                x0[l[k]]=random.random()
                x1[l[k]]=x0[l[k]]
            if np.dot(c_transmission,x1)<self.tau:
                x0=x1
                x0[l[k]]=0
                x0[l[k]]=(self.tau-np.dot(c_transmission,x0))/c_transmission[l[k]]
        
        
        #If the initializer apodizer is the solution of one of the math: L_p problem

        else:
            x0=self.pb.solve_model()
            x0=x0[self.idx_pup]                    
            
        t1=time.time()

        print('initialization problem solving time: {0:.2f}s\n'.format(t1-t0))
        
        
        # Compute the matrices required to calculate the cost function 

#
        Ke=np.dot(self.corono_field_t,self.corono_field_t.T)
    
        Kp=np.dot(self.direct_field_re_t_tmp,self.direct_field_re_t_tmp.T)
        
        psi_star = self.corono_field_t
        psi_planet = self.direct_field_re_t_tmp

    
        # Define the cost function, the gradient of the cost function and the solver
        #of the linear problem that has to be solved at each step of the Frank-Wolfe
        #algorithm


#            grad1=lambda x:(2*np.dot(Ke,x)*np.dot(np.dot(x,Kp),x)-2*np.dot(Kp,x)*np.dot(np.dot(x,Ke),x))\
#            /(np.dot(np.dot(x,Kp),x))**2       
#            fonc1 =lambda x:np.dot(np.dot(x,Ke),x)/np.dot(np.dot(x,Kp),x)

        grad1=lambda x:gradient_function(x,psi_star,psi_planet)
        
        fonc1 =lambda x:cost_function(x,psi_star,psi_planet)
        
        solve_C1=lambda x,g:solve_closed_form(g,c_transmission,self.tau)
        
            
        # gather the parameters of the fmin_cond function in a  dictionnary

        params = dict()
        params['nbitermax'] = self.nmax
        params['stopvarj'] = self.gradmin
        params['verbose'] = False
        params['log'] = True    
        
        # Apply the Frank-Wolfe algorithm

        x, val, log = fmin_cond(fonc1, grad1, solve_C1, x0, psi_star, psi_planet,linesearch=1, **params)
        
        
        a=np.zeros_like(self.Apod)
        a[self.idx_pup]=x[:self.npp]
        self.Apod=a
        
        return 