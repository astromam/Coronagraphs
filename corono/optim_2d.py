#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#!/usr/bin/env python3
Created on Fri Mar  9 11:36:39 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

#%% Initialization problem
#from memory_profiler import profile
import numpy as np
import json
import time
import gc
import psutil
import os
from memory_profiler import profile

try:
    import stdgrb
except ImportError:
    stdgrb = False
    print('stdgrb is False')

try:
    import gurobipy as gb
except ImportError:
    gb = False
    print('gb is False')    

import scipy
import scipy.optimize
from .utils import update_params, uniform_disk        
from . import design, default

def MemUse():
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0]*10**-9  #RSS (resident set size) in GB
	print('memory use: {0:.3f} GB'.format(memoryUse))

def describe_array(array):
    print(type(array))
    print(array.dtype)
    print(array.shape)

#%%
"""
Problem Matrix class
"""
class ProblemMatrix(object):
    r"""
    Defines the class for Matrix of optimization problem
    """
    default_params = default.get_default_params_2d_ProblemMatrix()

    @profile
    def __init__(self,corono=None, **kwargs):
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
        
        if corono is None:
            self.corono_t = [design.APLC2d()]
            print('Warning: default coronagraph')
        else:
            if isinstance(corono, list) == True:
                self.corono_t = corono
            else:
                self.corono_t = [corono]
        
        if self.LSRobustness == False:
            self.corono_t = [self.corono_t[0]]
        
        self.corono   = self.corono_t[0]
        self.ncorono  = len(self.corono_t)
        self.LyotStop_vec_t = np.zeros((self.ncorono, (self.corono.nPup**2)))
        for i in range(self.ncorono):
            self.LyotStop_vec_t[i] = np.reshape(self.corono_t[i].LyotStop2d, (self.corono.nPup**2))                
        
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
       
#        self.corono_field_t    = None

        self.A       = None
        self.b       = None
        self.c       = None
        
        self.m       = None

        self.TR      = np.sum(self.Pupil_vec)
                
        self.Apod    = np.zeros((self.corono.nPup**2))

#%%
        Mask2d = uniform_disk(self.corono.nFPM, self.corono.nFPM/2., CtrBtwnPix=self.CtrBtwnPix)

        if self.corono.Pupil2dSym == False:
            self.Mask1d = np.reshape(Mask2d, (self.corono.nFPM**2))
        else:
            Mask2dquarter = np.zeros_like(Mask2d)
            Mask2dquarter[self.corono.nFPM//2:, self.corono.nFPM//2:] = 1.
            self.Mask1d = np.reshape(Mask2d*Mask2dquarter, (self.corono.nFPM**2))
        
        self.msk     = (self.Mask1d > 0.)
        self.ccc     = np.arange(self.corono.nFPM**2)
        self.idx_msk = list(self.ccc[self.msk])
        self.nmm     = len(self.idx_msk)

        self.MM       = self.Mask1d[self.idx_msk]

        if self.CtrBtwnPix is True:
            val = 1/2 
        x2d,y2d = np.meshgrid(np.arange(self.corono.nPup)-self.corono.nPup//2+val, 
                              np.arange(self.corono.nPup)-self.corono.nPup//2+val)
        x2d /= self.corono.nPup
        y2d /= self.corono.nPup 
        
        if self.CtrBtwnPix is True:
            val = 1/2   
        p2d,q2d  = np.meshgrid(np.arange(self.corono.nFPM)-self.corono.nFPM//2+val, 
                               np.arange(self.corono.nFPM)-self.corono.nFPM//2+val)
        p2d *= 2*self.rMask/self.corono.nFPM
        q2d *= 2*self.rMask/self.corono.nFPM 
        
        if self.CtrBtwnPix is True:
            val = 1/2   
        u2d,v2d  = np.meshgrid(np.arange(self.corono.nImg2d)-self.corono.nImg2d//2+val, 
                               np.arange(self.corono.nImg2d)-self.corono.nImg2d//2+val)
        u2d *= self.corono.Fmax2d/self.corono.nImg2d
        v2d *= self.corono.Fmax2d/self.corono.nImg2d 

        x1d_tmp = x2d.ravel()
        y1d_tmp = y2d.ravel()
        
        p1d_tmp = p2d.ravel()
        q1d_tmp = q2d.ravel()
        
        u1d_tmp = u2d.ravel()
        v1d_tmp = v2d.ravel()
        
        self.x1d = x1d_tmp[self.idx_pup]
        self.y1d = y1d_tmp[self.idx_pup]
        
        self.p1d = p1d_tmp[self.idx_msk]
        self.q1d = q1d_tmp[self.idx_msk]
        
        self.u1d = u1d_tmp[self.idx_dz]
        self.v1d = v1d_tmp[self.idx_dz]
 
        LyotStop1d = self.corono.LyotStop2d.ravel()
        self.LL = LyotStop1d[self.idx_pup]
       

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
    @profile
    def compute_response_matrices(self, corono=None):
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
        if corono is None:
            pass
        else:
            corono_field_re_t_tmp = np.empty((self.npp, self.corono.nlam, 
                                                   self.corono.nImg2d**2))

            Apod2d = np.zeros((self.corono.nPup, self.corono.nPup))


            if self.ImPart is True:             
                corono_field_im_t_tmp = np.empty((self.npp, self.corono.nlam, 
                                                   self.corono.nImg2d**2))
    
                for i, val in enumerate(self.idx_pup):
                    (i0,j0) = np.unravel_index(val, (self.corono.nPup, self.corono.nPup))
                    Apod2d[i0,j0] = 1            
                    corono_field_re_t_tmp[i], corono_field_im_t_tmp[i] = \
                    corono.compute_corono_field_2d_vec(Apod2d)
                    Apod2d[i0,j0] = 0 
                    
                corono_field_re_t = np.reshape(
                        corono_field_re_t_tmp[:,:, self.idx_dz], 
                        (self.npp, self.corono.nlam*self.ndz))
                
                corono_field_re_t_tmp = None
                del corono_field_re_t_tmp
            
                corono_field_im_t = np.reshape(
                    corono_field_im_t_tmp[:,:, self.idx_dz], 
                    (self.npp, self.corono.nlam*self.ndz))
                corono_field_re_t = np.concatenate((corono_field_re_t, 
                                                    corono_field_im_t), axis=1)
                
                corono_field_im_t = None
                del corono_field_im_t
            
                corono_field_im_t_tmp = None
                del corono_field_im_t_tmp

            else:    
                for i, val in enumerate(self.idx_pup):
                    (i0,j0) = np.unravel_index(val, (self.corono.nPup, self.corono.nPup))
                    Apod2d[i0,j0] = 1            
                    corono_field_re_t_tmp[i] = \
                    corono.compute_corono_field_2d_real_vec(Apod2d)
                    Apod2d[i0,j0] = 0 
                    
                corono_field_re_t = np.reshape(
                        corono_field_re_t_tmp[:,:, self.idx_dz], 
                        (self.npp, self.corono.nlam*self.ndz))
                
                corono_field_re_t_tmp = None
                del corono_field_re_t_tmp

            return corono_field_re_t

    #%%
    
    def Q_direct(self,lam):   
        if self.ImPart is True:
            LFCD = np.exp(-2j*np.pi*(self.corono.lam0/lam)*(self.x1d[:, None]*self.u1d[None, :] + self.y1d[:, None]*self.v1d[None, :]))
        else:
            LFCD = np.cos(-2*np.pi*(self.corono.lam0/lam)*(self.x1d[:, None]*self.u1d[None, :] + self.y1d[:, None]*self.v1d[None, :]))
        
        LFCD *= (self.corono.lam0/lam)*self.corono.Fmax2d/(self.corono.nPup*self.corono.nImg2d)
        LFCD *= self.LL[:,None]
        return LFCD
    
    #%%
    #@profile
    def Q_corono(self, lam):
        if self.ImPart is True:
            FAB = np.exp(-2j*np.pi*(self.corono.lam0/lam)*(self.x1d[:, None]*self.p1d[None, :] + self.y1d[:, None]*self.q1d[None, :]))
        else:
            FAB = np.cos(-2*np.pi*(self.corono.lam0/lam)*(self.x1d[:, None]*self.p1d[None, :] + self.y1d[:, None]*self.q1d[None, :]))
        
        FAB *= (self.corono.lam0/lam)*2*self.corono.rMask/(self.corono.nPup*self.corono.nFPM)
        
        if self.ImPart is True:
            MFBC = FAB.conjugate().T    
        else:
            MFBC = FAB.T
        MFBC *= self.MM[:,None]
        
        LFCD = self.Q_direct(lam)
        
        Q  = - (FAB.dot(MFBC)).dot(LFCD)
        Q += LFCD
        
        return Q


#%%
    @profile
    def compute_response_matrices_new(self, corono=None):
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
        if corono is None:
            pass
        else:
            corono_field_re_t = np.empty((self.npp, self.nlam, self.ndz))
            for i, lam in enumerate(self.corono.lam_t):
                corono_field_re_t[:, i] = self.Q_corono(lam)
                
            return np.reshape(corono_field_re_t, (self.npp, self.nlam*self.ndz))



#%%        
    @profile
    def solve_model(self):
        """
        Solves the optimization problem model for the model with the selected 
        solver.
               
        Returns
        -------
        Apod
            Apodizer solution :math:`\Phi` for the optimization problem.
        
        """
        #print('start - compute matrices')
        #MemUse()
        self.compute_matrices()

        #print('start - solve model')
        #MemUse()
        t0 = time.time()

        if stdgrb and self.solver == 'stdgrb':
            self.print_log('solving problem with stdgrb package')
            if self.slvSparse == 0:
                Apodtmp, val = stdgrb.lp_solve(self.c, A=(self.A).T, b=self.b, 
                                               ub = np.ones(self.npp+self.neps+self.nvv),                                           
                                               crossover=self.slvCrossover, 
                                               logtoconsole=self.slvLogToConsole, 
                                               method=self.slvMethod)
            else:
            # convert A matrix into sparse matrix
                print('Conversion of A into sparse matrix')
                As=scipy.sparse.csr_matrix((self.A).T)
                Apodtmp, val = stdgrb.lp_solve_sparse(self.c, A=As, b=self.b, 
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
                
                
                
                print('preparing to save optimization problem')
                print('ok')
                
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
        #print('end')
        #MemUse()
        
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

        str_FirstDerGlobal = ''
        if self.MinIsland is True:
            str_FirstDerGlobal = '_1stderglo={FirstDerGlobalLim}'
            
        str_LSRobustness = ''
        if self.LSRobustness is True:
            str_LSRobustness = '_LSRobustness=1'

        fname_corono = self.corono.get_filename()             
            
        fname_gen_optim  = '{problem_name}' \
        + str_opt + str_FirstDerGlobal + str_LSRobustness \
        + '_{solver}'        
        
        return '{pupil_name}_'.format(**params) + fname_corono + fname_gen_optim.format(**params)

#%%    
    @profile
    def compute_matrices(self):
        r"""
        Computes the matrices for the optimization problem.
                                
        """ 
#        if self.A is None or self.b is None or self.c is None:
        t00 = time.time()
        self.print_log('computing A, b, and c matrices')
        self.compute_problem_matrices()
        t11 = time.time()                
        self.print_log('computing time (Abc matrices): {0:.2f}s\n'.format(t11-t00))
#        else:
#            if self.problem_name == 'MaxTau':
#                self.print_log('updating A matrix')
#                self.update_cDarkHole()
#            else:
#                self.print_log('updating b matrix')
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
    default_params = default.get_default_params_2d_MaxTauProblemMatrix()

    @profile
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
                
        """
        super(MaxTau, self).__init__(**kwargs)
        
        self.neps = 0
        self.npp_bis = 0
        self.idx_pup_bis = [0]        
        self.nvv  = 0
        if self.MinIsland is True:
            self.idx_pup_bis = list(set().union(list(np.asarray(self.idx_pup)-1),
                                        list(np.asarray(self.idx_pup)-self.corono.nPup), 
                                        self.idx_pup))
            self.npp_bis     = len(self.idx_pup_bis)
            self.nvv         = 4*self.npp_bis

        
#%%        
#    @profile
#    def compute_problem_matrices(self):
#        r"""
#        Computes the matrices for the optimization problem that consists in 
#        maximizing the amplitude transmission of the apodizer :math:`\Phi` 
#        for a set contrast :math:`C` in a given search area in the 
#        coronagraphic image. In terms of matrices, the optimization problem 
#        writes as
#            
#        .. math:: \max_{C} c^{T}.x,
#            
#        under the constraint :math:`A.x \leq b`.
#                
#        The variable :math:`x` represents the apodizer transmission 
#        function :math:`\Phi`. The variables follow the notations of [1]_ and 
#        [2]_.       
#        
#        Notes
#        -----------        
#        A0 : array_like
#            Contrast constraint on the coronagraphic electric field 
#            :math:`\Psi_D` that is represented the following equation:
#                
#            :math:`\Psi_D(\xi,\lambda)-10^{-C/2}\Psi_0(\xi,\lambda) \leq 0`. 
#            
#            :math:`\xi` and :math:`\lambda` denote the image plane coordinate 
#            and wavelength. The term :math:`\Psi_0` represents the 
#            coronagraphic electric field in the absence of focal plane mask 
#            (FPM).
#            
#        A1 : array_like
#            Contrast constraints on the coronagraphic electric field Psi_D
#            that is represented the following equations:
#                
#            :math:`-\Psi_D(\xi,\lambda)-10^{-C/2}\Psi_0(\xi,\lambda) \leq 0`.
#        
#        Returns 
#        ----------
#        A, b, c: array_like, array_like, array_like
#            The matrices for the optimization problem.
#            A and b are concatenations of the matrices for the constraints that 
#            are described above.
#            
#            The cost function c to maximize is the transmission of the apodizer 
#            inside the pupil :math:`P_0`.
#            
#            .. math:: \max_{C}[\int_{P_0} \Phi(r)dr].
#            
#        References
#        ----------
#        .. [1] M. N'Diaye, L. Pueyo, and R. Soummer, "Apodized Pupil Lyot 
#            Coronagraphs for Arbitrary Apertures. IV. Reduced Inner Working 
#            Angle and Increased Robustness to Low-order Aberrations", ApJ 799, 
#            2, 225 (2015).
#            
#            http://iopscience.iop.org/article/10.1088/0004-637X/799/2/225/meta.
#            
#        .. [2] M. N'Diaye, R. Soummer, L. Pueyo, A. Carlotti, C. Stark, 
#            M. Perrin, "Apodized Pupil Lyot Coronagraphs for Arbitrary 
#            Apertures. V. Hybrid Shaped Pupil Designs for Imaging Earth-like 
#            planets with Future Space Observatories", ApJ 818, 2, 163 (2016). 
#            
#            http://iopscience.iop.org/article/10.3847/0004-637X/818/2/163/meta
#            
#        """                    
#        # Compute constant term that includes contrast and normalization
#        cst = (10.**(-self.cDarkHole/2.)/np.sqrt(2.))*self.corono.Fmax2d/(self.corono.nImg2d*self.corono.nPup)
#
#        # Compute contrast constraints on the coronagraphic electric field
#        for k in range(self.ncorono):
#            
#            # Compute coronagraph response matrix
#            t00 = time.time()
#            self.print_log('computing corono response matrix for 2D problem')             
#            corono_field_t = self.compute_response_matrices(self.corono_t[k])
#            t11 = time.time()
#            self.print_log('computing time (response matrices): {0:.2f}s\n'.format(t11-t00))
#            
#            LyotStop_vec   = self.LyotStop_vec_t[k]
#            
#            A0tmp  =  corono_field_t \
#            - cst*self.Pupil_vec[self.idx_pup, None]*LyotStop_vec[self.idx_pup, None]           
#            A1tmp  = -corono_field_t \
#            - cst*self.Pupil_vec[self.idx_pup, None]*LyotStop_vec[self.idx_pup, None]
#            
#            # Add terms corresponding to the MinIsland auxiliary variables        
#            AZ0vv = np.zeros((self.nvv, np.shape(A0tmp)[1]))
#
#            print(A0tmp.shape)
#            print(A1tmp.shape)
#            print(AZ0vv.shape)
#            
#            A0 = np.concatenate((A0tmp, AZ0vv))
#            A1 = np.concatenate((A1tmp, AZ0vv))            
#            
#            if k == 0:
#                self.A = np.concatenate((A0,A1), axis=1)
#            else:
#                self.A = np.concatenate((self.A, A0, A1), axis=1)
#            
#            print(A0.shape)
#            print(A1.shape)
#            print(self.A.shape)
#            
#            A0 = None
#            A1 = None
#            del A0
#            del A1
#            gc.collect()
#         
#        # Yield the A and b matrices for the optimization problem                               
#        self.b = np.zeros((len(self.A.T)))
#
#        # Add apodizer normalization contraints for gurobi solvers
#        if (stdgrb and self.solver == 'stdgrb') \
#        or (gb and self.solver == 'gurobipy'):
#            self.compute_problem_matrices_gurobi()
#
#        # Add apodizer minimal islands constraints    
#        if self.MinIsland is True:
#            self.compute_problem_matrices_MinIsland()
#            
#        # Compute the cost function
#        self.c = np.concatenate((-self.Pupil_vec[self.idx_pup]/self.TR, 
#                                 np.zeros(self.nvv)), axis=0)
#        
#        print(self.A.shape)
#        
#        # Return the A, b, and c matrices
#        return self.A, self.b, self.c
#%%
    @profile
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
        # Compute intermediate variables for electric field constraints 
        nI1 = 1 
        if self.ImPart is True:
            nI1 = 2

        # Compute constant term that includes contrast and normalization
        cst = (10.**(-self.cDarkHole/2.)/np.sqrt(2.))*self.corono.Fmax2d/(self.corono.nImg2d*self.corono.nPup)

        self.A = np.zeros((self.npp+self.nvv, self.ncorono*2*nI1*self.nlam*self.ndz))

        # Compute contrast constraints on the coronagraphic electric field
        for k in range(self.ncorono):

            LyotStop_vec   = self.LyotStop_vec_t[k]
            
            t0 = time.time()                                    
            self.A[:self.npp, 2*k*nI1*self.nlam*self.ndz:(2*k+1)*nI1*self.nlam*self.ndz] = \
            self.compute_response_matrices(self.corono_t[k])
#            self.A[:self.npp, 2*k*nI1*self.nlam*self.ndz:(2*k+1)*nI1*self.nlam*self.ndz] = \
#            self.compute_response_matrices_new(self.corono_t[k])
            t1 = time.time()
            print('response matrices: {0}'.format(self.A[:self.npp, 2*k*self.nlam*self.ndz:(2*k+1)*self.nlam*self.ndz].shape))
            print('compute response matrices: {0:.5f}s'.format(t1-t0))
            
            self.A[:self.npp, (2*k+1)*nI1*self.nlam*self.ndz:(2*k+2)*nI1*self.nlam*self.ndz] = \
            -self.A[:self.npp, 2*k*nI1*self.nlam*self.ndz:(2*k+1)*nI1*self.nlam*self.ndz] 
 
            self.A[:self.npp, 2*k*nI1*self.nlam*self.ndz:2*(k+1)*nI1*self.nlam*self.ndz] -= \
            cst*self.Pupil_vec[self.idx_pup, None]*LyotStop_vec[self.idx_pup, None]
                           
        # Yield the A and b matrices for the optimization problem                               
        self.b = np.zeros((len(self.A.T)))

        # Add apodizer normalization contraints for gurobi solvers
        if (stdgrb and self.solver == 'stdgrb') \
        or (gb and self.solver == 'gurobipy'):
            self.compute_problem_matrices_gurobi()

        # Add apodizer minimal islands constraints    
        if self.MinIsland is True:
            self.compute_problem_matrices_MinIsland()
            
        # Compute the cost function
        self.c = np.concatenate((-self.Pupil_vec[self.idx_pup]/self.TR, 
                                 np.zeros(self.nvv)), axis=0)
        
#        print(self.A.shape)
        
        # Return the A, b, and c matrices
        return self.A, self.b, self.c

#%%
    @profile
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
            
        b3 : array_like
            Constraint on the amplitude transmission of the apodization 
            :math:`\Phi`
            
            :math:`\Phi(r) \leq 1`.        
        
        """
        #print('start - compute gurobi matrices')
        #MemUse()
        # Compute constraints on the apodizer transmission
        A2tmp  = -np.identity(self.npp)
        #A2tmp = -scipy.sparse.identity(self.npp)
        
        
        # Add terms corresponding to the MinIsland auxiliary variables        
        AZ0vv = np.zeros((self.nvv, self.npp))
        #AZ0vv = scipy.sparse.csr_matrix((self.nvv, self.npp))
        
        A2 = np.concatenate((A2tmp, AZ0vv))
        #A2 = scipy.sparse.vstack((A2tmp,AZ0vv))
        #A2 = scipy.sparse.csr_matrix(A2)
        
        # Add constraints on the apodizer first derivative with 
        # auxiliary variables                        
        b2  = np.zeros(self.npp)
        b3  = np.ones(self.npp)

        # Update the A, b, and c matrices
        if self.A is None:
            self.A = np.concatenate((A2, -A2), axis=1)
            #self.A = scipy.sparse.hstack((self.A,A2,-A2))
            #self.A = scipy.sparse.csr_matrix(self.A)            
        else:
            	self.A = np.concatenate((self.A,A2,-A2), axis = 1)
            #self.A = scipy.sparse.bsr_matrix(self.A)
            	#self.A = scipy.sparse.hstack((self.A,A2,-A2))
            	#self.A = scipy.sparse.csr_matrix(self.A)
        
        A2 = None
        del A2
        gc.collect()
        
        if self.b is None:
            self.b = np.concatenate((b2,  b3))
        else:
            self.b = np.concatenate((self.b, b2,  b3)) 
        
        b2 = None
        b3 = None
        del b2
        del b3
        gc.collect()
        
#%%
    @profile
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
        # Reshape the matrice A to account for the auxialiary variables
#        A00    = np.zeros((self.nvv, np.shape(self.A)[1]))            
#        self.A = np.concatenate((self.A, A00))
        
            # Compute the derivative operator of the pupil along x and y axis
            dx0_op = np.zeros((self.corono.nPup**2, self.npp_bis))
            dx1_op = np.zeros((self.corono.nPup**2, self.npp_bis))
    
            for k, val in enumerate(self.idx_pup_bis):
                if val//self.corono.nPup != self.corono.nPup-1:
                    dx0_op[val, k]      = -1
                    dx0_op[val+self.corono.nPup, k] = 1
                    
            for k, val in enumerate(self.idx_pup_bis):
                if val % self.corono.nPup != self.corono.nPup-1:
                    dx1_op[val, k]   = -1
                    dx1_op[val+1, k] = 1
    
            ADx  = dx0_op[self.idx_pup]
            ADy  = dx1_op[self.idx_pup]
                    
            # Compute an intermediate matrix for further computation of A matrices
            AI   = np.identity(self.npp_bis)
            
            # Compute intermediate matrices for further computation of A matrices                                
            AZ2  = np.zeros((self.npp_bis, self.npp_bis))
        
            # Add constraints on the apodizer first derivative with 
            # auxiliary variables        
            A10x  = np.concatenate(( ADx, -AI,  AI, AZ2, AZ2))        
            
            A10y  = np.concatenate(( ADy, AZ2, AZ2, -AI,  AI))        
            
            # Compute intermediate matrices for further computation of A matrices                                
            AZ1 = np.zeros((self.npp, self.npp_bis))
                    
            # Add positivity constraints on the auxiliary variables
            A12x  = np.concatenate((AZ1,  -AI, AZ2, AZ2, AZ2))
            A13x  = np.concatenate((AZ1,  AZ2, -AI, AZ2, AZ2))
            
            A12y  = np.concatenate((AZ1,  AZ2, AZ2, -AI, AZ2))
            A13y  = np.concatenate((AZ1,  AZ2, AZ2, AZ2, -AI))
    
            # Compute an intermediate matrix for b terms for A6, A7, A8, and A9 
            bZ   = np.zeros((2*4*self.npp_bis))
    
            # Add boundary constraints on the apodizer first derivative
            A14Z = np.zeros((self.npp))
            A141 = np.ones(2*2*self.npp_bis)
            A14  = np.concatenate((A14Z, A141))
    
            # Compute b term to bound the integral of the apodizer derivative
            b14  = [self.FirstDerGlobalLim]
                    
            # Update the A and b matrices
            self.A = np.concatenate((self.A, A10x, -A10x, A10y, -A10y, 
                                     A12x, A13x, A12y, A13y, A14[:, None]), axis=1)
            self.b = np.concatenate((self.b,   bZ,  b14))       

        else:
            print('Warning: Set MinIsland keyword to True to add its constraints!')
      


#%%            
    def update_cDarkHole(self):
#        cst = (10.**(-self.cDarkHole/2.)/np.sqrt(2.))*self.corono.Fmax2d/(self.corono.nImg2d*self.corono.nPup)
#
#        A0  =  self.corono_field_t - cst*self.Pupil_vec[self.idx_pup, None]*self.LyotStop_vec[self.idx_pup, None]            
#        A1  = -self.corono_field_t - cst*self.Pupil_vec[self.idx_pup, None]*self.LyotStop_vec[self.idx_pup, None]    
#
#        self.A = np.concatenate((A0,A1), axis=1)
#        if (stdgrb and self.solver == 'stdgrb') or (gb and self.solver == 'gurobipy'):
#            A2  = -np.identity(self.npp)
#            A3  =  np.identity(self.npp)
#            self.A = np.concatenate((self.A,A2,A3), axis=1)

        print('Warning: update_cDarkHole() method is outdated!!!')            

#%%
    @profile
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
                    for i in range(self.npp ))), gb.GRB.MINIMIZE)
            
            # Add constraint:                
            self.m.addConstrs((gb.quicksum((ApodTmp[i]*self.A[i,j] 
                    for i in range(self.npp + self.nvv) if self.A[i,j])) <=  self.b[j] 
                    for j in range(nA)), "cpos")
            
            # Update model        
            self.m.update()

            # Solve model
#            print('save model')
#            self.m.write('/Users/mndiaye/Desktop/model.rlp')
            #MemUse()

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
    default_params = default.get_default_params_2d_MaxContrastProblemMatrix()
    
    @profile
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
        else:
            self.neps = self.ndz

        self.npp_bis = 0
        self.idx_pup_bis = [0]  
        self.nvv   = 0
        if self.MinIsland is True:
            self.idx_pup_bis = list(set().union(list(np.asarray(self.idx_pup)-1),
                                        list(np.asarray(self.idx_pup)-self.corono.nPup), 
                                        self.idx_pup))
            self.npp_bis     = len(self.idx_pup_bis)
            self.nvv         = 4*self.npp_bis
            
        self.N0 = np.zeros((self.neps, self.npp))

#%%    
    @profile
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
        nI1 = 1 
        if self.ImPart is True:
            nI1 = 2

        if self.Lnorm == 'Linf':
            I0 = np.ones(self.ndz)
            I0 = I0[None,:]
            I1 = np.ones(nI1*self.corono.nlam*self.ndz)
            I1 = I1[None,:]
            c1 = [1]
        else:
            I0 = np.identity(self.ndz)
            I1 = np.hstack([I0 for k in range(nI1*self.corono.nlam)])
            c1 = np.array(self.rad2d)

        Z0 = np.zeros(self.neps)


        for k in range(self.ncorono):
            # Compute coronagraph response matrix
            t00 = time.time()
            self.print_log('computing corono response matrix for 2D problem')             
            corono_field_t = self.compute_response_matrices(self.corono_t[k])
            t11 = time.time()
            self.print_log('computing time (response matrices): {0:.2f}s\n'.format(t11-t00))
        
            # Compute constraints on the coronagraphic electric field
            A0tmp  = np.concatenate(( corono_field_t, -I1), axis=0)
            A1tmp  = np.concatenate((-corono_field_t, -I1), axis=0)

            # Add terms corresponding to the MinIaland auxiliary variables        
            AZ0vv = np.zeros((self.nvv,  np.shape(A0tmp)[1]))

            A0tmp = np.concatenate((A0tmp, AZ0vv))
            A1tmp = np.concatenate((A1tmp, AZ0vv))

            # Compute b term corresponding to A0 and A1
            b01 = np.zeros((2*len(corono_field_t.T)))

            #  Yield the A and b matrices for the optimization problem
            if k == 0:
                self.A = np.concatenate((A0tmp,A1tmp), axis=1)
                self.b = b01*1
            else:
                self.A = np.concatenate((self.A, A0tmp, A1tmp), axis=1)
                self.b = np.concatenate((self.b, b01,))

        # Compute constraint on the auxiliary variable epsilon
        A20  = np.concatenate((np.zeros((self.npp, self.ndz)), 
                               -I0,
                               np.zeros((self.nvv, self.ndz))), axis=0)

        # Compute constraint on the integral of the apodizer transmission
        A21  = np.concatenate((-self.Pupil_vec[self.idx_pup]/self.TR, 
                               Z0,
                               np.zeros((self.nvv))), axis=0)
        
        # Compute b terms corresponding to A4 and A5
        b20  = np.zeros(self.ndz)
        b21  = [-self.tau]
        
        #  Yield the A and b matrices for the optimization problem       
        self.A = np.concatenate((self.A, A20,A21[:,None]), axis=1)
        self.b = np.concatenate((self.b, b20,b21))

        # Add apodizer normalization contraints for gurobi solvers
        if (stdgrb and self.solver == 'stdgrb')\
        or (gb and self.solver == 'gurobipy'):        
            self.compute_problem_matrices_gurobi()

        # Add apodizer minimal islands constraints
        if self.MinIsland is True:
            self.compute_problem_matrices_MinIsland()
            
        # Compute the cost function            
        self.c = np.concatenate((np.zeros(self.npp), c1,
                                 np.zeros(self.nvv)), axis=0)
        
        # Returns the A, b, and c matrices        
        return self.A, self.b, self.c

#%%
    @profile
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
    @profile
    def compute_problem_matrices_MinIsland(self):
        r"""
        Computes matrices to add constraints that minimizes the islands in the
        apodizer transmission.
        
        Notes
        -----
        A10x, A10y, bZ : array_like, array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`\frac{d\Phi(r)}{dr} - v^{+}(r) + v^{-}(r) \leq 0`,
            
            where :math:`v^{+}` and :math:`v^{-}` are two auxiliary variables
            to constrain the absolute value of the derivative to remain 
            positive. 
            
        A11x, A11y, bZ : array_like, array_like, array_like
            Constraint of the first derivative of the apodization such that
            
            :math:`-\frac{d\Phi(r)}{dr} + v^{+}(r) - v^{-}(r) \leq 0`.
            
        A12x, A12y : array_like, array_like
            Constraint on the auxiliary variable :math:`v^{+}` to force it 
            to be positive with
            
            :math:`- v^{+}(r) \leq 0`.
        
        A13x, A13y : array_like, array_like
            Constraint on the auxiliary variable :math:`v^{-}` to force it 
            to be positive with
            
            :math:`- v^{-}(r) \leq 0`. 
            
        A14, b14: array_like, array_like
            Constraint on the apodizer first derivative through the auxiliary 
            variables with
            
            :math:`\int_{P_0} (v^{+}(r) + v^{-}(r))dr \leq \delta`.
        
        """       
        if self.MinIsland is True:

        # Reshape the matrice A to account for the auxialiary variables
#        A00    = np.zeros((self.nvv, np.shape(self.A)[1]))
#        self.A = np.concatenate((self.A, A00))

            # Compute the derivative operator of the pupil
            dx0_op = np.zeros((self.corono.nPup**2, self.npp_bis))
            dx1_op = np.zeros((self.corono.nPup**2, self.npp_bis))
    
            for k, val in enumerate(self.idx_pup_bis):
                if val//self.corono.nPup != self.corono.nPup-1:
                    dx0_op[val, k]      = -1
                    dx0_op[val+self.corono.nPup, k] = 1
                    
            for k, val in enumerate(self.idx_pup_bis):
                if val % self.corono.nPup != self.corono.nPup-1:
                    dx1_op[val, k]   = -1
                    dx1_op[val+1, k] = 1
    
            ADx  = dx0_op[self.idx_pup]
            ADy  = dx1_op[self.idx_pup]
                   
            # Compute an intermediate matrix for further computation of A matrices
            AI   = np.identity(self.npp_bis)

            # Add terms corresponding to eps and the MinIsland auxiliary variables     
            AZ0 = np.zeros((self.neps, self.npp_bis))
            
            # Compute intermediate matrices for further computation of A matrices                                
            AZ2 = np.zeros((self.npp_bis, self.npp_bis)) 
    
            # Add constraints on the apodizer first derivative with 
            # auxiliary variables        
            A10x  = np.concatenate(( ADx, AZ0, -AI,  AI, AZ2, AZ2))        
            
            A10y  = np.concatenate(( ADy, AZ0, AZ2, AZ2, -AI,  AI))        
            
            # Compute intermediate matrices for further computation of A matrices                                
            AZ1 = np.zeros((self.npp, self.npp_bis))
     
            # Add positivity constraints on the auxiliary variables
            A12x  = np.concatenate((AZ1, AZ0, -AI, AZ2, AZ2, AZ2))
            A13x  = np.concatenate((AZ1, AZ0, AZ2, -AI, AZ2, AZ2))
            
            A12y  = np.concatenate((AZ1, AZ0, AZ2, AZ2, -AI, AZ2))
            A13y  = np.concatenate((AZ1, AZ0, AZ2, AZ2, AZ2, -AI))
           
            # Compute an intermediate matrix for b terms from A10 to A14 
            bZ   = np.zeros((2*4*self.npp_bis))
    
            # Add boundary constraints on the apodizer first derivative
            A14Z = np.zeros((self.npp))
            A141 = np.ones(2*2*self.npp_bis)
            A14  = np.concatenate((A14Z, np.zeros((self.neps)), A141))
    
            # Compute b term to bound the integral of the apodizer derivative
            b14  = [self.FirstDerGlobalLim]
                    
            # Update the A, b, and c matrices
            self.A = np.concatenate((self.A, A10x, -A10x, A10y, -A10y, A12x, A13x, A12y, A13y, A14[:, None]), axis=1)
            self.b = np.concatenate((self.b,   bZ, b14))       

        else:
            print('Warning: Set MinIsland keyword to True to add its constraints!')


#%%            
    def update_tau(self):
#        b0  = np.zeros((self.corono.nlam*self.ndz*2))       
#        b4  = np.zeros(self.ndz)
#        b5  = [-self.tau]
#
#        self.b = np.concatenate((b0,b0,b4,b5))
#
#        if (stdgrb and self.solver == 'stdgrb') or (gb and self.solver == 'gurobipy'):        
#            b2  = np.zeros(self.npp)
#            b3  = np.ones(self.npp)
#            self.b = np.concatenate((self.b,b2,b3))
#            
        print('Warning: update_tau() method is outdated!!!') 
            
#%%
    @profile
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