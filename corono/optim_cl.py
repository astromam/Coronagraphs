# -*- coding: utf-8 -*-
"""
#!/usr/bin/env python3
Created on Fri Mar  9 11:36:39 2018

@author: mndiaye
"""

#%% Initialization problem
import numpy as np
import json
import gurobipy as gb
from corono import coronagraph_cl as cg

#%%
"""
Default parameters
"""
#%%
def get_default_params_ProblemMatrix():
    """
    Gets the default parameters for the optimization problem matrix.
       
    Parameters
    ----------
    cDarkHole : float
        contrast goal in log scale inside the search area in the 
        coronagraphic image
    
    tau : float
        integrated amplitude transmission goal in fraction of the pupil 
        amplitude transmission
    
    Returns    
    ----------
    tmp : dict
        dictionnary of parameters with their default values
        
    """
    tmp = {'cDarkHole':8,'tau':0.2}
    return tmp

#%%
def get_default_params_MaxContrastProblemMatrix():
    """
    Gets the default parameters for the Max contrast optimization problem.
    
    Parameters
    ---------- 
    tmp : dict
        dictionary from the get_default_matrix_pb
        
    Lnorm : string
        L-norm for the optimization problem ('Linf' : L-infinite norm, 
        'L1' : L1-norm)
            
    Returns    
    ----------
    tmp : dict
        updated dictionary
        
    """
    
    tmp = get_default_params_ProblemMatrix()
    tmp.update({'Lnorm':'L1'})
    return tmp

#%%
"""
Problem Matrix class
"""
class ProblemMatrix(object):
    """
    Defines the class for Matrix of optimization problem
    """
    default_params = get_default_params_ProblemMatrix()
  
    def __init__(self,corono=cg.APLC1d(),**kwargs):
        """
        __init__ : method
            build the constructor for the ProblemMatrix class
        
        Attributes
        ----------
        params : dict
            dictionary of parameters for the Coronagraph class
        
        corono : class
            object of the Coronagraph class
        
        dz : vector_like
            dark zone points in the coronagraphic image
            
        aaa : vector_like
            vector indexing the points in the coronagraphic image
            
        idz_dz : vector_like
            vector indexing the points of the dark zone in the coronagraphic 
            image
            
        ndz : integer
            number of points of the dark zone in the coronagraphic image
            
        pup : vector_like
            index of non zero points in the aperture
            
        bbb : vector_like
            vector indexing the points in the pupil
            
        idx_pup : vector_like
            vector indexing the non zero points in the pupil
            
        npp : integer
            number of non zero points in the aperture
            
        lys : vector_like
            index of non zero points in the Lyot stop
            
        idx_lys : vector_like
            vector indexing the non zero points of the pupil in the Lyot stop
            
        direct_field_t_re, direct_field_t_im : 3d array
            real and imaginary part of the non coronagraphic response matrix 
            for all the points in the pupil and at all the wavelengths
            
        corono_field_t_re, corono_field_t_im : 3d array
            real and imaginary part of the non coronagraphic response matrix 
            for all the points in the pupil and at all the wavelengths
         
        corono_field_t_re2 : 3d array
            real part of the non coronagraphic response matrix 
            for all the non zero points in the pupil and at all the wavelengths
                                
        A, b, c : matrices
            matrix to solve the problem for a given variable x
            A.x <= b under the cost function c.T.x
        
        TR : float
            integrated amplitude transmission of the pupil with respect to that
            of the clear pupil
            
        m : gurobi model
            gurobi model of the problem to solve
            
        Apod : vector_like
            apodizer to be generated
        
        """
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
        
        self.A       = None
        self.b       = None
        self.c       = None
        
        self.m       = None
        
        self.Apod    = np.zeros((self.corono.nPup))
        
        self.TR      = np.sum(2.*np.pi*self.corono.Pupil *np.linspace(
                0.5,self.corono.nPup+0.5,num=self.corono.nPup)\
                /(2.*self.corono.nPup)**2) 

#%%    
    def compute_matrices(self):
        """
        Virtual function for the matrix computation.
        """
        print('Warning: virtual fct - no A, b and c matrices will be computed')

#%%
    def __contains__(self, item):
        """
        Checks params for a given item.
        
        Parameters
        ----------
        item : string 
            key in params dictionary
        
        Return:
        ----------
        value for the item in the dictionary  
            
        """
        return item in self.params
    
#%%     
    def __getattr__(self, name):
        """
        Checks the attribute for the params.
        
        Parameters
        ----------
        name : string
            name of the variable to be retrieved
        
        Return:
        ----------
        the value of name in params
        
        """
        return self.params[name]

#%%        
    def check_params(self):
        """
        Check the params.
        
        Return:
        ----------
        return the values for all the keys in the params
        
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
            filename to load    
        
        """
        
        f=open(fname,'r')
        params=json.loads(f.read())
        self.__init__(**params)
        f.close()              

#%%        
    def solve_model(self):
        """
        Solves the optimization problem for the model using the gurobi solver.
        """
        try:
            
            self.m.Params.Method       = 2
            self.m.Params.LogToConsole = 1
            self.m.Params.Crossover    = 0
            
            self.m.optimize()

            Apodtmp = np.zeros((self.npp))
            for i in range(self.npp):
                Apodtmp[i] = self.m.getVars()[i].x
            self.Apod[self.idx_pup] = Apodtmp
            
            test = np.zeros((self.corono.nPup, 2))
            test[:,0] = self.corono.r
            test[:,1] = self.Apod   
#            if fpath: write_apod1d(fpath, test)    
                
            return self.Apod

        except gb.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError:
            print('Encountered an attribute error')

        
#%%
"""
MaxTau ProblemMatrix subclass
"""
class MaxTau(ProblemMatrix):
    """
    Defines the ProblemMatrix subclass for the optimization problem that 
    maximizes the apodizer transmission for a given contrast in the search area.
    """
    def __init__(self, corono=cg.APLC1d(), **kwargs):
        """
        Constructor for the Matrix problem with the coronagraph object
        """
        super().__init__(**kwargs)

#%%        
    def compute_matrices(self):
        """
        Computes the matrices for the optimization problem that consists in 
        maximizing the apodizer transmission for a set contrast in a given 
        search area in the coronagraphic image.
        
        Parameters
        -----------        
        A0, A1 : matrices
            contrast constraints on the coronagraphic electric field
            
        A2, A3 : matrices
            plus and minus identity matrices for the non zero points in the
            pupil
            
        b0, b1 : matrices
            zero matrices with size is related to A0 and A1
            
        b2, b3 : matrices
            zero matrices with size is related to A2 and A3
       
        
        Return : 
        ----------
        A, b, c:
            the matrices for the optimization problem described above
        
        
        """
        direct_field_t1_re = np.zeros((self.corono.nPup, self.corono.nlam*self.ndz))
        for j in range(self.corono.nlam*self.ndz):
            direct_field_t1_re[:,j] = \
            self.direct_field_t_re[:,(self.corono.nlam-1)//2,0]
        direct_field_t2_re = direct_field_t1_re[self.idx_pup,:]

        
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

#%%
    def compute_gurobi_model(self):
        """
        Generates the gurobi solver model for the MaxTau problem.
        
        Parameters 
        -----------
        Apodtmp : vector_like
            vector of the apodizer in the non zero points of the pupil
        
        Returns
        -----------
        m : gurobi model
            gurobi model of the MaxTau problem to solve
            
        """
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
        else:
            raise ValueError('Be careful: A or b or c is not defined')
        return self.m
 

#%%
"""
MaxContrast ProblemMatrix subclass
"""
class MaxContrast(ProblemMatrix):
    """
    Defines the ProblemMatrix subclass for the optimization problem that 
    maximizes the contrast in a given search area for a given integrated 
    apodizer transmission.
    """
    default_params = get_default_params_MaxContrastProblemMatrix()
    
    def __init__(self, corono=cg.APLC1d(), **kwargs):
        """
        builds the constructor for the Matrix problem with 
        the coronagraph object
        """
        super().__init__(**kwargs)

#%%    
    def compute_matrices(self):
        """
        Computes the matrices for the optimization problem that consists in 
        maximizing the contrast in a given search area in the coronagraphic 
        image for a set integrated apodizer transmission.

        Parameters
        -----------
        Lnorm : string
            type of L-norm for the optimization problem
            
        I0, I1, N0, Z0, c1 : matrices
            intermediate matrices for the generation of the matrices A0 to A5
            
        A0, A1 : matrices
            contrast constraints on the coronagraphic electric field
            
        A2, A3 : matrices
            plus and minus identity matrices for the non zero points in the
            pupil
            
        A4 : matrix
            
        A5 : matrix
            matrix for the integrated apodizer transmission
        
            
        b0, b1 : matrices
            zero matrices with size is related to A0 and A1
            
        b2 : matrix
            zero matrix with size is related to A2
        
        b3 : matrix
            one matrix with size relative to A3
            
        b4 : matrix
            zero matrix with size relative to A4
            
        b5 : matrix
            single value matrix with tau, the set threshold for the integrated 
            apodizer transmission      
        
        Return:
        -----------
        A, b, c:
            the matrices for the optimization problem described above                                            
        
        """
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

#%%
    def compute_gurobi_model(self):
        """
        Generates the gurobi solver model for the MaxContrast problem.
        
        Parameters 
        -----------
        ApodEpstmp : vector_like
            vector of the apodizer in the non zero points of the pupil 
            and neps points for the coronagraphic image
        
        Returns
        -----------
        m : gurobi model
            gurobi model of the MaxTau problem to solve
            
        """        
        if self.A is None or self.b is  None or self.c is None:
            print('computing matrices')
            self.compute_matrices()
                       
        nn = np.shape(self.A)[1]
        # Create a new model  
        self.m = gb.Model("LP max C new")
        
        if self.Lnorm == 'Linf':
            self.neps = 1
        else:
            self.neps = self.ndz
        # Create variables
        ApodEpsTmp = self.m.addVars(self.npp + self.neps, lb=0.0, name="ApodTmp")        
        # Set objective
        self.m.setObjective(gb.quicksum((self.c[i+self.npp]*ApodEpsTmp[i+self.npp] 
                for i in range(self.neps))), gb.GRB.MINIMIZE)
        # Add constraint:
        self.m.addConstrs((gb.quicksum((ApodEpsTmp[i]*self.A[i,j] 
                for i in range(self.npp + self.neps))) <=  self.b[j] 
                for j in np.arange(nn)), "cpos")
        
        self.m.update()
            
        return self.m   

#%%