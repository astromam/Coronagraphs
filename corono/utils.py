#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:10:51 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

import numpy as np
import scipy.special
import copy

#%% check python version
"""
Check python version
"""
try:
    reload  # Python 2.7
    reload=reload
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3
        
#%%
def to_dict(**kwargs):
    """
    Returns the arguments in the dictonary
    """
    return kwargs

#%%
def uniform_disk(n, radius, CtrBtwnPix=False):
    """
    Generates a uniform disk in a 2D array.
    
    Parameters
    ----------
    n : integer
        size of the array
    
    radius : float
        radius of the disk
    
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ----------
    res : array_like
        (ys x xs) array with a uniform disk of radius "radius".
        
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2 
    xx,yy  = np.meshgrid(np.arange(n)-n/2+val, np.arange(n)-n/2+val)
    mydist = np.hypot(yy,xx)
    res    = np.zeros_like(mydist)
    res[mydist <= radius] = 1.0
    return res

#%%
def radius_disk(n, radius, CtrBtwnPix=False):
    """
    Generates a uniform disk in a 2D array.
    
    Parameters
    ----------
    n : integer
        size of the array
    
    radius : float
        radius of the disk
    
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ----------
    res : array_like
        (ys x xs) array with a uniform disk of radius "radius".
        
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2 
    xx,yy  = np.meshgrid(np.arange(n)-n/2+val, np.arange(n)-n/2+val)
    mydist = np.hypot(yy,xx)/n
    return mydist


#%%
def sft(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square. 

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    
    sign = -1.0
    if inv:
        sign = 1.0

    U = np.zeros((1,NB))
    X = np.zeros((1,NA))
    
    X[0,:] = (1./NA)*(np.arange(NA)-NA/2.+val)
    U[0,:] =  (m/NB)*(np.arange(NB)-NB/2.+val)
       
#     A1 = np.exp(sign * 2.j*np.pi* U.T.dot(X))
#     A3 = np.exp(sign * 2.j*np.pi* X.T.dot(U))
    A1 = sign*1j*np.sin(2.*np.pi* U.T.dot(X))
    A1 += np.cos(2.*np.pi* U.T.dot(X))
    
    A3 = sign*1j*np.sin(2.*np.pi* X.T.dot(U))
    A3 += np.cos(2.*np.pi* X.T.dot(U))
    B  = (A1.dot(A2)).dot(A3)

    return coeff*B

#%%
def isft(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].

    See Also
    --------
    sft() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

#%% import Bessel function
def besselJ0(z):
    """
    Computes the Bessel function of zero order for a given array
    
    Parameters
    ---------
    z : array_like
        variable for which the bessel function is computed
    
    Returns
    ---------
    temp : array_like
        Bessel function of zero order for the z array
    
    """
    temp=scipy.special.jv(0,z)
    temp[np.isnan(temp)]=0
    return temp

#%%
def write_apod1d(fpath, test):
    """
    Saves 1D apodizer
    
    Parameters
    ----------
    fpath : string
        Filepath in which to save fpath
    
    test : array_like 
        Array to be saved
    
    """
    np.savetxt(fpath, test)
    
#%%
def load_apod1d(fpath):
    """
    load 1D apodizer
    
    Parameters
    ----------
    fpath : string
        Path of the file to load
        
    Returns
    ---------
    rApod : array_like
        Radial coordinate of the apodizer points to be loaded
    
    Apod : array_like
        Apodizer to be loaded
    
    """
    test  = np.loadtxt(fpath)
    rApod = test[:,0]
    Apod  = test[:,1]
    
    return rApod,Apod    

#%%
def sft_even(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square and represents an even function. 

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
                
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    
    U = np.zeros((1,NB))
    X = np.zeros((1,NA))
    
    X[0,:] = (1./NA)*(np.arange(NA)-NA/2+val)
    U[0,:] =  (m/NB)*(np.arange(NB)-NB/2+val)

    A1 = np.cos(2.*np.pi* U.T.dot(X))    
    A3 = np.cos(2.*np.pi* X.T.dot(U))
    
    B  = (A1.dot(A2)).dot(A3)

    return coeff*B

#%%
def isft_even(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].
    Assumes the original array is square and represents an even function. 

    See Also
    --------
    sft_even() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft_even(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

#%%
def update_params(params, **kwargs):
    """
    Update a dictionary of parameters
    
    Parameters
    --------
    params : dict
        the original dictionary
    
    kwargs : dict
        parameters given by the user for the keys with the values to update
    
    Returns
    --------
    params : dict
        a copy of the original dictionary with an update of the parameters
        given by the user
    
    """
    params2 = copy.deepcopy(params)
    if len(kwargs) != 0:
        for key, value in kwargs.items():
            params2[key] = value
    return params2

#%%
def line_search_armijo(f, xk, pk, gfk, old_fval=None,
                       args=(), c1=1e-4, alpha0=0.99):
    """
    Armijo linesearch function that works with matrices

    find an approximate minimum of f(xk+alpha*pk) that satifies the
    armijo conditions.

    Parameters
    ----------

    f : function
        loss function
    xk : np.ndarray
        initial position
    pk : np.ndarray
        descent direction
    gfk : np.ndarray
        gradient of f at xk
    old_fval : float
        loss value at xk
    args : tuple, optional
        arguments given to f
    c1 : float, optional
        c1 const in armijo rule (>0)
    alpha0 : float, optional
        initial step (>0)

    Returns
    -------
    alpha : float
        step that satisfy armijo conditions
    fc : int
        nb of function call
    fa : float
        loss value at step alpha

    """
    xk = np.atleast_1d(xk)
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval

    derphi0 = np.sum(pk * gfk)  # Quickfix for matrices
    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0)

    return alpha, fc[0], phi1
#%%
def line_search_ratio(x,deltax,Ke,Kp): 
            a=np.dot(Ke,deltax)
            b=np.dot(Ke,x)
            c=np.dot(Kp,deltax)
            d=np.dot(Kp,x)
            g=lambda L,t  : (L[0]*(t**2)+L[1]*t+L[2])/(L[3]*(t**2)+L[4]*t+L[5])
            L=[]
            L.append(np.dot(deltax,a))
            L.append(2*np.dot(deltax,b))
            L.append(np.dot(x,b))
            L.append(np.dot(deltax,c))
            L.append(2*np.dot(deltax,d))
            L.append(np.dot(x,d))
            P=[]
            P.append(L[0]*L[4]-L[3]*L[1])
            P.append(2*(L[0]*L[5]-L[3]*L[2]))
            P.append(L[1]*L[5]-L[4]*L[2])
            Disc=(P[1]**2)-4*P[0]*P[2]
#            P=[]
#            P.append((np.dot(deltax,a)*2*np.dot(x,c)-(np.dot(deltax,c)*2*np.dot(x,a))))
#            P.append(2*(((np.dot(deltax,a))*np.dot(x,d))-(np.dot(x,b)*np.dot(deltax,c))))
#            P.append(((2*np.dot(x,a))*np.dot(x,d))-(2*np.dot(x,b)*np.dot(x,c)))
#            Disc=(P[1]**2)-4*P[0]*P[2]
            if Disc>0:
                alpha0=(-P[1]+np.sqrt(Disc))/(2*P[0])
                alpha1=(-P[1]-np.sqrt(Disc))/(2*P[0])
                alp=[0,1,alpha0,alpha1]
                l=[g(L,0),g(L,1)]
                if alpha0>0 and alpha0<1:
                    l.append(g(L,alpha0))
                else:
                        l.append(l[1])
                if alpha1>0 and alpha1<1:
                    l.append(g(L,alpha1))
                else:
                    l.append(l[1])
            else :
                if g(L,0)<g(L,1):
                    alpha=0
                else:
                    alpha=1
                

                
            alpha=alp[l.index(min(l))]
            f_val=g(L,alpha)
            return(f_val,alpha)
    
 #%%   
def fmin_cond(f, df, solve_c, x0, Ke, Kp,linesearch, nbitermax=200,
              stopvarj=1e-9, verbose=False, log=False):
    r""" Solve constrained optimization with conditional gradient

        The function solves the following optimization problem:

    .. math::
        \min_x \quad f(x)

        \text{s.t.} \quad x\in P

    where :

    - f is differentiable (df) and Lipshictz gradient
    - solve_c is the solver for the linearized problem of the form
        .. math::
            \min_x \quad x^T v

            \text{s.t.} \quad x\in P


    Parameters
    ----------
    f : function
        Smooth function f: R^d -> R
    df : function
        Gradient of f, df:R^d -> R^d
    solve_c : function
        Solver for linearized problem, solve_c:R^d -> R^d
    x_0 : (d,) numpy.array
        Initial point
    nbitermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    x : ndarray
        solution
    val : float
        Optimal value at solution
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------
    """

    loop = 1

    if log:
        log = {'loss': []}

    x = x0
    f_val = f(x0)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print(('{:5s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32))
        print(('{:5d}|{:8e}|{:8e}'.format(it, f_val, 0)))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        g = df(x)

        # solve linearization
        xc = solve_c(x, g)

        deltax = xc - x

        # line search
        if linesearch==1:
            f_val,alpha=line_search_ratio(x,deltax,Ke,Kp)
            
        else:
                
            alpha, fc, f_val = line_search_armijo(f, x, deltax, g, f_val)
        
        if alpha is not None :
            
            x = x + alpha * deltax
        
        else:
            loop = 0
        # test convergence
        if it >= nbitermax:
            loop = 0

        delta_fval = (f_val - old_fval) / abs(f_val)
        if abs(delta_fval) < stopvarj:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print(('{:5s}|{:12s}|{:8s}'.format(
                    'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32))
            print(('{:5d}|{:8e}|{:8e}'.format(it, f_val, delta_fval)))

    if log:
        return x, f_val, log
    else:
        return x, f_val
    
    #%%
    
def grad(w,Ke,Kp): return (2*np.dot(Ke,w)*np.dot(np.dot(w,Kp),w)-2*np.dot(Kp,w)*np.dot(np.dot(w,Ke),w))\
/(np.dot(np.dot(w,Kp),w))**2

#%%
def fonc(w,Ke,Kp): return np.dot(np.dot(w,Ke),w)/np.dot(np.dot(w,Kp),w)

#%%

def solve_closed_form(g,w,tau):
    x=np.zeros_like(w)
    s=g/np.asarray(w)
    s=sorted(range(len(s)), key=lambda k: s[k])
    i=0
    while np.dot(w.T,x)<tau:        
        x[s[i]]=1
        i+=1
    x[s[i-1]]=0
    x[s[i-1]]=(tau-np.dot(w,x))/w[s[i-1]]
    return x
        