# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:47:55 2019

@author: adamh
"""
import numpy as np
from scipy.optimize.linesearch import scalar_search_armijo


#%%
def cost_function_snr(x,psi_star,psi_planet):
    """
    function that computes the cost function for the frank wolfe algorithm. 
    It corresponds to the SNR for an apodizer x
    """
    a=np.dot(x,psi_star)
    b=np.dot(x,psi_planet)
    
    return (np.dot(a,a.T)/np.dot(b,b.T))

#%%
def gradient_function_snr(x,psi_star,psi_planet):
    """
    function that computes the gradient of the cost function (snr) for the frank wolfe algorithm
    
    """
    a=np.dot(psi_star.T,x)
    b=np.dot(psi_planet.T,x)
    c=np.dot(b.T,b)
    
    return 2*(c*np.dot(psi_star,a)-np.dot(a,a.T)*np.dot(psi_planet,b))/c**2
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
def line_search_ratio_matrix_k(x,deltax,Ke,Kp): 
        """
        Closed-form line-search for the Frank-Wolfe algorithme and a cost function
        of the form 
        :math f(x) = \frac{x^{T}\Psi^{T}_{star}\Psi_{star}x}{x^{T}\Psi_{planet}^{T}\Psi_{planet}x}
        This algorithm uses directly Ke and Kp which corresponds to :math \Psi^{T}_{star}\Psi_{star}
        and \Psi_{planet}^{T}\Psi_{planet}. The function "line_search_ratio" should be preferred
        
        
        returns :
        alpha : float in [0,1]
            Optimal step
            
        f_val : Value of the cost function after the step \alpha
        """

    
        a=np.dot(Ke,deltax)
        b=np.dot(Ke,x)
        c=np.dot(Kp,deltax)
        d=np.dot(Kp,x)
        g=lambda L,t  : (L[0]*(t**2)+L[1]*t+L[2])/(L[3]*(t**2)+L[4]*t+L[5])
        L=[np.dot(deltax,a),
           2*np.dot(deltax,b),
           np.dot(x,b),
           np.dot(deltax,c),
           2*np.dot(deltax,d),
           np.dot(x,d)]

        P=[L[0]*L[4]-L[3]*L[1],
           2*(L[0]*L[5]-L[3]*L[2]),
           L[1]*L[5]-L[4]*L[2],
           ]

        Disc=(P[1]**2)-4*P[0]*P[2]
                
        #if P has real roots, they are computed. Then, we check if they are in [0,1]. 
        #Finally, the various possible steps are compared and the one that minimizes the 
        #cost function is selected

        if Disc>0:
            alpha0=(-P[1]+np.sqrt(Disc))/(2*P[0])
            alpha1=(-P[1]-np.sqrt(Disc))/(2*P[0])
            
            #alp is the list of all possible optimal steps.
            alp=[0,1,alpha0,alpha1]
            
            #l is the list of the value of the cost function after these steps.
            l=[g(L,0),g(L,1),0,0]
            
            if alpha0>0 and alpha0<1:
                l[2]=(g(L,alpha0))
            else:
                
                #if alpha0 is not an admissible step, we just set l[2] to l[1] just to be sure it won't be choosen
                    l[2]=l[1]
            if alpha1>0 and alpha1<1:
                l[3]=(g(L,alpha1))
            else:
                #if alpha1 is not an admissible step, we just set l[2] to l[1] just to be sure it won't be choosen

                l[3]=l[1]
        else :
        #if P has no real roots, the optimal is either 0 or 1 so these two
        #possibilities are compared
            if g(L,0)<g(L,1):
                alpha=0
            else:
                alpha=1
            

            
        alpha=alp[l.index(min(l))]
        f_val=g(L,alpha)
        return(f_val,alpha)
#%%
def line_search_ratio(x,deltax,psi_star,psi_planet): 
        """
        Closed-form line-search for the Frank-Wolfe algorithme and a cost function
        of the form 
        :math f(x) = \frac{x^{T}\Psi^{T}_{star}\Psi_{star}x}{x^{T}\Psi_{planet}^{T}\Psi_{planet}x}
        
        
        
        returns :
        alpha : float in [0,1]
            Optimal step
            
        f_val : Value of the cost function after the step \alpha
        """
        
        #Compute some vectors that would be useful to implement the linesearch algorithm
        a=np.dot(psi_star.T,x)
        b=np.dot(psi_star.T,deltax)
        c=np.dot(psi_planet.T,x)
        d=np.dot(psi_planet.T,deltax)
        
        #Compute the coefficients of the numerator and the denominator of the cost function
        #these coefficients are saved in the list L


        a_1=np.dot(b.T,b)
        b_1=2*np.dot(b.T,a)
        c_1=np.dot(a.T,a)
        a_2=np.dot(d.T,d)
        b_2=2*np.dot(d.T,c)
        c_2=np.dot(c.T,c)
        
        #g compute the cost function from the coefficients that are saved in L
        #to avoid unnecessary calculations
        
        g=lambda L,t  : (L[0]*(t**2)+L[1]*t+L[2])/(L[3]*(t**2)+L[4]*t+L[5])
        
        
        L=[a_1,b_1,c_1,a_2,b_2,c_2]
        
        #The list P contains the coefficients of the numerator of the derivative.
        #The possible optimal step alpha is among 0, 1 or one of the roots of P
        P=[L[0]*L[4]-L[3]*L[1],
           2*(L[0]*L[5]-L[3]*L[2]),
           L[1]*L[5]-L[4]*L[2]]
    
        
        
        Disc=(P[1]**2)-4*P[0]*P[2]
        
        #if P has real roots, they are computed. Then, we check if they are in [0,1]. 
        #Finally, the various possible steps are compared and the one that minimizes the 
        #cost function is selected
        if Disc>0:
            alpha0=(-P[1]+np.sqrt(Disc))/(2*P[0])
            alpha1=(-P[1]-np.sqrt(Disc))/(2*P[0])
            #alp is the list of all possible optimal steps.
            alp=[0,1,alpha0,alpha1]
            
            #l is the list of the value of the cost function after these steps.
            l=[g(L,0),g(L,1),0,0]
            
            if alpha0>0 and alpha0<1:
                
                l[2]=(g(L,alpha0))
                
            else:
                
            #if alpha0 is not an admissible step, we just set l[2] to l[1] just to be sure it won't be choosen
                    l[2]=l[1]
                    
            if alpha1>0 and alpha1<1:
                
                l[3]=(g(L,alpha1))
                
            else:
            #if alpha1 is not an admissible step, we just set l[2] to l[1] just to be sure it won't be choosen

                l[3]=l[1]
                
        #if P has no real roots, the optimal is either 0 or 1 so these two
        #possibilities are compared
        else :
            if g(L,0)<g(L,1):
                alpha=0
            else:
                alpha=1
            

            
        alpha=alp[l.index(min(l))]
        f_val=g(L,alpha)
        return(f_val,alpha)
    
    
    
 #%%   
def fmin_cond(f, df, solve_c, x0, psi_star, psi_planet,linesearch, nbitermax=200,
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
            f_val,alpha=line_search_ratio(x,deltax,psi_star,psi_planet)
            
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

def solve_closed_form(g,w,tau):
    """
    Solve the linear optimization problem that has to be solved at each step
    of the Frank-Wolfe algorithm. It computes the closed-form solution of this problem
    
    Returns
    -------
    x : ndarray
        closed-form solution
    
    """
    x=np.zeros_like(w)
    
    s=g/np.asarray(w)
    s=sorted(range(len(s)), key=lambda k: s[k])
    
    
    #While the transmission of  the apodizer x has not reached :math \tau, a maximum
    #weight is is placed on the smallest elements of the vector s
    
    i=0
    while np.dot(w.T,x)<tau:        
        x[s[i]]=1
        i+=1
    x[s[i-1]]=0
    x[s[i-1]]=(tau-np.dot(w,x))/w[s[i-1]]
    
    return x