#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:53:11 2019

@author: ahessas
"""
from time import time
import numpy as np
from scipy.optimize.linesearch import scalar_search_armijo
import scipy
import pylab as pl
import corono as coro
import random
from qpsolvers import solve_qp
#%% parameters
"""
Parameters
"""
nlambis=11
corono_name  = 'APLC' # 'APLC' or 'SP'
problem_name = 'MaxContrastL1' # ,'MaxTau' # 'MaxContrastLinf' #'MaxContrastL1'
solver       = 'scipy.linprog' # 'stdgrb', 'gurobipy', 'scipy.linprog'
slvLogToConsole = 0
slvCrossover    = 0
slvMethod       = 2
allLogToConsole = 0

FirstDer    = False
SecondDer   = False
MinIsland   = False
FirstDerLim = 0.01
SecondDerLim= 0.001 
FirstDerGlobalLim = 10.

nPup = 500
nFPM = 50
nImg = 44
Fmax = 11
R    = 1

bw   = 0.1
nlam =5

PupilID    = 0.15
rMask       = 4.0

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75

LyotStopID = 0.3
LyotStopOD = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 3.5
rho1 = 10

# contrast in the dark region
cDarkHole = 9.0

# tau (integrated Pupil transmission)
tau   = 0.5
r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilID)*1.0
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, problem_name = problem_name,
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)

""" 
Coronagraph defintion
"""
corono0 = coro.design.APLC1d(**params)

""" 
Problem definition and solving 1d
"""
problem1 = coro.optim_1d.MaxTau(corono=corono0, **params)
problem2 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L2', **params)
problem3 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='Linf', **params)
"""
Construction des matrices du problème d'optimisation linéaire MaxTau i.e. 
On fixe le contraste voulue et on maximise la transmission Tau Sous contrainte
de contraste
"""
def get_matrices_Tau(problem):
    problem.compute_response_matrices()
    cst = 10.**(-problem.cDarkHole/2.)/np.sqrt(2.)
    A_p =  problem.corono_field_t \
        - cst*problem.direct_field_re_t_tmp[:,(problem.corono.nlam-1)//2,0, None]
    A_m  = -problem.corono_field_t \
        - cst*problem.direct_field_re_t_tmp[:,(problem.corono.nlam-1)//2,0, None]
    A=np.concatenate((A_p,A_m),axis=1)
    b=np.zeros(np.shape(A)[1])
    ctmp = np.zeros(problem.npp )
    ctmp[:problem.npp] = np.asarray(problem.idx_pup)+0.5
    c = - problem.corono.R*2.*np.pi*ctmp/(problem.corono.nPup*problem.TR)
    return A,b,c
"""
Construction des matrices du problème d'optimisation linéaire MaxContrastL1 i.e. 
On minimise la norme L1 du résidu sous contrainte d'une transmission Tau voulue
"""
def get_matrices_Contrast_un(problem):
    problem.compute_response_matrices()
    I0 = np.identity(problem.ndz)
    I1 = np.hstack([I0 for k in range(problem.corono.nlam*2)])            
    c = np.concatenate((np.zeros(problem.npp), np.ones(problem.ndz)))
    A0  = np.concatenate(( problem.corono_field_t, -I1), axis=0)
    A1  = np.concatenate((-problem.corono_field_t, -I1), axis=0)
    ctmp = np.zeros(problem.npp)
    ctmp[:problem.npp] = np.asarray(problem.idx_pup)
    A20 = np.concatenate((np.zeros((problem.npp, problem.ndz)), -I0),axis=0)
    
    
    A21  = np.concatenate(((-1/sum(ctmp)*(ctmp)), 
                              (np.zeros(problem.neps))), axis=0)

    b0  = np.zeros((2*problem.corono.nlam*problem.ndz*2))
    
    b1  = np.zeros(problem.ndz)
    b2  = [-problem.tau]
    A = np.concatenate((A0,A1,A20,A21[:,None]), axis=1)
    b = np.concatenate((b0,b1,b2))
    return A,b,c
"""
On résoud ici le problème d'optimisation linéaire L1 à l'aide du solver Scipy
Tout en affichant l'intensité résiduelle
 """   
def solve1(problem):
    A,b,c=get_matrices_Contrast_un(problem)
    bds = np.zeros((len(A), 2))
    bds[:,1] = 1.
    sol=scipy.optimize.linprog(c,A.T,b,method='interior-point', bounds=bds, options={'sparse':False})
#    problem.Apod[problem.idx_pup]=sol.x[:problem.npp]
#    pl.plot(corono0.r, problem.Apod, label=solver)
#    pl.show()
#    #Display of the intensity profiles of the coronagraphic images'
#    poly_direct_image1 = corono0.compute_direct_intensity_1d(problem.Apod)
#    poly_corono_image1 = corono0.compute_corono_intensity_1d(problem.Apod)
#    pl.axvline(x=rho0,color='red')
#    pl.axvline(x=rho1,color='red')
#    pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label=solver)
#    pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#    pl.xlabel(r'Angular separation in $\lambda_0$/D')
#    pl.ylabel('Normalized intensity in log scale')
#    pl.ylim(1e-13, 1e-4)
#    pl.legend()
#    pl.show()
    a=np.zeros_like(problem.Apod)
    a[problem.idx_pup]=sol.x[:problem.npp]
    return a[problem.idx_pup]
"""
Construction des matrices du problème d'optimisation linéaire MaxContrastLinf i.e. 
On minimise la norme Linf du résidu sous contrainte d'une transmission Tau voulue
"""
def get_matrices_Contrast_inf(problem):
    problem.compute_response_matrices()
#    if problem.problem_name == 'MaxContrastLinf':
    c = np.concatenate((np.zeros(problem.npp),[1]))
    I0 = np.ones(problem.ndz)
    I0 = I0[None,:]
    I1 = np.ones(problem.ndz*problem.corono.nlam*2)
    I1 = I1[None,:]
    A0  = np.concatenate(( problem.corono_field_t, -I1), axis=0)
    A1  = np.concatenate((-problem.corono_field_t, -I1), axis=0)
    ctmp = np.zeros(problem.npp )
    ctmp[:problem.npp] = np.asarray(problem.idx_pup)
    A20 = np.concatenate((np.zeros((problem.npp, problem.ndz)), -I0),axis=0)
    
    
    A21  = np.concatenate(((-1/sum(ctmp)*(ctmp)), [0]), axis=0)

    b0  = np.zeros((2*problem.corono.nlam*problem.ndz*2))
    
    b1  = np.zeros(problem.ndz)
    b2  = [-problem.tau]
    A = np.concatenate((A0,A1,A20,A21[:,None]), axis=1)
    b = np.concatenate((b0,b1,b2))
#    else :
#        A=b=c=0
    return A,b,c
"""
On résoud ici le problème d'optimisation linéaire L1 à l'aide du solver Scipy
Tout en affichant l'intensité résiduelle
"""

def solveinf(problem):
    A,b,c=get_matrices_Contrast_inf(problem)
    bds = np.zeros((len(A), 2))
    bds[:,1] = 1.
    sol=scipy.optimize.linprog(c,A.T,b,method='interior-point', bounds=bds, options={'sparse':False})
#    problem.Apod[problem.idx_pup]=sol.x[:problem.npp]
#    pl.plot(corono0.r, problem.Apod, label=solver)
#    pl.show()
#    #Display of the intensity profiles of the coronagraphic images'
#    poly_direct_image1 = corono0.compute_direct_intensity_1d(problem.Apod)
#    poly_corono_image1 = corono0.compute_corono_intensity_1d(problem.Apod)
#    pl.axvline(x=rho0,color='red')
#    pl.axvline(x=rho1,color='red')
#    pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label=solver)
#    pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#    pl.xlabel(r'Angular separation in $\lambda_0$/D')
#    pl.ylabel('Normalized intensity in log scale')
#    pl.ylim(1e-13, 1e-4)
#    pl.legend()
#    pl.show()
    a=np.zeros_like(problem.Apod)
    a[problem.idx_pup]=sol.x[:problem.npp]
    return a[problem.idx_pup]
"""
Construction des matrices du problème d'optimisation quadratique qui cherche à
minimiser le résidu L2. Cette formulation du problème est néanmoins bancale et 
on préferera utiliser "get_matrices_Contrast_deux2" à l'aide de la fonction 
"Solve22"

 """  
def get_matrices_Contrast_deux(problem):
    problem.compute_response_matrices()
    I0 = np.identity(problem.ndz)
    I1 = np.hstack([I0 for k in range(problem.corono.nlam*2)])            
    Q0=np.zeros((problem.npp,problem.npp))
    Q1=np.zeros((problem.npp,problem.ndz))
    Q2=np.zeros((problem.ndz,problem.npp))
    Q3=np.ones((problem.ndz,problem.ndz))
    Q10 = np.concatenate((Q0,Q1), axis=1)
    Q20 = np.concatenate((Q2,Q3), axis=1)
    Q=np.concatenate((Q10,Q20),axis=0)
    A0  = np.concatenate(( problem.corono_field_t, -I1), axis=0)
    A1  = np.concatenate((-problem.corono_field_t, -I1), axis=0)
    ctmp = np.zeros(problem.npp)
    ctmp[:problem.npp] = np.asarray(problem.idx_pup)
    A20 = np.concatenate((np.zeros((problem.npp, problem.ndz)), -I0),axis=0)
    
    
    A21  = np.concatenate(((-1/sum(ctmp)*(ctmp)), 
                              (np.zeros(problem.neps))), axis=0)
    A30=np.identity(problem.npp)
    A31=np.zeros((problem.ndz,problem.npp))
    A3=np.concatenate((A30,A31))
    A4=-A3

    b0  = np.zeros((2*problem.corono.nlam*problem.ndz*2))
    
    b1  = np.zeros(problem.ndz)
    b2  = [-problem.tau]
    b3 = np.ones(problem.npp)
    b4 = np.zeros(problem.npp)
    A = np.concatenate((A0,A1,A20,A21[:,None],A3,A4), axis=1)
    
    b = np.concatenate((b0,b1,b2,b3,b4))
    return A,b,Q
"""   
On résoud ici le problème d'optimisation quadratique L2 posé par
 "get_matrices_Contrast_deux" à l'aide du solver quadprog
Préferez néanmoins "solve22"
"""
def solve2(problem):
    A,b,Q=get_matrices_Contrast_deux(problem)
    Q+=10**-14*np.identity(len(Q))
    solve='quadprog'
    t_start = time()
    x=solve_qp(Q,np.zeros(len(Q)),A.T,b,None,None,solve)
    t_end = time()
    print("Solve time: {0:.2f} [ms]".format(1000. * (t_end - t_start)))
#    problem.Apod[problem.idx_pup]=x[:problem.npp]
#    pl.plot(corono0.r, problem.Apod, label=solve)
#    pl.show()
#    #Display of the intensity profiles of the coronagraphic images'
#    poly_direct_image1 = corono0.compute_direct_intensity_1d(problem.Apod)
#    poly_corono_image1 = corono0.compute_corono_intensity_1d(problem.Apod)
#    pl.axvline(x=rho0,color='red')
#    pl.axvline(x=rho1,color='red')
#    pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),)
#    pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#    pl.xlabel(r'Angular separation in $\lambda_0$/D')
#    pl.ylabel('Normalized intensity in log scale')
#    pl.ylim(1e-13, 1e-4)
#    pl.legend()
#    mono_direct_image1 = corono0.compute_direct_intensity_1d(problem.Apod, poly=False)
#    mono_corono_image1 = corono0.compute_corono_intensity_1d(problem.Apod, poly=False)
#    colors = pl.cm.rainbow(np.linspace(0,1,nlambis))
#
#
##pl.title('Intensity profiles of the coronagraphic images')
##pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
##pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
##pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
#    for i in range(corono0.nlam):
#        pl.semilogy(corono0.xi,mono_corono_image1[i]/mono_direct_image1[(corono0.nlam+1)//2].max(), 
#        label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[i]), color = colors[i])
#        pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#        pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#        pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#        pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#        pl.xlabel(r'Angular separation in $\lambda_0$/D')
#        pl.ylabel('Normalized intensity in log scale')
#        pl.ylim(1e-13, 1e-3)
#        pl.legend()
#        pl.tight_layout()
#    pl.show()
    a=np.zeros_like(problem.Apod)
    a[problem.idx_pup]=x[:problem.npp]
    return a[problem.idx_pup]

"""
Construction des matrices du problème d'optimisation quadratique qui cherche à
minimiser le résidu L2. 
"""
def get_matrices_Contrast_deux2(problem):
    problem.compute_response_matrices()
    Q=np.dot(problem.corono_field_t,problem.corono_field_t.T)
    A1=-np.identity(problem.npp)
    A2=np.identity(problem.npp)
    ctmp = np.zeros(problem.npp)
    ctmp[:problem.npp] = np.asarray(problem.idx_pup)
    ctmp=(-1/sum(ctmp)*(ctmp))
    A = np.concatenate((A1,A2,ctmp[None,:]),axis=0)
    

    b0  = np.zeros(problem.npp)
    
    b1  = np.ones(problem.npp)
    b2  = [-problem.tau]  
    b = np.concatenate((b0,b1,b2))
    return A,b,Q
"""
On résoud ici le problème d'optimisation quadratique L2 posé par
"get_matrices_Contrast_deux2" à l'aide du solver quadprog.
On affiche également l'intensité résiduelle pour différentes longueurs d'onde
"""
def solve22(problem):
    A,b,Q=get_matrices_Contrast_deux2(problem)
#    Q+=10**-18*np.identity(len(Q))
    Valp,Vecp=np.linalg.eigh(Q)
    #On régularise la matrice pour en faire une matrice définie-positive
    Q=np.diag(np.clip(Valp,10**-17,max(Valp)))
    Inv=np.linalg.inv(Vecp.T)
    Q=np.dot(Inv,Q)
    Q=np.dot(Q,Vecp.T)
    solve='quadprog'
    t_start = time()
    x=solve_qp(Q,np.zeros(len(Q)),A,b,None,None,solve)
    t_end = time()
    print("Solve time: {0:.2f} [ms]".format(1000. * (t_end - t_start)))
#    problem.Apod[problem.idx_pup]=x[:problem.npp]
#    pl.plot(corono0.r, problem.Apod,label=solve)
#    pl.show()
#    #Display of the intensity profiles of the coronagraphic images'
#    pl.figure()
#    poly_direct_image1 = corono0.compute_direct_intensity_1d(problem.Apod)
#    poly_corono_image1 = corono0.compute_corono_intensity_1d(problem.Apod)
#    pl.axvline(x=rho0,color='red')
#    pl.axvline(x=rho1,color='red')
#    pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),)
#    pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#    pl.xlabel(r'Angular separation in $\lambda_0$/D')
#    pl.ylabel('Normalized intensity in log scale')
#    pl.ylim(1e-13, 1e-4)
#    pl.legend()
#    mono_direct_image1 = corono0.compute_direct_intensity_1d(problem.Apod, poly=False)
#    mono_corono_image1 = corono0.compute_corono_intensity_1d(problem.Apod, poly=False)
#    colors = pl.cm.rainbow(np.linspace(0,1,nlambis))
#
#
##pl.title('Intensity profiles of the coronagraphic images')
##pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
##pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
##pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
#    for i in range(corono0.nlam):
#        pl.semilogy(corono0.xi,mono_corono_image1[i]/mono_direct_image1[(corono0.nlam+1)//2].max(), 
#        label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[i]), color = colors[i])
#        pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#        pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#        pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#        pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#        pl.xlabel(r'Angular separation in $\lambda_0$/D')
#        pl.ylabel('Normalized intensity in log scale')
#        pl.ylim(1e-13, 1e-3)
#        pl.legend()
#        pl.tight_layout()
#    pl.show()
    a=np.zeros_like(problem.Apod)
    a[problem.idx_pup]=x[:problem.npp]
    return a[problem.idx_pup]

"""          
On reformule le problème d'optimisation quadratique en imposant de chercher les
vecteurs solutions sur un certains sous espace vectorielle. On réduit ainsi
la dimensionalité du problème. Notez que l et u représente le range de l'espace de
vecteurs propres dans lequel on recherche la solution. Vecp et Valp contiennent 
respectivement les valeurs propres (classé dans l'ordre croissant) 
et les vecteurs propres associé a ces valeurs propres de la matrice Q du problème 
d'optimisation quadratique
"""
def get_matrices_reduced(problem,l,u):
    A,b,Q=get_matrices_Contrast_deux2(problem)
    Valp,Vecp=np.linalg.eigh(Q)
    Q=np.diag(np.clip(Valp,min(abs(Valp)),max(Valp))[l:u])
    Phi=np.ones((len(Vecp),len(Q)))
    for i in range(len(Q)):
        Phi[:,i]=Vecp[:,l+i]
    A=np.dot(A,Phi)
    return A,b,Q,Phi
"""
On résoud ici le problème d'optimisation quadratique L2 posé par
 "get_matrices_reduced" à l'aide du solver quadprog.
On affiche également l'intensité résiduelle pour différentes longueurs d'onde
"""
def solve_reduced(problem,l,u):
    A,b,Q,Phi=get_matrices_reduced(problem,l,u)
    solve='quadprog'
    t_start = time()
    beta=solve_qp(Q,np.zeros(len(Q)),A,b,None,None,solve)
    t_end = time()
    print("Solve time: {0:.2f} [ms]".format(1000. * (t_end - t_start)))
    x=np.dot(Phi,beta)
    problem.Apod[problem.idx_pup]=x[:problem.npp]
#    pl.plot(corono0.r, problem.Apod/np.max(problem.Apod), label=solve)
#    pl.show()
#    #Display of the intensity profiles of the coronagraphic images'
#    poly_direct_image1 = corono0.compute_direct_intensity_1d(problem.Apod)
#    poly_corono_image1 = corono0.compute_corono_intensity_1d(problem.Apod)
#    pl.axvline(x=rho0,color='red')
#    pl.axvline(x=rho1,color='red')
#    pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),)
#    pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#    pl.xlabel(r'Angular separation in $\lambda_0$/D')
#    pl.ylabel('Normalized intensity in log scale')
#    pl.ylim(1e-13, 1e-4)
#    pl.legend()
#    mono_direct_image1 = corono0.compute_direct_intensity_1d(problem.Apod, poly=False)
#    mono_corono_image1 = corono0.compute_corono_intensity_1d(problem.Apod, poly=False)
#    colors = pl.cm.rainbow(np.linspace(0,1,nlambis))
#
#
##pl.title('Intensity profiles of the coronagraphic images')
##pl.semilogy(corono0.xi,poly_direct_image1/poly_direct_image1.max(),label='Direct')
##pl.semilogy(corono0.xi,poly_direct_image2/poly_direct_image2.max(),label='Direct')
##pl.semilogy(corono0.xi,poly_direct_image3/poly_direct_image3.max(),label='Direct')
#    for i in range(corono0.nlam):
#        pl.semilogy(corono0.xi,mono_corono_image1[i]/mono_direct_image1[(corono0.nlam+1)//2].max(), 
#        label=r'{0:.2f}$\lambda_0$'.format(corono0.lam_t[i]), color = colors[i])
#        pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
#        pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#        pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
#        pl.axhline(10**(-cDarkHole), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--')
#        pl.xlabel(r'Angular separation in $\lambda_0$/D')
#        pl.ylabel('Normalized intensity in log scale')
#        pl.ylim(1e-13, 1e-3)
#        pl.legend()
#        pl.tight_layout()
#    pl.show()
    return problem.Apod[problem.idx_pup] 
            
    

def compute_matrix_planet(problem):
    
    direct_field_t_tmp = np.zeros((problem.npp, problem.corono.nlam, problem.corono.nImg+1), dtype='complex128')
    Apod1d    = np.zeros((problem.corono.nPup))
    for i, val in enumerate(problem.idx_pup):
        Apod1d[val] = 1
        direct_field_t_tmp[i] = problem.corono.lam0/problem.corono.lam_t[:,None]*np.pi*problem.corono.hankel_kernel_all.dot(
        problem.Pupil1d*Apod1d*problem.LyotStop1d*problem.r/problem.R)*problem.R/problem.nPup
        Apod1d[val] = 0            
        problem.direct_field_re_t_tmp = direct_field_t_tmp.real
        problem.direct_field_im_t_tmp = direct_field_t_tmp.imag
        direct_field_re_t = np.reshape(problem.direct_field_re_t_tmp[:,:,problem.idx_dz], (problem.npp, problem.corono.nlam*problem.ndz))

        #direct_field_im_t = np.reshape(
                #problem.direct_field_im_t_tmp[:,:,problem.idx_dz], 
                #(problem.npp, problem.corono.nlam*problem.ndz))
    return direct_field_re_t,np.dot(direct_field_re_t,direct_field_re_t.T)
#    return(np.concatenate((direct_field_re_t,direct_field_im_t),axis=1)      )

def get_matrices_ratio(problem):
    A,b,K1=get_matrices_Contrast_deux2(problem)
    K2=compute_matrix_planet(problem)[1]
    Valp1,Vecp1=np.linalg.eigh(K1)
    Valp2,Vecp2=np.linalg.eigh(K2)
    K1=np.diag(np.clip(Valp1,10**-32,max(Valp1)))
    Inv1=np.linalg.inv(Vecp1.T)
    K1=np.dot(Inv1,K1)
    K1=np.dot(K1,Vecp1.T)
    K2=np.diag(np.sqrt(np.clip(Valp2,0,max(Valp2))))
    Inv2=np.linalg.inv(Vecp2.T)
    K2=np.dot(Inv2,K2)
    K2=np.dot(K2,Vecp2.T)
    Q=np.dot(np.dot(K2,K1),K2)
    Valp,Vecp=np.linalg.eigh(Q)
    Q=np.diag(np.clip(Valp,10**-14,max(Valp)))
    Inv=np.linalg.inv(Vecp.T)
    Q=np.dot(Inv,Q)
    Q=np.dot(Q,Vecp.T)
    return (A,b,Q,K2)

def solve_ratio(problem):
    A,b,Q,K2=get_matrices_ratio(problem)
    A=np.dot(A,K2)
    solve='quadprog'
    t_start = time()
    y=solve_qp(Q,np.zeros(len(Q)),A,b,None,None,solve)
    t_end = time()
    print("Solve time: {0:.2f} [ms]".format(1000. * (t_end - t_start)))
    x=np.dot(K2,y)
    a=np.zeros_like(problem.Apod)
    a[problem.idx_pup]=x[:problem.npp]
    pl.plot(corono0.r, a,label=solve)
    pl.show()
    return a[problem.idx_pup]


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


def grad(w,Ke,Kp): return (2*np.dot(Ke,w)*np.dot(np.dot(w,Kp),w)-2*np.dot(Kp,w)*np.dot(np.dot(w,Ke),w))\
/(np.dot(np.dot(w,Kp),w))**2

def fonc(w,Ke,Kp): return np.dot(np.dot(w,Ke),w)/np.dot(np.dot(w,Kp),w)

def solve_C(g,A,b,bds):
    sol=scipy.optimize.linprog(g,A.T,b,method='interior-point', bounds=bds, options={'sparse':False})
    return sol.x


def gradp(w,Pe,Pp): return (2*np.dot(Pe,w)*np.dot(np.dot(w,Pp),w)-2*np.dot(Pp,w)*np.dot(np.dot(w,Pe),w))\
/(np.dot(np.dot(w,Pp),w))**2

def foncp(w,Pe,Pp): return np.dot(np.dot(w,Pe),w)/np.dot(np.dot(w,Pp),w)

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
        
        

def solve_frank_wolfe(problem, init='Linf', nmax=10000000,gradmin=1e-9,solve=1,linesearch=0,x=[]):
    x0=np.zeros_like(problem.Apod)
    ctmp = np.zeros(problem.npp)
    ctmp[:problem.npp] = np.asarray(problem.idx_pup)
    ctmp=(1/sum(ctmp)*(ctmp)) 
    t_0 = time()
    if init =='L2':
        x0[problem.idx_pup]=solve22(problem)[:problem.npp]
        x0=x0[problem.idx_pup]
    elif init =='L1':
        x0[problem.idx_pup]=solve1(problem)[:problem.npp]
        x0=x0[problem.idx_pup]

    elif init=='unif':
       x0[problem.idx_pup]=(problem.tau)*(np.ones(problem.npp))
       x0=x0[problem.idx_pup]
       
    elif init=='Linf':
        x0[problem.idx_pup]=solveinf(problem)[:problem.npp]
        x0=x0[problem.idx_pup]
    elif init=='Random':
        l=[i for i in range (len(problem.idx_pup))]
        l=np.random.permutation(l)
        x0=x0[problem.idx_pup]
        x1=np.ones_like(x0)
        k=-1
        while np.dot(ctmp,x1)>problem.tau and k<len(x0)-1:
            k+=1
            x0[l[k]]=random.random()
            x1[l[k]]=x0[l[k]]
        if np.dot(ctmp,x1)<problem.tau:
            x0=x1
            x0[l[k]]=0
            x0[l[k]]=(tau-np.dot(ctmp,x0))/ctmp[l[k]]
            
        
    elif init=='Manual':
        x0[problem.idx_pup]=x
        x0=x0[problem.idx_pup]

       
  
    A,b,Ke=get_matrices_Contrast_deux2(problem)

    Kp=compute_matrix_planet(problem)[1]

    grad1=lambda x:grad(x,Ke,Kp)
    fonc1 =lambda x:fonc(x,Ke,Kp)
    if solve==1:
        solve_C1=lambda x,g:solve_closed_form(g,ctmp,problem.tau)
    else:
        A=A[-1,:][None,:]
        b=b[-1]
        bds = np.zeros((len(x0), 2))
        bds[:,1] = 1.
        solve_C1=lambda x,g:solve_C(g,A.T,b,bds)
    params = dict()
    params['nbitermax'] = nmax
    params['stopvarj'] = gradmin
    params['verbose'] = True
    params['log'] = True
    t_start = time()

    x, val, log = fmin_cond(fonc1, grad1, solve_C1, x0, Ke, Kp,linesearch, **params)
    a=np.zeros_like(problem.Apod)
    a[problem.idx_pup]=x[:problem.npp]
    t_end = time()
    print("Solve time: {0:.2f} [ms]".format(1000. * (t_end - t_start)))
    return a[problem.idx_pup],t_end - t_start,t_end - t_0, log


def calcul_snr(problem,Apod):
    g=np.dot(compute_matrix_planet(problem)[0].T,Apod)
    h=np.dot(problem.corono_field_t.T,Apod)
    a=np.linalg.norm(g, 2)/np.linalg.norm(h, 2)
    
    return a**2
#    return "Le rapport signal sur bruit dans la dark zone est de {0:.2f} ".format(a**2)

def square(list):
    return [i ** 2 for i in list]


def analyse_apod(corono0,problem,Apod,color0='r',l=''):
    
    #Plot de l'apodiseur
#    pl.figure(1)
#    pl.clf()
#    pl.plot(corono0.r, Apod_pyth)
#    pl.xlabel(r'Pupil radius r')
#    pl.ylabel('Apodizer amplitude transmission')
#    pl.legend()
#    pl.show()
    
    #Compute electric field
    Apod_pyth=np.zeros_like(problem.Apod)
    Apod_pyth[len(Apod_pyth)-problem.npp:]=Apod[:]
    poly_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth)
    poly_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth)

    mono_direct_image1 = corono0.compute_direct_intensity_1d(Apod_pyth, poly=False)
    mono_corono_image1 = corono0.compute_corono_intensity_1d(Apod_pyth, poly=False)
    
    #Plot réponse de l'apodiseur
    pl.figure(2)
    pl.title(r'Intensity profiles of the coronagraphic images for $\tau$ = 0.5')
    pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label=l,color=color0)
    pl.axvline(x=corono0.rMask, ymin=-12, ymax =2, linewidth=1, color='r', linestyle='--')
    pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
    pl.xlabel(r'Angular separation in $\lambda_0$/D')
    pl.ylabel('Normalized intensity in log scale')
    pl.ylim(1e-12, 1e-3)
    pl.legend()
    return




#%%
#pl.figure()
#pl.title("Résultats optimisation pour Tau=0.75")
#pl.plot(apodr[14],'+y' , label=r'Frank Wolfe convergé')
#pl.plot(apod1000,'g' , label='Frank Wolfe 1000 itérations')
##pl.plot(apodl2,'--' , label=r'Min $L_2$ classique')
#pl.legend(loc='best')
Ke=get_matrices_Contrast_deux2(problem2)[2]
Valp,Vecp=np.linalg.eigh(Ke)
index_min = np.argmin(abs(Valp))
x = range(len(Valp))

pl.figure()
pl.ylim(bottom=10**-24)
pl.axhline(10**(-16), xmin=corono0.xi.min(), xmax=corono0.xi.max(), linewidth=1, color='k', linestyle='--',label='kernel')
pl.title('Characterization of the eigenvalues of the matrix K ')
pl.ylabel('Modulus of the eigenvalues of the matrix K')
pl.semilogy(x[index_min:],abs(Valp)[index_min:],label='Positive eigenvalues')
pl.semilogy(x[:index_min],abs(Valp)[:index_min],label='Negative eigenvalues')
pl.legend()
pl.grid()
pl.show()
pl.savefig('C:/Users/adamh/OneDrive/Bureau/Stage github/Coronagraphs/imgs/repartition_vp_tau_05.pdf',bbox_inches='tight')

#%% Temps de calcul
fichier = np.loadtxt("C:/Users/adamh/OneDrive/Bureau/Stage github/Coronagraphs/données/Temps_calcul_tau_07.txt")
L=fichier[:,0]
t00=fichier[:,1]
t10=fichier[:,2]
#t20=fichier[:,3]
#t30=fichier[:,4]


fig=pl.figure()
fig.suptitle("computation time depending on the pupil shape", fontsize=12)
pl.semilogy(L, t00,'-b',ls='--', label=r'Basic solver')
pl.semilogy(L, t10, '-y',ls='--', label=r'Closed-form solver')
pl.ylabel('computation time (s)', fontsize=10 )
pl.xlabel(r'$N_{pup}$', fontsize=10)
#pl.semilogy(L, t20,'-b', label=r' calculation time with basic solver and closed-form linesearch')
#pl.semilogy(L, t30,'-y', label=r' calculation time with closed-form linesearch and solver')
pl.legend()
pl.grid(b=True, which='major')
pl.grid(b=True, which='minor',linestyle='--')
pl.savefig('imgs/Comput_time_solver.pdf',bbox_inches='tight')


#%% Test valeur propre et vecteur propre
Valp=np.loadtxt("données/Valp_tau_05.txt")
Vecp=np.loadtxt("données/Vecp_tau_05.txt")


pl.figure()
pl.ylim(bottom=10**-1000)
pl.title('Modules des valeurs propres de la matrice K')
pl.semilogy(abs(Valp))
pl.grid()
pl.show()
pl.figure()
pl.title("Illustration des transformées de Fourier des vecteurs propres")
pl.semilogy(sorted(np.fft.fftfreq(len(Vecp[:,1]))),np.abs(np.fft.fftshift(np.fft.fft(Vecp[:,200]))), label='Exemple pour un vecteur propre appartenant au noyau')
pl.grid()
pl.semilogy(sorted(np.fft.fftfreq(len(Vecp[:,1]))),np.abs((np.fft.fftshift(np.fft.fft(Vecp[:,-1])))), label='Exemple pour un vecteur propre hors du noyau')
pl.grid()
pl.legend(loc='upper right')
pl.show()
pl.figure()
pl.title("Exemple de vecteur propre appartenant au noyau")
pl.plot(Vecp[:,1][0:-1])
pl.grid()
pl.show()
pl.figure()
pl.title("Exemple de vecteur propre hors du noyau")
pl.plot(Vecp[:,-1][0:-1])
pl.grid()
pl.show()



M=np.zeros((len(Valp)//2,len(Valp)))
for i in range (len(Valp)):
    M[:,i]=np.abs(np.fft.fftshift(np.fft.fft(Vecp[:,i])))[len(Valp)//2:len(Valp)]
pl.figure()
pl.title('Repartition spectrale de la puissance')
pl.imshow(M**2)
 

#%% Test noyau selon largeur de bande
pl.figure()
AllValp=np.loadtxt("données/Valp_tau_05_selon_bw.txt")
n=5
colors = pl.cm.rainbow(np.linspace(0,0.5,n))
for k in range (n):
    bw=(k+1)*.3
    pl.semilogy(abs(AllValp[k]),label='largeur de bande de {0:.1f}'.format((k+1)*0.3), color = colors[k])
    pl.axhline(10**-16,color='r',linestyle='--')
pl.legend(loc='upper left')
pl.grid()
pl.show()

