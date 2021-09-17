#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:23:07 2019
Author: Adam Hessas <adam.hessas@ecl17.ec-lyon.fr> (https://github.com/astromam)

License: MIT license
"""
import corono as coro
import numpy as np
import os

"""
Check if the directory to save all the data exists. If not, creates it.
"""

if not os.path.exists('../../../data/1D/Lp norms and SNR optim'):
        os.mkdir('../../../data/1D/Lp norms and SNR optim')
        
#%%        
"""
Parameters
"""
nlambis=11
corono_name  = 'APLC' # 'APLC' or 'SP'
problem_name = 'MaxContrastL1' # ,'MaxTau' # 'MaxContrastLinf' #'MaxContrastL1' 
#'MaxContrastL2'
solver       = 'gurobipy' # 'stdgrb', 'gurobipy', 'scipy.linprog'
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
nlam = 5

PupilID    = 0.15
rMask       = 4

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75
LyotStopID = 0.30
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

#Number of simulations (i.e. number of \tau)
nsim=20
 
"""
Compute the L1 optimization solutions and the norms of the associated residual
"""
#Create the list of the L1 residual for the various apodizers generated
L11=[]
#Create the list of the L2 residual for the various apodizers generated
L12=[]
#Create the list of the Linf residual for the various apodizers generated
L1inf=[]
#Create the list of the SNR for the various apodizers generated
L1r=[]
#Create the array that contains every solution apodizers for the L1 problem
apod1=np.zeros((nsim-1,nPup))

for k in range (nsim-1):
    tau   = (k+1)*1/(nsim)
    params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, problem_name = 'MaxContrastL1',
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
    corono0 = coro.design.APLC1d(**params)
    problem = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1', **params)
    
    #Solve the problem for the current tau and save the solution
    a=problem.solve_model()
    apod1[k]=a
    
    #Compute the direct and coronagraphic response matrices by using the 
    #function compute_response_matrices() for the MaxSNR problem
    params['problem_name']='MaxSNR'
    problem = coro.optim_1d.MaxSNR(corono=corono0,nmax=200000,gradmin=1e-9, initialisation='L1' ,**params)
    problem.compute_response_matrices()
    
    #Compute the electric field and the direct electric field residual in the 
    #dark zone by doing the matrix products
    h=np.dot(problem.corono_field_t.T,a[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,a[problem.idx_pup]) 
    
    #Save the L1, L2 and Linf residual as well as the SNR

    L11.append(sum([abs(x) for x in h]))
    L12.append(np.linalg.norm(h, 2))
    L1inf.append(max(h))
    
    #It is squared because of the fact that the SNR is the squared ratio of the L2 norm
    # of the direct residual divided by the coronographic residual 
    L1r.append((np.linalg.norm(g, 2)/np.linalg.norm(h, 2))**2)
#%%

"""
Compute the L_2 optimization solutions and the norms of the associated residual
""" 


#Create the list of the L1 residual for the various apodizers generated
L21=[]
#Create the list of the L2 residual for the various apodizers generated
L22=[]
#Create the list of the Linf residual for the various apodizers generated
L2inf=[]
#Create the list of the SNR for the various apodizers generated
L2r=[]
#Create the array that contains every solution apodizers for the L2 problem
apod2=np.zeros((nsim-1,nPup))

for k in range (nsim-1):
    tau   = (k+1)*1/(nsim)

    params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver,problem_name = 'MaxContrastL2',
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
    corono0 = coro.design.APLC1d(**params)
    problem = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L2', **params)
    
    #Solve the problem for the current tau and save the solution
    a=problem.solve_model()
    apod2[k]=a
    
    #Compute the direct and coronagraphic response matrices by using the 
    #function compute_response_matrices() for the MaxSNR problem
    params['problem_name']='MaxSNR'
    problem = coro.optim_1d.MaxSNR(corono=corono0,nmax=200000,gradmin=1e-9, initialisation='L1' ,**params)
    problem.compute_response_matrices()
   
    #Compute the electric field and the direct electric field residual in the 
    #dark zone by doing the matrix products
    h=np.dot(problem.corono_field_t.T,a[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,a[problem.idx_pup])
    
    #Save the L1, L2 and Linf residual as well as the SNR
    
    L21.append(sum([abs(x) for x in h]))
    L2inf.append(max(h))
    L22.append(np.linalg.norm(h, 2))
    #It is squared because of the fact that the SNR is the squared ratio of the L2 norm
    # of the direct residual divided by the coronographic residual 

    L2r.append((np.linalg.norm(g, 2)/np.linalg.norm(h, 2))**2)
    
#%%
    
"""
Compute the L_inf optimization solutions and the norms of the associated residual
""" 
#Create the list of the L1 residual for the various apodizers generated
Linf1=[]
#Create the list of the L2 residual for the various apodizers generated
Linf2=[]
#Create the list of the Linf residual for the various apodizers generated
Linfinf=[]
#Create the list of the SNR for the various apodizers generated
Linfr=[]
#Create the array that contains every solution apodizers for the Linf problem
apodinf=np.zeros((nsim-1,nPup))

for k in range (nsim-1):
    tau   = (k+1)*1/(nsim)

    params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver, problem_name = 'MaxContrastLinf',
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
    corono0 = coro.design.APLC1d(**params)
    problem = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='Linf', **params)
    
    #Solve the problem for the current tau and save the solution
    a=problem.solve_model()
    apodinf[k]=a

    #Compute the direct and coronagraphic response matrices by using the 
    #function compute_response_matrices() for the MaxSNR problem
    params['problem_name']='MaxSNR'
    problem = coro.optim_1d.MaxSNR(corono=corono0,nmax=200000,gradmin=1e-9, initialisation='L1' ,**params)
    problem.compute_response_matrices()
    
    #Compute the electric field and the direct electric field residual in the 
    #dark zone by doing the matrix products
    h=np.dot(problem.corono_field_t.T,a[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,a[problem.idx_pup])
    
    #Save the L1, L2 and Linf residual as well as the SNR

    Linf1.append(sum([abs(x) for x in h]))
    Linf2.append(np.linalg.norm(h, 2))
    Linfinf.append(max(h))
    
    #It is squared because of the fact that the SNR is the squared ratio of the L2 norm
    # of the direct residual divided by the coronographic residual 
    Linfr.append((np.linalg.norm(g, 2)/np.linalg.norm(h, 2))**2)

    
#%%
"""
Compute the Frank-Wolfe optimizations initialized by the L_1 solutions and the norms of the associated residual
"""
#Create the list of the L1 residual for the various apodizers generated
Lr11=[]
#Create the list of the L2 residual for the various apodizers generated
Lr12=[]
#Create the list of the Linf residual for the various apodizers generated
Lr1inf=[]
#Create the list of the SNR for the various apodizers generated
Lr1r=[]
#Create the array that contains every solution apodizers of the FW algorithm
#initialized with the L1 solution
apodr1=np.zeros((nsim-1,nPup))

for k in range (nsim-1):
    tau   = (k+1)*1/(nsim)

    params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver,problem_name = 'MaxSNR',
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
    corono0 = coro.design.APLC1d(**params)
    #Solve the problem for the current tau and save the solution. Tau is printed too
    #so the user can folow the evolution of the algorithm

    problem = coro.optim_1d.MaxSNR(corono=corono0,nmax=200000,gradmin=1e-9, initialisation='L1' ,**params)
    print ("Tau ={0}".format(tau))
    a=problem.solve_model()
    apodr1[k]=a
    
    #Compute the electric field and the direct electric field residual in the 
    #dark zone by doing the matrix products
    problem.compute_response_matrices()
    h=np.dot(problem.corono_field_t.T,a[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,a[problem.idx_pup])

    #Save the L1, L2 and Linf residual as well as the SNR
    Lr11.append(sum([abs(x) for x in h])) 
    Lr12.append(np.linalg.norm(h, 2))
    Lr1inf.append(max(h))
    
    #It is squared because of the fact that the SNR is the squared ratio of the L2 norm
    # of the direct residual divided by the coronographic residual 
    Lr1r.append((np.linalg.norm(g, 2)/np.linalg.norm(h, 2))**2)

#%%
"""
Compute the Frank-Wolfe optimizations initialized by the L_2 solutions and the norms of the associated residual
""" 
#Create the list of the L1 residual for the various apodizers generated
Lr1=[]
#Create the list of the L2 residual for the various apodizers generated
Lr2=[]
#Create the list of the Linf residual for the various apodizers generated
Lrinf=[]
#Create the list of the SNR for the various apodizers generated
Lrr=[]
#Create the array that contains every solution apodizers of the FW algorithm
#initialized with the L2 solution
apodr=np.zeros((nsim-1,nPup))

for k in range (nsim-1):
    tau   = (k+1)*1/(nsim)

    params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver,problem_name = 'MaxSNR',
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
    corono0 = coro.design.APLC1d(**params)
    #Solve the problem for the current tau and save the solution. Tau is printed too
    #so the user can folow the evolution of the algorithm

    problem = coro.optim_1d.MaxSNR(corono=corono0,nmax=200000,gradmin=1e-9, initialisation='L2' ,**params)
    print ("Tau ={0}".format(tau))
    a=problem.solve_model()
    apodr[k]=a
    
    #Compute the electric field and the direct electric field residual in the 
    #dark zone by doing the matrix products
    problem.compute_response_matrices()
    h=np.dot(problem.corono_field_t.T,a[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,a[problem.idx_pup])
    
    #Save the L1, L2 and Linf residual as well as the SNR
    Lr1.append(sum([abs(x) for x in h]))
    Lr2.append(np.linalg.norm(h, 2))
    Lrinf.append(max(h))
    
    #It is squared because of the fact that the SNR is the squared ratio of the L2 norm
    # of the direct residual divided by the coronographic residual 
    Lrr.append((np.linalg.norm(g, 2)/np.linalg.norm(h, 2))**2)
#%%
"""
Compute the Frank-Wolfe optimizations initialized by the L_inf solutions and the norms of the associated residual&
"""
#Create the list of the L1 residual for the various apodizers generated
Lrinf1=[]
#Create the list of the L2 residual for the various apodizers generated
Lrinf2=[]
#Create the list of the Linf residual for the various apodizers generated
Lrinfinf=[]
#Create the list of the SNR for the various apodizers generated
Lrinfr=[]
#Create the array that contains every solution apodizers of the FW algorithm
#initialized with the Linf solution
apodrinf=np.zeros((nsim-1,nPup))

for k in range (nsim-1):
    tau   = (k+1)*1/(nsim)

    params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
                 bw = bw, nlam = nlam,
                 PupilID = PupilID, rMask = rMask, 
                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
                 LyotStopID = LyotStopID,
                 LyotStopOD = LyotStopOD,
                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
                 solver = solver,problem_name = 'MaxSNR',
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
    corono0 = coro.design.APLC1d(**params)
   
    #Solve the problem for the current tau and save the solution. Tau is printed too
    #so the user can folow the evolution of the algorithm
    problem = coro.optim_1d.MaxSNR(corono=corono0,nmax=200000,gradmin=1e-9, initialisation='Linf' ,**params)
    print ("Tau ={0}".format(tau))
    a=problem.solve_model()
    apodrinf[k]=a
    
    #Compute the electric field and the direct electric field residual in the 
    #dark zone by doing the matrix products
    problem.compute_response_matrices()
    h=np.dot(problem.corono_field_t.T,a[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,a[problem.idx_pup])
    
    #Save the L1, L2 and Linf residual as well as the SNR
    Lrinf1.append(sum([abs(x) for x in h]))
    Lrinf2.append(np.linalg.norm(h, 2))
    Lrinfinf.append(max(h))
    
    #It is squared because of the fact that the SNR is the squared ratio of the L2 norm
    # of the direct residual divided by the coronographic residual 
    Lrinfr.append((np.linalg.norm(g, 2)/np.linalg.norm(h, 2))**2)





#%%
"""
Built the vector of the various tau for which the optimization has been made
"""
x=np.linspace(1/nsim,1-1/nsim,nsim-1)
    
'''
Save the residuals 
'''
#Save the resiudals achieved with L1 solutions
data = np.array([x,L11,L1inf,L12,L1r])
data = data.T
datafile_path = "../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_1.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, data, fmt='%.10f')
    
#Save the resiudals achieved with L2 solutions
data = np.array([x,L21,L2inf,L22,L2r])
data = data.T
datafile_path = "../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_22.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, data, fmt='%.10f')

#Save the resiudals achieved with Linf solutions
data = np.array([x,Linf1,Linfinf,Linf2,Linfr])
data = data.T
datafile_path = "../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_inf.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, data, fmt='%.10f')
    
#Save the resiudals achieved with the Frank-Wolfe algorithm initialized by the L1 solution
data = np.array([x,Lr11,Lr1inf,Lr12,Lr1r])
data = data.T
datafile_path = "../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_ratio_init_L1.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, data, fmt='%.10f')
    
#Save the resiudals achieved with the Frank-Wolfe algorithm initialized by the L2 solution
data = np.array([x,Lr1,Lrinf,Lr2,Lrr])
data = data.T
datafile_path = "../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_ratio_init_L2.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, data, fmt='%.10f')

#Save the resiudals achieved with the Frank-Wolfe algorithm initialized by the Linf solution
data = np.array([x,Lrinf1,Lrinfinf,Lrinf2,Lrinfr])
data = data.T
datafile_path = "../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_ratio_init_Linf.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, data, fmt='%.10f')

    
'''
Save the solution apodizers for each minimization method 
'''
#Save the apodizers solutions of the L1 problem
datafile_path = "../../../data/1D/Lp norms and SNR optim/apod_l1.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(apod1).T, fmt='%.10f')

#Save the apodizers solutions of the L2 problem
datafile_path = "../../../data/1D/Lp norms and SNR optim/apod_l2.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(apod2).T, fmt='%.10f')
    
#Save the apodizers solutions of the Linf problem
datafile_path = "../../../data/1D/Lp norms and SNR optim/apod_linf.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(apodinf).T, fmt='%.10f')
    
#Save the apodizers reached with the Frank-Wolfe algorithm initialized with the L1 solution
datafile_path = "../../../data/1D/Lp norms and SNR optim/apod_fw_initL1.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(apodr1).T, fmt='%.10f')
    
#Save the apodizers reached with the Frank-Wolfe algorithm initialized with the L2 solution
datafile_path = "../../../data/1D/Lp norms and SNR optim/apod_fw_initL2.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(apodr).T, fmt='%.10f')

#Save the apodizers reached with the Frank-Wolfe algorithm initialized with the Linf solution
datafile_path = "../../../data/1D/Lp norms and SNR optim/apod_fw_initLinf.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(apodrinf).T, fmt='%.10f')




#%%
"""
Compute the evolution of the intensiity residual in the dark zone for the
L1,L2 and Linf solutions of the optimization problem
"""

#design the corono 
nlambis = 11
nImgbis = 110
Fmaxbis = 11    
params2    = coro.update_params(params, nlam=nlambis, nImg=nImgbis, Fmax=Fmaxbis) 
corono0 = coro.design.APLC1d(**params2)

#Create the array that contains every residual in the dark zone for the L1, L2 and Linf 
#solutions

residual_1=np.zeros((nsim-1,len(corono0.xi)))
residual_2=np.zeros((nsim-1,len(corono0.xi)))
residual_inf=np.zeros((nsim-1,len(corono0.xi)))

#Compute every residual in the dark zone
for i in range (nsim-1):
    #Residual for the L1 solution
    poly_direct_image1 = corono0.compute_direct_intensity_1d(apod1[i])
    poly_corono_image1 = corono0.compute_corono_intensity_1d(apod1[i])
    residual_1[i]=poly_corono_image1/poly_direct_image1.max()
    
    #Residual for the L2 solution
    poly_direct_image1 = corono0.compute_direct_intensity_1d(apod2[i])
    poly_corono_image1 = corono0.compute_corono_intensity_1d(apod2[i])
    residual_2[i]=poly_corono_image1/poly_direct_image1.max()
    
    #Residual for the Linf solution
    poly_direct_image1 = corono0.compute_direct_intensity_1d(apodinf[i])
    poly_corono_image1 = corono0.compute_corono_intensity_1d(apodinf[i])
    residual_inf[i]=poly_corono_image1/poly_direct_image1.max()

'''
Save the residual for each minimization method as well as the vector
corono0.xi
'''
#Save the residual in the dark zone for the L1 problem
datafile_path = "../../../data/1D/Lp norms and SNR optim/residual_dz_l1.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(residual_1).T, fmt='%.20f')

#Save the residual in the dark zone for the L2 problem
datafile_path = "../../../data/1D/Lp norms and SNR optim/residual_dz_l2.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(residual_2).T, fmt='%.20f')
    
#Save the residual in the dark zone for the Linf problem
datafile_path = "../../../data/1D/Lp norms and SNR optim/residual_dz_linf.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(residual_inf).T, fmt='%.20f')

#Save corono0.xi, the x-axis vector of the image plane
datafile_path = "../../../data/1D/Lp norms and SNR optim/corono_xi.txt"
with open(datafile_path, 'w+') as datafile_id:
    np.savetxt(datafile_id, np.array(corono0.xi), fmt='%.20f')


