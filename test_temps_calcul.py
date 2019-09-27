#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:58:40 2019

@author: ahessas
"""

import corono as coro
import numpy as np
import Optimquad as opt
import pylab as pl



"""
Parameters
"""
nlambis=11
corono_name  = 'APLC' # 'APLC' or 'SP'
problem_name = 'MaxContrastL1' # ,'MaxTau' # 'MaxContrastLinf' #'MaxContrastL1' 
#'MaxContrastL2'
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
nlam = 5

PupilID    = 0.2 
rMask       = 4.0

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75
LyotStopID = 0.40
LyotStopOD = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 4
rho1 = 10

# contrast in the dark region
cDarkHole = 9.0

# tau (integrated Pupil transmission)
tau   = 0.7
r   = np.arange(nPup)*R/nPup + R/(2*nPup)
Pupil1d      = (r>PupilID)*1.0
LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0

L=[10,30,100,300,500,1000]
#
#t00=[]
#t01=[]
#L0=[]
#for k in L:
#    nPup=k
#    r   = np.arange(nPup)*R/nPup + R/(2*nPup)
#    Pupil1d      = (r>PupilID)*1.0
#    LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0
#    params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
#                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
#                 bw = bw, nlam = nlam,
#                 PupilID = PupilID, rMask = rMask, 
#                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
#                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
#                 LyotStopID = LyotStopID,
#                 LyotStopOD = LyotStopOD,
#                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
#                 solver = solver, problem_name = 'MaxContrastL1',
#                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
#                 slvCrossover = slvCrossover, slvMethod = slvMethod,
#                 allLogToConsole = allLogToConsole,
#                 FirstDer = FirstDer, SecondDer = SecondDer,
#                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
#                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
#    corono0 = coro.design.APLC1d(**params)
#    problem = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1', **params)
#    f,t0,t1=opt.solve_frank_wolfe(problem,init='Linf', nmax=10000000,gradmin=2e-9,solve=0,linesearch=0)
#    t00.append(t0)
#    t01.append(t1)
#    L0.append(opt.calcul_snr(problem,f))
#
#
#t10=[]
#t11=[]
#L1=[]
#
#for k in L:
#    nPup=k
#    r   = np.arange(nPup)*R/nPup + R/(2*nPup)
#    Pupil1d      = (r>PupilID)*1.0
#    LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0
#    params = coro.to_dict(rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau,
#                 nPup = nPup, nFPM=nFPM, nImg=nImg, Fmax = Fmax,
#                 bw = bw, nlam = nlam,
#                 PupilID = PupilID, rMask = rMask, 
#                 rMask1 = rMask1, rMask2 = rMask2, rMask3 = rMask3,
#                 OPDx2 = OPDx2, OPDx3 = OPDx3, 
#                 LyotStopID = LyotStopID,
#                 LyotStopOD = LyotStopOD,
#                 r = r, R=R, Pupil1d = Pupil1d, LyotStop1d = LyotStop1d,
#                 solver = solver, problem_name = 'MaxContrastL1',
#                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
#                 slvCrossover = slvCrossover, slvMethod = slvMethod,
#                 allLogToConsole = allLogToConsole,
#                 FirstDer = FirstDer, SecondDer = SecondDer,
#                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
#                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
#    corono0 = coro.design.APLC1d(**params)
#    problem = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1', **params)
#    f,t0,t1=opt.solve_frank_wolfe(problem,init='Linf', nmax=10000000,gradmin=2e-9,solve=1,linesearch=0)
#    t10.append(t0)
#    t11.append(t1)
#    L1.append(opt.calcul_snr(problem,f))
#    
t20=[]
t21=[]
L2=[]

for k in L:
    nPup=k
    r   = np.arange(nPup)*R/nPup + R/(2*nPup)
    Pupil1d      = (r>PupilID)*1.0
    LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0
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
    f,t0,t1=opt.solve_frank_wolfe(problem,init='Linf', nmax=10000000,gradmin=2e-9,solve=0,linesearch=1)
    t20.append(t0)
    t21.append(t1)
    L2.append(opt.calcul_snr(problem,f))
    

t30=[]
t31=[]
L3=[]

for k in L:
    nPup=k
    r   = np.arange(nPup)*R/nPup + R/(2*nPup)
    Pupil1d      = (r>PupilID)*1.0
    LyotStop1d   = (r>LyotStopID)*(r<LyotStopOD)*1.0
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
    f,t0,t1=opt.solve_frank_wolfe(problem,init='Linf', nmax=10000000,gradmin=2e-9,solve=1,linesearch=1)
    t30.append(t0)
    t31.append(t1)
    L3.append(opt.calcul_snr(problem,f))


#%%

"""
Parameters
"""
nlambis=11
corono_name  = 'APLC' # 'APLC' or 'SP'
problem_name = 'MaxContrastL1' # ,'MaxTau' # 'MaxContrastLinf' #'MaxContrastL1' 
#'MaxContrastL2'
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
nlam = 5

PupilID    = 0.2 
rMask       = 4.0

rMask1      = 2.0
rMask2      = 3.0
rMask3      = 3.5
OPDx2       = 0.5
OPDx3       = 0.75
LyotStopID = 0.40
LyotStopOD = 1.0

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 = 4
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
                 solver = solver, problem_name = 'MaxContrastL1',
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
corono0 = coro.design.APLC1d(**params)
problem1 = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1', **params)
#A=opt.solve_frank_wolfe(problem1,'Linf',500000,gradmin=1e-9,solve=1,linesearch=1,x=[])
#B=opt.solve_frank_wolfe(problem1,'Linf',500000,gradmin=1e-7,solve=1,linesearch=1,x=[])

log1=A[3]['loss']
log2=B[3]['loss']
Apod1=A[0]
Apod2=B[0]
#    

pl.plot(log1, label='Basic Linesearch')

pl.plot(log2, label='Closed-form linesearch')

pl.title(r'Evolution of the Cost-fonction for $\tau$=0.5')
pl.legend(loc='best')
pl.xlabel(r'Iteration number')
pl.ylabel('Cost function')
pl.grid(b=True,which='minor',linestyle='--')
pl.grid(b=True, which='both')

opt.analyse_apod(corono0,problem1,Apod1,color0='c',l='Basic linesearch optimisation')
opt.analyse_apod(corono0,problem1,Apod2,color0='m',l='Closed-form linesearch optimisation')

pl.figure()
pl.plot(Apod2, label='Closed-form linesearch')
pl.plot(Apod1, label='Basic Linesearch')

pl.title(r'Optimized apodizers for $\tau$=0.5')
pl.legend(loc='best')
pl.xlabel(r'')
pl.ylabel('Transmission rate')
pl.grid(b=True,which='minor',linestyle='--')
pl.grid(b=True, which='both')



#%%

fichier = np.loadtxt("données/Temps_calcul_tau_07.txt")
L=fichier[:,0]
t00=fichier[:,1]
t10=fichier[:,2]
#t20=fichier[:,3]
#t30=fichier[:,4]


fig=pl.figure()
fig.suptitle("calculation time depending on the pupil shape", fontsize=14)
pl.semilogy(L, t00,'-r',ls='--', label=r'calculation time with basic solver and linesearch')
pl.semilogy(L, t10, '-y',ls='--', label=r'calculation time with basic linesearch and closed-form solver')
pl.semilogy(L, t20,'-r', label=r' calculation time with basic solver and closed-form linesearch')
pl.semilogy(L, t30,'-y', label=r' calculation time with closed-form linesearch and solver')
pl.legend()
pl.grid()
#'''
#Enregistrer les data
#'''
data = np.array([L,t00,t10,t20,t30])
data = data.T
#here you transpose your data, so to have it in two columns

datafile_path = "données/Temps_calcul_tau_07_old.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, data, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f'])

