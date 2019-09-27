#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:23:07 2019

@author: ahessas
"""
import corono as coro
import numpy as np
import pylab as pl
import Optimquad as opt


"""
Parameters
"""
Type='Comparaison' #'Analyse'
methode1 ='SNR' #'MaxTau','Linf','L2','L1'
init1='L1' #'L2','L1','unif','Random'
n1=1000

methode2 ='SNR' #'MaxTau','Linf','L2','L1'
init2='L1' #'L2','L1','unif','Random'
n2=500000
#x=B

nlambis=11
corono_name  = 'APLC' # 'APLC' or 'SP'
problem_name = 'MaxL1' # ,'MaxTau' # 'MaxContrastLinf' #'MaxContrastL1' 
#'MaxContrastL2' #'MaxSNR'
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
tau   = 0.6
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
problem = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1', **params)

if methode1 =='L1' :
    A=opt.solve1(problem)

    
elif methode1 =='L2' :
    A=opt.solve22(problem)

    
elif methode1 =='Linf' :
    A=opt.solveinf(problem)

elif methode1 =='SNR':
    A=opt.solve_frank_wolfe(problem,init1,n1)[0]

if Type =='Analyse':
    opt.analyse_apod(corono0,problem,A,color0='y')
    pl.figure()
    pl.plot(A)


    
if Type =='Comparaison':
    if methode2 =='L1' :
        B=opt.solve1(problem)

    
    elif methode2 =='L2' :
        B=opt.solve22(problem)

    
    elif methode2 =='Linf' :
        B=opt.solveinf(problem)

    elif methode2 =='SNR':
        if init2=='Manual':
            B=opt.solve_frank_wolfe(problem,init2,n2,10**-10,x)[0]
        else:
            B=opt.solve_frank_wolfe(problem,init2,n2)[0]
    pl.clf()    
    opt.analyse_apod(corono0,problem,A,color0='r')
    opt.analyse_apod(corono0,problem,B,color0='g')

    pl.figure()
    pl.title("Comparaison des apodiseurs 1 et 2")
    pl.plot(A,'--' , label=r'Apodiseur 1')
    pl.plot(B, label=r'Apodiseur 2')
    pl.legend()


    


#C = L_inf 300000 iterations tau=0.5