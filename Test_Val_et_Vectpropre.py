#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:45:57 2019

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
                 solver = solver,problem_name = 'MaxContrastL1',
                 corono_name = corono_name, slvLogToConsole = slvLogToConsole,
                 slvCrossover = slvCrossover, slvMethod = slvMethod,
                 allLogToConsole = allLogToConsole,
                 FirstDer = FirstDer, SecondDer = SecondDer,
                 FirstDerLim = FirstDerLim, SecondDerLim = SecondDerLim,
                 MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim)
corono0 = coro.design.APLC1d(**params)
problem = coro.optim_1d.MaxContrast(corono=corono0, Lnorm='L1', **params)
    
Q=opt.get_matrices_Contrast_deux2(problem)[2]
Valp,Vecp=np.linalg.eigh(Q)
pl.figure()
pl.title('Modules des valeurs propres de la matrice K')
pl.semilogy(abs(Valp))
pl.grid()
pl.show()
pl.figure()
pl.title("Transformée de Fourier d'un vecteur propre hors du noyau")
pl.semilogy(sorted(np.fft.fftfreq(len(Vecp[:,1]))),np.abs((np.fft.fftshift(np.fft.fft(Vecp[:,-1])))))
pl.grid()
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

pl.figure()
pl.title("Transformée de Fourier d'un vecteur propre appartenant au noyau")
pl.semilogy(sorted(np.fft.fftfreq(len(Vecp[:,1]))),np.abs(np.fft.fftshift(np.fft.fft(Vecp[:,200]))))
pl.grid()
pl.show()

M=np.zeros((len(Valp)//2,len(Valp)))
for i in range (len(Valp)):
    M[:,i]=np.abs(np.fft.fftshift(np.fft.fft(Vecp[:,i])))[len(Valp)//2:len(Valp)]
pl.figure()
pl.title('Repartition spectrale de la puissance')
 