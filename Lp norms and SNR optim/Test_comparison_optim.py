#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:23:07 2019

@author: ahessas
"""
import corono as coro
import numpy as np
import pylab as pl


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
nsim=5

#%%

# Compute the L1 optimization solutions and the norms of the associated residual

L11=[]
L1inf=[]
L12=[]
L1r=[]
apod1=np.zeros((nsim-1,500))
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
    f=problem.solve_model()
    apod1[k]=f
    params['problem_name']='MaxTau'
    problem = coro.optim_1d.MaxTau(corono=corono0, **params)
    problem.compute_response_matrices()
    h=np.dot(problem.corono_field_t.T,f[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,f[problem.idx_pup])    L11.append(sum([abs(x) for x in h]))
    L1inf.append(max(h))
    L12.append(np.linalg.norm(h, 2))
    L1r.append(np.linalg.norm(g, 2)/np.linalg.norm(h, 2))
#%%

# Compute the L_inf optimization solutions and the norms of the associated residual

apodinf=np.zeros((nsim-1,500))
Linf1=[]
Linfinf=[]
Linf2=[]
Linfr=[]

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
    f=problem.solve_model()
    apodinf[k]=f
    params['problem_name']='MaxTau'
    problem = coro.optim_1d.MaxTau(corono=corono0, **params)
    problem.compute_response_matrices()
    h=np.dot(problem.corono_field_t.T,f[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,f[problem.idx_pup])
    Linf1.append(sum([abs(x) for x in h]))
    Linfinf.append(max(h))
    Linf2.append(np.linalg.norm(h, 2))
    Linfr.append(np.linalg.norm(g, 2)/np.linalg.norm(h, 2))

#%%

# Compute the L2 optimization solutions and the norms of the associated residual

L21=[]
L2inf=[]
L22=[]
L2r=[]
apod2=np.zeros((nsim-1,500))

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
    f=problem.solve_model()
    apod2[k]=f
    params['problem_name']='MaxTau'
    problem = coro.optim_1d.MaxTau(corono=corono0, **params)
    problem.compute_response_matrices()
    h=np.dot(problem.corono_field_t.T,f[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,f[problem.idx_pup])
    L21.append(sum([abs(x) for x in h]))
    L2inf.append(max(h))
    L22.append(np.linalg.norm(h, 2))
    L2r.append(np.linalg.norm(g, 2)/np.linalg.norm(h, 2))
    
#%%

# Compute the Frank-Wolfe optimizations initialized by the L_2 solutions and the norms of the associated residual
    
Lr1=[]
Lrinf=[]
Lr2=[]
Lrr=[]
grad_final_opt=[]
apodr=np.zeros((nsim-1,500))
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
    problem = coro.optim_1d.MaxSNR(corono=corono0,nmax=200000,gradmin=1e-9, initialisation='L2' ,**params)
    print ("Tau ={0}".format(tau))
    f=problem.solve_model()
    apodr[k]=f
    problem.compute_response_matrices()
    h=np.dot(problem.corono_field_t.T,f[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,f[problem.idx_pup])
    Lr1.append(sum([abs(x) for x in h]))
    Lrinf.append(max(h))
    Lr2.append(np.linalg.norm(h, 2))
    Lrr.append(np.linalg.norm(g, 2)/np.linalg.norm(h, 2))
    
#%%

# Compute the Frank-Wolfe optimizations initialized by the L_1 solutions and the norms of the associated residual&
    
Lr11=[]
Lr1inf=[]
Lr12=[]
Lr1r=[]
grad_final_opt1=[]
apodr1=np.zeros((nsim-1,500))
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
    problem = coro.optim_1d.MaxSNR(corono=corono0,nmax=200000,gradmin=1e-9, initialisation='L1' ,**params)
    print ("Tau ={0}".format(tau))
    f=problem.solve_model()
    apodr1[k]=f
    problem.compute_response_matrices()
    h=np.dot(problem.corono_field_t.T,f[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,f[problem.idx_pup])
    Lr11.append(sum([abs(x) for x in h]))
    Lr1inf.append(max(h))
    Lr12.append(np.linalg.norm(h, 2))
    Lr1r.append(np.linalg.norm(g, 2)/np.linalg.norm(h, 2))
#%%

# Compute the Frank-Wolfe optimizations initialized by the L_inf solutions and the norms of the associated residual&

Lrinf1=[]
Lrinfinf=[]
Lrinf2=[]
Lrinfr=[]
grad_final_opt1=[]
apodrinf=np.zeros((nsim-1,500))
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
    problem = coro.optim_1d.MaxSNR(corono=corono0,nmax=200000,gradmin=1e-9, initialisation='Linf' ,**params)

    print ("Tau ={0}".format(tau))
    f=problem.solve_model()
    apodrinf[k]=f
    problem.compute_response_matrices()
    h=np.dot(problem.corono_field_t.T,f[problem.idx_pup])
    g=np.dot(problem.direct_field_re_t_tmp.T,f[problem.idx_pup])
    Lrinf1.append(sum([abs(x) for x in h]))
    Lrinfinf.append(max(h))
    Lrinf2.append(np.linalg.norm(h, 2))
    Lrinfr.append(np.linalg.norm(g, 2)/np.linalg.norm(h, 2))





#%%
#Built the vector of the various tau for which the optimization has been made
x=np.linspace(1/nsim,1-1/nsim,nsim-1)
    
'''
Enregistrer les data
'''
data = np.array([x,L11,L1inf,L12,L1r])
data = data.T
#here you transpose your data, so to have it in two columns

datafile_path = "données/Résultats_minimisation_1.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, data, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f'])
    
data = np.array([x,Linf1,Linfinf,Linf2,Linfr])
data = data.T
#here you transpose your data, so to have it in two columns

datafile_path = "données/Résultats_minimisation_inf.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, data, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f'])

data = np.array([x,L21,L2inf,L22,L2r])
data = data.T
#here you transpose your data, so to have it in two columns

datafile_path = "données/Résultats_minimisation_22.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, data, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f'])

data = np.array([x,Lr1,Lrinf,Lr2,Lrr])
data = data.T
#here you transpose your data, so to have it in two columns

datafile_path = "données/Résultats_minimisation_ratio_init_L2.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, data, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f'])
    
data = np.array([x,Lr11,Lr1inf,Lr12,Lr1r])
data = data.T
#here you transpose your data, so to have it in two columns

datafile_path = "données/Résultats_minimisation_ratio_init_L1.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, data, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f'])
    
data = np.array([x,Lrinf1,Lrinfinf,Lrinf2,Lrinfr])
data = data.T
#here you transpose your data, so to have it in two columns

datafile_path = "données/Résultats_minimisation_ratio_init_Linf.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, data, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f'])

datafile_path = "données/apod_l1.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, np.array(apod1).T, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f'])

datafile_path = "données/apod_l2.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, np.array(apod2).T, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f'])
datafile_path = "données/apod_linf.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, np.array(apodinf).T, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f'])

datafile_path = "données/apod_fw_initL2.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, np.array(apodr).T, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f'])

datafile_path = "données/apod_fw_initLinf.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, np.array(apodrinf).T, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f'])

datafile_path = "données/apod_fw_initL1.txt"
with open(datafile_path, 'w+') as datafile_id:
#here you open the ascii file
    np.savetxt(datafile_id, np.array(apodr1).T, fmt=['%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f','%.10f'])


#%%
#Plot a 3 dimensionnal view of the apodizers
import matplotlib.pyplot as plt



length = apodinf.shape[0]
width = apodinf.shape[1]
y,x= np.meshgrid( np.arange(width),(np.arange(length)+1)/20)

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(x, y, apod2,rstride=1, cstride=0)
plt.show()


#%%
#Plot the evolution of the intensiity residual in the dark zone

#design the corono 
nlambis = 11
nImgbis = 110
Fmaxbis = 11    
params2    = coro.update_params(params, nlam=nlambis, nImg=nImgbis, Fmax=Fmaxbis) 
corono0 = coro.design.APLC1d(**params2)


colors = pl.cm.rainbow(np.linspace(0,1.0,9))
pl.clf()
pl.grid()
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.title(r'$L_{2}$ Solutions')
pl.ylim(1e-14, 1e-3)
for i in range (9):
    poly_direct_image1 = corono0.compute_direct_intensity_1d(apod2[2*i+1])
    poly_corono_image1 = corono0.compute_corono_intensity_1d(apod2[2*i+1])
    pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label=r'$\tau=${0:.1f}'.format((i+1)/10), color = colors[i])
pl.legend()
pl.tight_layout()
pl.savefig('imgs/evol_residue_2.pdf',bbox_inches='tight')

pl.figure()

pl.grid()
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.title(r'$L_{1}$ Solutions')
pl.ylim(1e-14, 1e-3)
for i in range (9):
    poly_direct_image1 = corono0.compute_direct_intensity_1d(apod1[2*i+1])
    poly_corono_image1 = corono0.compute_corono_intensity_1d(apod1[2*i+1])
    pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label=r'$\tau=${0:.1f}'.format((i+1)/10), color = colors[i])
pl.legend()
pl.tight_layout()
pl.savefig('imgs/evol_residue_1.pdf',bbox_inches='tight')

pl.figure()
pl.grid()
pl.axvline(x=corono0.rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=corono0.rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.title(r'$L_{\infty}$ Solutions')
pl.ylim(1e-14, 1e-3)
for i in range (9):
    poly_direct_image1 = corono0.compute_direct_intensity_1d(apodinf[2*i+1])
    poly_corono_image1 = corono0.compute_corono_intensity_1d(apodinf[2*i+1])
    pl.semilogy(corono0.xi,poly_corono_image1/poly_direct_image1.max(),label=r'$\tau=${0:.1f}'.format((i+1)/10), color = colors[i])
pl.legend()
pl.tight_layout()
pl.savefig('imgs/evol_residue_inf.pdf',bbox_inches='tight')

