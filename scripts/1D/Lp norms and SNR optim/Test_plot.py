#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:11:34 2019

Author: Adam Hessas <adam.hessas@ecl17.ec-lyon.fr> (https://github.com/astromam)

License: MIT license
"""
import numpy as np
import pylab as pl
import os

"""
Check if the directory to save all figures exists. If not, creates it.
"""

if not os.path.exists('../../../images/Lp norms and SNR optim'):
        os.mkdir('../../../images/Lp norms and SNR optim')


#%% 
'''
Test comparaison of the Lp norm residual achieved with each optimization methods
It coresponds to Fig 2.2
'''
#Load the Lp norm of the residual achieved with the L1 solution. x contains every tau
fichier = np.loadtxt("../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_1.txt")
x=fichier[:,0]
L11=fichier[:,1]
L1inf=fichier[:,2]
L12=fichier[:,3]
L1r=fichier[:,4]

#Load the Lp norm of the residual achieved with the Linf solution. x contains every tau
fichier = np.loadtxt("../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_inf.txt")
x=fichier[:,0]
Linf1=fichier[:,1]
Linfinf=fichier[:,2]
Linf2=fichier[:,3]
Linfr=fichier[:,4]

#Load the Lp norm of the residual achieved with the L2 solution. x contains every tau
fichier = np.loadtxt("../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_22.txt")
x=fichier[:,0]
L21=fichier[:,1]
L2inf=fichier[:,2]
L22=fichier[:,3]
L2r=fichier[:,4]

#Load the Lp norm of the residual achieved with the Frank-Wolfe algorithm initialized
#with the L2 solution. x contains every tau
fichier = np.loadtxt("../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_ratio_init_L2.txt")
x=fichier[:,0]
Lr1=fichier[:,1]
Lrinf=fichier[:,2]
Lr2=fichier[:,3]
Lrr=fichier[:,4]

#Load the Lp norm of the residual achieved with the Frank-Wolfe algorithm initialized
#with the L1 solution. x contains every tau
fichier = np.loadtxt("../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_ratio_init_L1.txt")
x=fichier[:,0]
Lr11=fichier[:,1]
Lr1inf=fichier[:,2]
Lr12=fichier[:,3]
Lr1r=fichier[:,4]

#Load the Lp norm of the residual achieved with the Frank-Wolfe algorithm initialized
#with the Linf solution. x contains every tau
fichier = np.loadtxt("../../../data/1D/Lp norms and SNR optim/Résultats_minimisation_ratio_init_Linf.txt")
x=fichier[:,0]
Lrinf1=fichier[:,1]
Lrinfinf=fichier[:,2]
Lrinf2=fichier[:,3]
Lrinfr=fichier[:,4]



pl.close("all")
fig=pl.figure(10,(12,5))
pl.clf()

axe1=fig.add_subplot(1,3,1)

pl.title(r'$L_{\infty}$ norm residual in the dark zone')
pl.semilogy(x, L2inf,'--' , label=r'Min $L_2$')
pl.semilogy(x, Linfinf, '-', label=r'Min $L_{\infty}$')
pl.semilogy(x, L1inf, '-', label=r'Min $L_1$')

pl.xlabel(r'$\tau$')
pl.ylabel(r'Residual')
pl.legend(loc='best')

axe2=fig.add_subplot(1,3,2)

pl.title(r'$L_1$ norm residual in the dark zone')
pl.semilogy(x, L11,'--' , label=r'Min $L_2$')
pl.semilogy(x, Linf1, '-', label=r'Min $L_{\infty}$')
pl.semilogy(x, L21, '-', label=r'Min $L_1$')

pl.xlabel(r'$\tau$')
pl.legend(loc='best')

axe0=fig.add_subplot(1,3,3)

pl.title(r'$L_2$ norm residual in the dark zone')
pl.semilogy(x, L22,'--', label=r'Min $L_2$')
pl.semilogy(x, Linf2, '-', label=r'Min $L_{\infty}$')
pl.semilogy(x, L12,'-', label=r'Min $L_1$')

pl.xlabel(r'$\tau$')
pl.legend(loc='best')

axe0.grid(b=True, which='major')
axe0.grid(b=True, which='minor',linestyle='--')

axe1.grid(b=True, which='major')
axe1.grid(b=True, which='minor',linestyle='--')

axe2.grid(b=True, which='major')
axe2.grid(b=True, which='minor',linestyle='--')

pl.savefig('../../../images/Lp norms and SNR optim/comp_lp_sans_snr.pdf',bbox_inches='tight')


#%%
'''
Plot the SNR for various initializations of the Frank-Wolfe optimization.
It corresponds to Fig 3.6
'''
pl.figure(figsize=(9,5))
pl.title(r'SNR for various initializations of the Frank-Wolfe algorithm')
pl.semilogy(x, Lrinfr, label=r'Initialization $L_{\infty}$')
pl.semilogy(x, Lrr, label=r'Initialization $L_{2}$')
pl.semilogy(x, Lr1r, label=r'Initialization $L_1$')

pl.semilogy(x, Linfr,'--', label=r'State of the art : $L_{\infty}$ norm minimization')
pl.legend()
pl.grid(b=True, which='major')
pl.grid(b=True,which='minor',linestyle='--')
pl.xlabel(r'$\tau$')
pl.ylabel('SNR in the darkzone')
pl.savefig('../../../images/Lp norms and SNR optim/rsb_selon_init_sans_rand.pdf',bbox_inches='tight')


#%%
"""
Plot the evolution of the SNR after a Frank-Wolfe optimization. It corresponds to
the Fig. 3.3
"""
ylim=[1,1e8]

fig2=pl.figure(11,(10,5))
pl.clf()

#Plot the residual for the L1 initialization
axe0=fig2.add_subplot(1,3,3)

pl.title(r'min $L_1$ initialization')
pl.semilogy(x, L1r, '-', label=r'Init.')
pl.semilogy(x, Lr1r,'-', label=r'SNR optim.')
pl.yticks((1e1,1e2,1e3,1e4),('','','',''))
pl.xlabel(r'$\tau$')
pl.legend(loc='best')
pl.ylim(ylim)

#Plot the residual for the Linf initialization
axe1=fig2.add_subplot(1,3,2)
pl.title(r'min $L_{\infty}$ initialization')
pl.semilogy(x, Linfr, '-', label=r'Init.')
pl.semilogy(x, Lrinfr,'-' , label=r'SNR optim.')
pl.yticks((1e1,1e2,1e3,1e4),('','','',''))
pl.xlabel(r'$\tau$')
pl.legend(loc='best')
pl.ylim(ylim)

#Plot the residual for the L2 initialization
axe2=fig2.add_subplot(1,3,1)
pl.title(r'min $L_2$ initialization')
pl.semilogy(x, L2r, '-', label=r'Init.')
pl.semilogy(x, Lrr,'-' , label=r'SNR optim.')
pl.xlabel(r'$\tau$')
pl.ylabel('SNR in the darkzone')
pl.legend(loc='best')
pl.ylim(ylim)

#Add a grid on each subplot
axe0.grid(b=True, which='major')
axe0.grid(b=True, which='minor',linestyle='--')

axe1.grid(b=True, which='major')
axe1.grid(b=True, which='minor',linestyle='--')

axe2.grid(b=True, which='major')
axe2.grid(b=True, which='minor',linestyle='--')

#Save the figure
pl.savefig('../../../images/Lp norms and SNR optim/comp_init_snr.pdf',bbox_inches='tight')

#%%
"""
#Plot a comparison of the shape of the L2 solution and the Frank Wolfe optimization initialized by it.
It corresponds to Fig. 3.4 and 3.5 depending on k 
"""
#k is the index that corresponds to the tau that is to be compared
k=13

#load the solution apodizer that we want to compare
Apod2 = np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_l2.txt").T[k]
Apodfw = np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_fw_initL2.txt").T[k]

#create the figure
fig=pl.figure(10,(12,5))
pl.plot(np.linspace(0,1,len(Apod2)),Apod2,label=r'$L_2$ minimization',color = 'blue')
pl.legend(loc='best')
pl.xlabel(r'Radial coordinate', fontsize=12)
pl.ylabel(r'Transmission rate', fontsize=12)
pl.title(r'Solution apodizers', fontsize=12)
pl.plot(np.linspace(0,1,len(Apodfw)),Apodfw,label='Frank-Wolfe optim',color = 'orange')
pl.legend(loc='best')

pl.savefig('../../../images/Lp norms and SNR optim/Comp_Apod_FW_tau={0:.2f}.pdf'.format(x[k]),bbox_inches='tight')

#%%
"""
#Plot a comparison of the shape of the solution apodizers for a given tau.
It corresponds to Fig. 2.6 and 2.7 depending on k 
"""
#k is the index that corresponds to the tau that is to be compared
k=13

#load the solution apodizer that we want to compare
Apod1 = np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_l1.txt").T[k]
Apod2 = np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_l2.txt").T[k]
Apodinf = np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_linf.txt").T[k]


#create the figure
fig=pl.figure()
pl.subplot(3,1,1)
pl.title(r'Solution apodizers for $\tau$={0:.2f}'.format(x[k]), fontsize=12)

pl.plot(np.linspace(0,1,len(Apod1)),Apod1,label=r'$L_1$ minimization',color = 'green')
pl.legend(loc='best')

pl.subplot(3,1,2)
pl.plot(np.linspace(0,1,len(Apod2)),Apod2,label=r'$L_2$ minimization',color = 'blue')
pl.legend(loc='best')
pl.ylabel(r'Transmission rate', fontsize=12)
pl.subplot(3,1,3)
pl.xlabel(r'Radial coordinate', fontsize=12)
pl.plot(np.linspace(0,1,len(Apodinf)),Apodinf,label=r'$L_{\infty}$ minimization',color = 'orange')
pl.legend(loc='best')

pl.savefig('../../../images/Lp norms and SNR optim/Comp_Apod_tau={0:.2f}.pdf'.format(x[k]),bbox_inches='tight')
#%%

"""
Plot an overview of the shape of all the apodizers that have been generated.
It corresponds to Fig. 2.8 and 3.7
"""
#Load the shape of every apodizers solutions for all the minimization methods

apod1=np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_l1.txt").T
apod2=np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_l2.txt").T
apodinf=np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_linf.txt").T
apod_fw_2=np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_fw_initL2.txt").T
apod_fw_1=np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_fw_initL1.txt").T
apod_fw_inf=np.loadtxt("../../../data/1D/Lp norms and SNR optim/apod_fw_initLinf.txt").T


#Compare the shape of the apodizer depending on the initialization of the Frank-Wolfe algorithm
#Create a figure with 3 columns
fig, axes = pl.subplots(nrows=1, ncols=3)

#Plot the shape of the Frank-wolfe solutions initialized with the Linf solution
im = axes.flat[0].imshow(apod_fw_inf,aspect="auto",extent=[0,len(apod1[1]),1,0])
axes.flat[0].title.set_text(r'$L_{\infty}$ initializations') 
axes.flat[0].set_xlabel('Pupil index') 
axes.flat[0].set_ylabel(r'Transmission $\tau$') 

#Plot the shape of the Frank-wolfe solutions initialized with the L1 solution
im = axes.flat[1].imshow(apod_fw_1,aspect="auto",extent=[0,len(apod1[1]),1,0])
axes.flat[1].title.set_text(r'$L_{1}$ initializations') 
axes.flat[1].set_xlabel('Pupil index') 

#Plot the shape of the Frank-wolfe solutions initialized with the L2 solution
im = axes.flat[2].imshow(apod_fw_2,aspect="auto",extent=[0,len(apod1[1]),1,0])
axes.flat[2].title.set_text(r'$L_{2}$ initializations')
axes.flat[2].set_xlabel('Pupil index') 

pl.savefig('../../../images/Lp norms and SNR optim/Comp_all_apod_FW.pdf',bbox_inches='tight')

#Compare the shape of the apodizer depending on the criteria that is minimized

fig.colorbar(im, ax=axes.ravel().tolist())

pl.show()

fig, axes = pl.subplots(nrows=1, ncols=3)

#Plot the shape of the Linf solutions for every Tau
im = axes.flat[0].imshow(apodinf,aspect="auto",extent=[0,len(apod1[1]),1,0])
axes.flat[0].title.set_text(r'$L_{\infty}$ solutions') 
axes.flat[0].set_xlabel('Pupil index') 
axes.flat[0].set_ylabel(r'Transmission $\tau$') 

#Plot the shape of the L1 solutions for every Tau
im = axes.flat[1].imshow(apod1,aspect="auto",extent=[0,len(apod1[1]),1,0])
axes.flat[1].title.set_text(r'$L_{1}$ solutions') 
axes.flat[1].set_xlabel('Pupil index') 

#Plot the shape of the L2 solutions for every Tau
im = axes.flat[2].imshow(apod2,aspect="auto",extent=[0,len(apod1[1]),1,0])
axes.flat[2].title.set_text(r'$L_{2}$ soultions')
axes.flat[2].set_xlabel('Pupil index') 



fig.colorbar(im, ax=axes.ravel().tolist())
pl.savefig('../../../images/Lp norms and SNR optim/Comp_all_apod.pdf',bbox_inches='tight')

pl.show()
#%%

"""
Plot the evolution of the intensity residual in the dark zone for the
L1,L2 and Linf solutions of the optimization problem.  It corresponds to
Fig. 2.3, 2.4 and 2.5 of the report 
"""
#Load the intensity residual in the dark zone for the
#L1,L2 and Linf solutions for all transmissions Tau
residual_1=np.loadtxt("../../../data/1D/Lp norms and SNR optim/residual_dz_l1.txt").T
residual_2=np.loadtxt("../../../data/1D/Lp norms and SNR optim/residual_dz_l2.txt").T
residual_inf=np.loadtxt("../../../data/1D/Lp norms and SNR optim/residual_dz_linf.txt").T

#Load the x-axis vector of the darkzone
xi=np.loadtxt("../../../data/1D/Lp norms and SNR optim/corono_xi.txt").T

#Define the inner working angle and outter working angle of the dark zone
rho0=3.5
rho1=10

#Define the colors of the curves
colors = pl.cm.rainbow(np.linspace(0,1.0,len(residual_1)))


#Plot the residual for the L1 solutions and save the figure
pl.figure()
pl.grid()
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.title(r'$L_{2}$ Solutions')
pl.ylim(1e-14, 1e-3)
for i in range (len(residual_1)):
    pl.semilogy(xi,residual_1[i],label=r'$\tau=${0:.2f}'.format((i+1)/(len(residual_1)+1)), color = colors[i])
pl.legend()
pl.tight_layout()
pl.savefig('../../../images/Lp norms and SNR optim/evol_residual_1.pdf',bbox_inches='tight')

#Plot the residual for the L2 solutions and save the figure
pl.figure()
pl.grid()
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.title(r'$L_{1}$ Solutions')
pl.ylim(1e-14, 1e-3)
for i in range (len(residual_1)):
    pl.semilogy(xi,residual_2[i],label=r'$\tau=${0:.2f}'.format((i+1)/(len(residual_2)+1)), color = colors[i])
pl.legend()
pl.tight_layout()
pl.savefig('../../../images/Lp norms and SNR optim/evol_residual_2.pdf',bbox_inches='tight')

#Plot the residual for the Linf solutions and save the figure
pl.figure()
pl.grid()
pl.axvline(x=rho0, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.axvline(x=rho1, ymin=-12, ymax =2, linewidth=1, color='b', linestyle='--')
pl.xlabel(r'Angular separation in $\lambda_0$/D')
pl.ylabel('Normalized intensity in log scale')
pl.title(r'$L_{\infty}$ Solutions')
pl.ylim(1e-14, 1e-3)
for i in range (len(residual_1)):
    pl.semilogy(xi,residual_inf[i],label=r'$\tau=${0:.2f}'.format((i+1)/(len(residual_inf)+1)), color = colors[i])
pl.legend()
pl.tight_layout()
pl.savefig('../../../images/Lp norms and SNR optim/evol_residual_inf.pdf',bbox_inches='tight')

#%%
"""
Plot an overviw of the shape of all the intensity residuals in the dark zone 
depending on the minimization criteria
"""

#Load the intensity residual in the dark zone for the
#L1,L2 and Linf solutions for all transmissions Tau
residual_1=np.loadtxt("../../../data/1D/Lp norms and SNR optim/residual_dz_l1.txt").T
residual_2=np.loadtxt("../../../data/1D/Lp norms and SNR optim/residual_dz_l2.txt").T
residual_inf=np.loadtxt("../../../data/1D/Lp norms and SNR optim/residual_dz_linf.txt").T

#Create a figure with 3 columns
fig, axes = pl.subplots(nrows=1, ncols=3)

#Plot the shape of the intensity residual for the Linf solution
im = axes.flat[0].imshow(residual_inf,aspect="auto",extent=[0,len(residual_1[1]),1,0])
axes.flat[0].title.set_text(r'$L_{\infty}$ solutions residual') 
axes.flat[0].set_xlabel('Dark zone index') 
axes.flat[0].set_ylabel(r'Transmission $\tau$') 

#Plot the shape of the intensity residual for the L1 solution
im = axes.flat[1].imshow(residual_1,aspect="auto",extent=[0,len(residual_1[1]),1,0])
axes.flat[1].title.set_text(r'$L_{1}$ solutions residual') 
axes.flat[1].set_xlabel('Dark zone index') 

#Plot the shape of the intensity residual for the L2 solution
im = axes.flat[2].imshow(residual_2,aspect="auto",extent=[0,len(residual_1[1]),1,0])
axes.flat[2].title.set_text(r'$L_{2}$ solutions residual')
axes.flat[2].set_xlabel('Dark zone index') 

pl.savefig('../../../images/Lp norms and SNR optim/Comp_all_residual.pdf',bbox_inches='tight')

fig.colorbar(im, ax=axes.ravel().tolist())

pl.show()

#%%
'''
Plot a 2d comparison of the shape of the optimized apodizers
'''
#Choose the index that corresponds to the tau for which the comparison will be done
k=13

#Create the figure and the polar grid that will be used
fig, ax = pl.subplots(1,3,subplot_kw=dict(projection='polar'))
fig.suptitle(r'$\tau$={0:.2f}'.format(x[k]), fontsize=16)
azm = np.linspace(0, 2 * np.pi)
r, th = np.meshgrid(np.linspace(0,1,len(apod1[4])), azm)

#Plot the L1 solution
z1 = np.tile(apod1[k], (r.shape[0], 1))
ax[0].pcolormesh(th, r, z1)
ax[0].set_yticklabels([])
ax[0].set_xticklabels([])
ax[0].set_xlabel(r'$L_1$ solution' )

#Plot the L2 solution
z2 = np.tile(apod2[k], (r.shape[0], 1))
ax[1].pcolormesh(th, r, z2)
ax[1].set_yticklabels([])
ax[1].set_xticklabels([])
ax[1].set_xlabel(r'$L_2$ solution' )

#Plot the Linf solution
z3 = np.tile(apodinf[k], (r.shape[0], 1))
mesh=ax[2].pcolormesh(th, r, z3)
ax[2].set_yticklabels([])
ax[2].set_xticklabels([])
ax[2].set_xlabel(r'$L_{\infty}$ solution' )




pl.savefig('../../../images/Lp norms and SNR optim/Comp_Apod_2d_tau={0:.2f}.pdf'.format(x[k]),bbox_inches='tight')
    #%%

'''
Save a bunch of 2d apodizer for all criteria to see the evolution of the shape of the
apodizer
'''
azm = np.linspace(0, 2 * np.pi)
r, th = np.meshgrid(np.linspace(0,1,len(apod1[4])), azm)

for k in range(len(x)):
    fig, ax=pl.subplots(1,3,subplot_kw=dict(projection='polar'))
    fig.suptitle(r'$\tau$={0:.2f}'.format(x[k]), fontsize=16)
    z2 = np.tile(apod1[k], (r.shape[0], 1))
    ax[0].pcolormesh(th, r, z2)
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[0].set_xlabel(r'$L_1$ solution' )
    z3 = np.tile(apod2[k], (r.shape[0], 1))
    ax[1].pcolormesh(th, r, z3)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xlabel(r'$L_2$ solution' )
    z4 = np.tile(apodinf[k], (r.shape[0], 1))
    ax[2].pcolormesh(th, r, z4)
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])
    ax[2].set_xlabel(r'$L_{\infty}$ solution' )
    pl.savefig('../../../images/Lp norms and SNR optim/comp_apod_Lp_{0:.1f}.png'.format(k+1),bbox_inches='tight')
