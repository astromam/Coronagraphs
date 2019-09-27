#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:11:34 2019

@author: ahessas
"""
import numpy as np
import pylab as pl


#%% Test comparaison of the optimization methods

fichier = np.loadtxt("données/Résultats_minimisation_1.txt")
x=fichier[:,0]
L11=fichier[:,1]
L1inf=fichier[:,2]
L12=fichier[:,3]
L1r=fichier[:,4]

fichier = np.loadtxt("données/Résultats_minimisation_inf.txt")
x=fichier[:,0]
Linf1=fichier[:,1]
Linfinf=fichier[:,2]
Linf2=fichier[:,3]
Linfr=fichier[:,4]

fichier = np.loadtxt("données/Résultats_minimisation_22.txt")
x=fichier[:,0]
L221=fichier[:,1]
L22inf=fichier[:,2]
L222=fichier[:,3]
L22r=fichier[:,4]

fichier = np.loadtxt("données/Résultats_minimisation_ratio_init_L2.txt")
x=fichier[:,0]
Lr1=fichier[:,1]
Lrinf=fichier[:,2]
Lr2=fichier[:,3]
Lrr=fichier[:,4]

fichier = np.loadtxt("données/Résultats_minimisation_ratio_init_L1.txt")
x=fichier[:,0]
Lr11=fichier[:,1]
Lr1inf=fichier[:,2]
Lr12=fichier[:,3]
Lr1r=fichier[:,4]


fichier = np.loadtxt("données/Résultats_minimisation_ratio_init_Linf.txt")
x=fichier[:,0]
Lrinf1=fichier[:,1]
Lrinfinf=fichier[:,2]
Lrinf2=fichier[:,3]
Lrinfr=fichier[:,4]



pl.close("all")
fig=pl.figure(10,(12,5))
pl.clf()

axe0=fig.add_subplot(1,3,3)

pl.title(r'$L_2$ norm residual in the dark zone')
pl.semilogy(x, L222,'--', label=r'Min $L_2$')
pl.semilogy(x, Linf2, '-', label=r'Min $L_{\infty}$')
pl.semilogy(x, L12,'-', label=r'Min $L_1$')

pl.xlabel(r'$\tau$')
pl.legend(loc='best')

axe1=fig.add_subplot(1,3,1)

pl.title(r'$L_{\infty}$ norm residual in the dark zone')
pl.semilogy(x, L22inf,'--' , label=r'Min $L_2$')
pl.semilogy(x, Linfinf, '-', label=r'Min $L_{\infty}$')
pl.semilogy(x, L1inf, '-', label=r'Min $L_1$')

pl.xlabel(r'$\tau$')
pl.ylabel(r'Residual')
pl.legend(loc='best')

axe2=fig.add_subplot(1,3,2)

pl.title(r'$L_1$ norm residual in the dark zone')
pl.semilogy(x, L11,'--' , label=r'Min $L_2$')
pl.semilogy(x, Linf1, '-', label=r'Min $L_{\infty}$')
pl.semilogy(x, L221, '-', label=r'Min $L_1$')

pl.xlabel(r'$\tau$')
pl.legend(loc='best')



axe0.grid(b=True, which='major')
axe0.grid(b=True, which='minor',linestyle='--')

axe1.grid(b=True, which='major')
axe1.grid(b=True, which='minor',linestyle='--')

axe2.grid(b=True, which='major')
axe2.grid(b=True, which='minor',linestyle='--')

pl.savefig('imgs/comp_lp_sans_snr.pdf',bbox_inches='tight')


#%%#Plot the SNR for various initializations of the Frank-Wolfe optimization

pl.clf()
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
pl.savefig('imgs/rsb_selon_init_sans_rand.pdf',bbox_inches='tight')


#%%
#Plot the evolution of the SNR after a Frank-Wolfe optimization

ylim=[1,1e8]

fig2=pl.figure(11,(10,5))
pl.clf()

axe0=fig2.add_subplot(1,3,3)

pl.title(r'min $L_1$ initialization')
pl.semilogy(x, L1r, '-', label=r'Init.')
pl.semilogy(x, Lr1r,'-', label=r'SNR optim.')
pl.yticks((1e1,1e2,1e3,1e4),('','','',''))
pl.xlabel(r'$\tau$')
pl.legend(loc='best')
pl.ylim(ylim)

axe1=fig2.add_subplot(1,3,2)

pl.title(r'min $L_{\infty}$ initialization')
pl.semilogy(x, Linfr, '-', label=r'Init.')
pl.semilogy(x, Lrinfr,'-' , label=r'SNR optim.')
pl.yticks((1e1,1e2,1e3,1e4),('','','',''))
pl.xlabel(r'$\tau$')
pl.legend(loc='best')
pl.ylim(ylim)

axe2=fig2.add_subplot(1,3,1)

pl.title(r'min $L_2$ initialization')
pl.semilogy(x, L22r, '-', label=r'Init.')
pl.semilogy(x, Lrr,'-' , label=r'SNR optim.')
pl.xlabel(r'$\tau$')
pl.ylabel('SNR in the darkzone')
pl.legend(loc='best')
pl.ylim(ylim)

axe0.grid(b=True, which='major')
axe0.grid(b=True, which='minor',linestyle='--')

axe1.grid(b=True, which='major')
axe1.grid(b=True, which='minor',linestyle='--')

axe2.grid(b=True, which='major')
axe2.grid(b=True, which='minor',linestyle='--')

pl.savefig('imgs/comp_init_snr.pdf',bbox_inches='tight')

#%%
#Plot a comparison of the shape of the L1, L2 and Linf solutions for tau=0.3


ApodL1 = np.loadtxt("données/apod_l1.txt").T[7]
ApodL2 = np.loadtxt("données/apod_l2.txt").T[7]
ApodLinf = np.loadtxt("données/apod_linf.txt").T[7]

fig=pl.figure(10,(12,5))


pl.clf()


axe0=fig.add_subplot(3,1,3)

pl.plot(ApodL2,label=r'$L_2$ minimization',color = 'blue')
pl.xlabel(r'Pupil index', fontsize=12)
pl.legend(loc='best')

axe1=fig.add_subplot(3,1,1)

pl.title(r'Solution apodizers', fontsize=12)
pl.plot(ApodLinf,label=r'$L_{\infty}$ minimization',color = 'orange')
pl.legend(loc='best')

axe2=fig.add_subplot(3,1,2)

pl.plot(ApodL1,label=r'$L_1$ minimization',color = 'green')
pl.ylabel(r'Transmission rate', fontsize=12)
pl.legend(loc='best')


axe1.grid(b=True, which='major')
axe1.grid(b=True, which='minor',linestyle='--')

axe0.grid(b=True, which='major')
axe0.grid(b=True, which='minor',linestyle='--')

axe2.grid(b=True, which='major')
axe2.grid(b=True, which='minor',linestyle='--')


pl.savefig('imgs/Comp_Apod_Tau_03.pdf',bbox_inches='tight')


#%%
#Plot a comparison of the shape of the L2 solution and the Frank Wolfe optimization initialized by it. 

Apod2 = np.loadtxt("données/apod_l2.txt").T[13]
Apodfw = np.loadtxt("données/apod_fw_initL2.txt").T[13]


fig=pl.figure(10,(12,5))
pl.clf()
pl.plot(Apod2,label=r'$L_2$ minimization',color = 'blue')

pl.xlabel(r'Pupil index', fontsize=12)


pl.title(r'Solution apodizers', fontsize=12)
pl.plot(Apodfw,label='Frank-Wolfe optim',color = 'orange')
pl.legend(loc='best')
pl.ylabel(r'Transmission rate', fontsize=12)
pl.legend(loc='best')



axe1.grid(b=True, which='major')
axe1.grid(b=True, which='minor',linestyle='--')

axe0.grid(b=True, which='major')
axe0.grid(b=True, which='minor',linestyle='--')

axe2.grid(b=True, which='major')
axe2.grid(b=True, which='minor',linestyle='--')


pl.savefig('imgs/Comp_Apod_FW_07.pdf',bbox_inches='tight')




#%%pl.figure()
#Plot an overviw of the shape of all the apodizers that have been generated

apod1=np.loadtxt("données/apod_l1.txt").T
apod2=np.loadtxt("données/apod_l2.txt").T
apodinf=np.loadtxt("données/apod_linf.txt").T
apod_fw_2=np.loadtxt("données/apod_fw_initL2.txt").T
apod_fw_1=np.loadtxt("données/apod_fw_initL1.txt").T
apod_fw_inf=np.loadtxt("données/apod_fw_initLinf.txt").T


#Compare the shape of the apodizer depending on the initialization of the Frank-Wolfe algorithm

fig, axes = pl.subplots(nrows=1, ncols=3)

im = axes.flat[0].imshow(apod_fw_inf,aspect="auto",extent=[0,500,1,0])

axes.flat[0].title.set_text(r'$L_{\infty}$ initializations') 
axes.flat[0].set_xlabel('Pupil index') 
axes.flat[0].set_ylabel(r'Transmission $\tau$') 


im = axes.flat[1].imshow(apod_fw_1,aspect="auto",extent=[0,500,1,0])

axes.flat[1].title.set_text(r'$L_{1}$ initializations') 
axes.flat[1].set_xlabel('Pupil index') 

im = axes.flat[2].imshow(apod_fw_2,aspect="auto",extent=[0,500,1,0])

axes.flat[2].title.set_text(r'$L_{2}$ initializations')
axes.flat[2].set_xlabel('Pupil index') 

#Compare the shape of the apodizer depending on the criteria that is minimized

fig.colorbar(im, ax=axes.ravel().tolist())

pl.show()

fig, axes = pl.subplots(nrows=1, ncols=3)

im = axes.flat[0].imshow(apodinf,aspect="auto",extent=[0,500,1,0])

axes.flat[0].title.set_text(r'$L_{\infty}$ solutions') 
axes.flat[0].set_xlabel('Pupil index') 
axes.flat[0].set_ylabel(r'Transmission $\tau$') 


im = axes.flat[1].imshow(apod1,aspect="auto",extent=[0,500,1,0])

axes.flat[1].title.set_text(r'$L_{1}$ solutions') 
axes.flat[1].set_xlabel('Pupil index') 

im = axes.flat[2].imshow(apod2,aspect="auto",extent=[0,500,1,0])

axes.flat[2].title.set_text(r'$L_{2}$ soultions')
axes.flat[2].set_xlabel('Pupil index') 



fig.colorbar(im, ax=axes.ravel().tolist())

pl.show()
