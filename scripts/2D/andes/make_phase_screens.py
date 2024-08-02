# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:07:41 2024

@author: asimmonnin, mndiaye, asp
"""



# %%

import numpy as np
import slow_fourier_transform as sft

#%%
# Dimensions de l'image
N = 800
# Créer un espace de fréquence en 2D
kx = (np.arange(N)-N//2)/(N/2)  # Fréquences spatiales pour les colonnes
ky = (np.arange(N)-N//2)/(N/2)
kx2, ky2 = np.meshgrid(kx, ky)
k = np.sqrt(kx2**2 + ky2**2)  # Norme des vecteurs de fréquences

# Appliquer la loi f^-2 pour obtenir la DSP
# Pour éviter la division par zéro, on ajoute un petit epsilon à k
epsilon = 1e-10
dsp = 1/(k**2 + epsilon)

# Génération de phases aléatoires
# Calcul des composantes complexes du spectre
amplitude = np.sqrt(dsp)
random = np.random.randn(*amplitude.shape)
opd_ff = amplitude * random

# Transformée de Fourier inverse pour obtenir l'image spatiale
random_opd_f2 = np.real(sft.isft(amplitude*random,800,400))

# Normalisation de l'image pour l'affichage
opd_f2 = (random_opd_f2  - np.min(random_opd_f2 )) / (np.max(random_opd_f2 ) - np.min(random_opd_f2 ))
