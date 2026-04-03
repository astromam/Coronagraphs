# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:36:25 2024

@author: asp
"""
# plot psf maps from contrast profile
# fits files produced by test_andes_corono_multi_wvl*.py

#%%
"""
### Initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
import os
from pathlib import Path
from mpl_toolkits.axes_grid1 import AxesGrid

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #  mdiaye 15!


#%%
"""
### scaling
"""

lam_c = 1.6e-6  #  "central" reference lambda, "of interest", in meters

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1./rad2mas

# angular separation of interest in mas
as_oi = 20.


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_base  = Path('D:/Andes/Data_corono/').resolve()
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots

# elif user == 'toto':

was_donow = '20260403141537'
fdir_res = fdir_res / was_donow
fdir_plt = fdir_plt / was_donow


# new ref 1 kHz 01/10/2025
# 0.89 0.35 3.9 20251001085445 ruane2018 YJH
# 20251106160002
# 0.89/0.35/3.9 + Lyot stop azimutal offset
# 20251107134453 2 pixels / 0.5% D 
# 20251107134538 4 pixels / 1.0% D 
# 20251107134615 8 pixels / 2.0% D 

# 0.89/0.35/3.9 + Lyot stop vertical offset
# 20251104110334 2 pixels / 0.5% D
# 20251104110517 4 pixels / 1.0% D
# 20251104110556 8 pixels / 2.0% D

# 0.89/0.35/3.9 + fpm defocus
# 20251103103334  10 nm RMS
# 20251103103358  30 nm RMS
# 20251103103429  50 nm RMS
# 20251105104900  70 nm RMS

# 0.89/0.35/3.9 + offsets
# 20251013163430 1 mas
# 20251013163502 2 mas
# 20251013163616 3 mas
# 20251015134855 5 mas

# 0.89/0.35/3.9 + disp
# 20251014150042  5 mas/µ
# 20251014150115 10 mas/µ
# 20251015162408 15 mas/µ
# 20251020152303 20 mas/µ

# 0.89/0.35/3.9 + ncpa
# 20251016135714 10 nm RMS
# 20251016135818 30 nm RMS
# 20251016135837 50 nm RMS
# 20251020152206 70 nm RMS

# 0.89 0.35 3.9 + LS angular position error
# 20251017115345 0.5°
# 20251017115549 1.0°
# 20251017115610 1.5°
# 20251017120335 2.0°
# 20251017120423 3.0°
# 20251017120448 4.0°


#%%
"""
### working directory of the OPD files
"""
# New set of OPDs from PASSATA 

# root = 'OPDs_PASSATA/OPD/WS/'
# opds_dir=(root+'ocam2k',)
# opds_dir=(root+'alice',)
# opds_dir=(root+'cam_500us',)


#%%
# root = 'OPDs_PASSATA/OPD/WS/500HzVarWS/'
# opds_dir=(root+'20250704_105833.0',
#           root+'20250704_112007.0',
#           root+'20250704_114101.0',
#           root+'20250704_121550.0',
#           root+'20250704_123540.0',
#           root+'20250704_125507.0',
#           root+'20250704_131433.0',
#           root+'20250704_135325.0',
#           root+'20250704_141253.0',
#           root+'20250707_114321.0')
root = 'OPDs_PASSATA/OPD/WS/1kHzVarWS/'
opds_dir=(root+'1',)
# opds_dir=(root+'1',
#           root+'2',
#           root+'3',
#           root+'4',
#           root+'5',
#           root+'6',
#           root+'7',
#           root+'8',
#           root+'9',
#           root+'10')


#%%
for dir_nb in range(len(opds_dir)):
            
    opd_set = os.path.basename(fdir_res / opds_dir[dir_nb] ).split('.')[0]
    print(fdir_res / opd_set)
    
    #%%
    """
    ### Read psf files & create profiles
    """

    cur_dir = Path(os.path.basename(opds_dir[dir_nb])).stem
    res_dir = (fdir_res / cur_dir)
    print(cur_dir, res_dir)
    
    if os.path.isdir(res_dir):
        
        os.makedirs(fdir_plt / opd_set, exist_ok=True)
    
    file_lst = os.listdir(res_dir)

    if opd_set != 'perfect':
        
        file_psf = [x for x in file_lst if 'ao_corr_psf_' in x]
        file_cro = [x for x in file_lst if 'ao_corr_coro_psf_' in x]
    
    else:
        
        file_psf = [x for x in file_lst if 'wonoise_psf_' in x]
        file_cro = [x for x in file_lst if 'wonoise_coro_psf_' in x]
    
    print("psf :", file_psf)
    print("psf coro :", file_cro)

    file_psf = file_psf[np.argmin(
        (lambda x:[len(i) for i in x])(file_psf))]
    file_cro = file_cro[np.argmin(
        (lambda x:[len(i) for i in x])(file_cro))]

    base_psf = os.path.basename(file_psf).split('.')[0]
    base_cro = os.path.basename(file_cro).split('.')[0]
    
    Int_D0 = fits.getdata(fdir_res / opd_set / file_psf)
    Int_D  = fits.getdata(fdir_res / opd_set / file_cro)
    
    head_psf = fits.getheader(fdir_res / opd_set / file_psf)
    
    lam_min = head_psf['LMIN']
    lam_stp = head_psf['LSTP']
    lam_itv = head_psf['LITV']
    lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)
    lam_lst = lam_lst[np.where(lam_lst < 1860e-9)]
    
    if np.median(lam_lst) > 1e-6:
        wvl = 'YJH'
    else:
        wvl = 'RIZ'
    
    nL = len(lam_lst)
    
    nImg = head_psf['NIMG']
    pscale = head_psf['PSCL']
    D = head_psf['DIAM']
    mB = head_psf['SFPM']
    # lam_c = head_psf['LMBD']
    hlf_fov = nImg * pscale / 2.

    # if lambda of interest / D
    rW_mas = (lam_c / D) * mas2rad

    a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
    b_ = a_.copy()
    aa, bb = np.meshgrid(a_, b_)
    rad_pix = np.sqrt(aa**2 + bb**2)
    rad_mas = rad_pix * (mas2rad * hlf_fov / (D * 1e9) )
    
    #  angular separation
    aS = np.arange(nImg//2) * (mas2rad * hlf_fov / (D * 1e9) )
              
    # filename of the plot
    fnm =  f'ao_corrected_coro_psf_{wvl}band'
    # fname_images_svg = fnm+'.svg'
    fname_images_pdf = fnm+'.pdf'
    fname_images_png = fnm+'.png'

    # filepath for the direct and coronagraphic images
    fpath_images_pdf = fdir_plt / opd_set / fname_images_pdf
    # fpath_images_png = fdir_plt / opd_set / fname_images_png

    # index of wvl to display
    iD = [0,nL//4+1,nL//2-1,nL*3//4-1,nL-1]
    
    # boundaries for the images in log scale
    vmin0 = -5
    vmax0 = 0
    
    fig = plt.figure(4, figsize=(16,6))
    plt.clf()
    plt.tight_layout()
    # plt.suptitle('ao corrected psf (top) vs ao corrected coro. psf (bottom)')
    
    grid = AxesGrid(fig, 111,
            nrows_ncols=(2, 5),
            axes_pad=0.3,
            cbar_mode='single',
            cbar_location='right',
            cbar_pad=0.2,
            label_mode='L'
            )
    
    for i in range(5):
        
        print('position maxi in flattened psf array: ',
              np.argmax(Int_D0[iD[i],:]))
        im = grid[i].imshow(
            np.log10(Int_D0[iD[i],:]),vmin=vmin0,vmax=vmax0,cmap='inferno',
            extent=(-aS[-1],aS[-1],-aS[-1],aS[-1]))
        grid[i].set_ylabel("sep. [mas]")
        grid[i].set_title(str(int(lam_lst[iD[i]]*1e9+.1))+'nm')
         
        im = grid[i+5].imshow(
            np.log10(Int_D[iD[i],:]),vmin=vmin0,vmax=vmax0,cmap='inferno',
            extent=(-aS[-1],aS[-1],-aS[-1],aS[-1]))
        grid[i+5].add_patch(
            Ellipse( (0,0),rW_mas*mB,rW_mas*mB,
                    color='w',ls=':',hatch='xxx',fill=False,alpha=0.7))
        grid[i+5].set_xlabel("sep. [mas]")
        grid[i+5].set_ylabel("sep. [mas]")
        # grid[i+5].add_patch(
        #     Ellipse( (0,0), as_oi*2, as_oi*2,color='w',ls=':',lw=2,fill=False))
        
    # colorbar
    cbar = grid[0].cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Intensity in log scale', rotation=270)
    
    # plt.savefig(fpath_images_svg)
    plt.savefig(fpath_images_pdf, bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(fpath_images_png)

    plt.show()
    plt.close()

