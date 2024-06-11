# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:36:25 2024

@author: asp
"""

#%%
"""
### Initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from pathlib import Path
from mpl_toolkits.axes_grid1 import AxesGrid

# from psf_profile import profile

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!


#%%
"""
stackoverflow...
"""

def recursive_search(path: str) -> "list[str]":
    """get all files from an absolute path

    :param path: absolute path of the directory to search
    :type path: str
    :return: a list of all files
    :rtype: list[str]
    """
    found_files = []
    if not os.path.isdir(path):
        raise RuntimeError(f"'{path}' is not a directory")
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isfile(full_path):
            found_files.append(full_path)
        elif os.path.isdir(full_path):
            found_files.extend(recursive_search(full_path))
    return found_files


#%%
"""
### scaling
"""

lam_c = 1.6e-6  #  "central" reference lambda, "of interest", in meters
slc = False  #  supersed lam_c with lam_lst[i]

# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# angular separation of interest in mas
as_oi = 25.


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots

elif user == 'Adrien':
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/results/').resolve()
    fdir_plt   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/plots/').resolve()

elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path( fdir_base / 'results' ).resolve()
    fdir_plt   = Path( fdir_base / 'plots' ).resolve()


#%%
"""
### working directory of the OPD files
"""
# New set of OPDs from PASSATA 

# opds_dir=('OPDs_PASSATA/OPD/20231124_090126.0/',
#               'OPDs_PASSATA/OPD/20231122_142204.0/',
#               'OPDs_PASSATA/OPD/20240227_234849-007/20240227_234849.0',
#               'OPDs_PASSATA/OPD/20240228_053027-001/20240228_053027.0',
#               'OPDs_PASSATA/OPD/20240302_000411-003/20240302_000411.0',
#               'OPDs_PASSATA/OPD/20240313_133532-004/20240313_133532.0',
#               'OPDs_PASSATA/OPD/20240228_112033-002/20240228_112033.0')
opds_dir=('OPDs_PASSATA/OPD/WS/JQ1/20240515_163822',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_091216',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_100705',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_103452',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_105822',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_111658',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_113534',
          'OPDs_PASSATA/OPD/WS/JQ1/20240517_121247',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_181033',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_183817',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_190418',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_192251',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_194126',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_195959',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_201835',
          'OPDs_PASSATA/OPD/WS/JQ2/20240517_203708',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_182041/20240509_182041.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_183915/20240509_183915.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_191620/20240509_191620.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_193453/20240509_193453.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_195327/20240509_195327.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_201200/20240509_201200.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_203033/20240509_203033.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_204907/20240509_204907.0',
          'OPDs_PASSATA/OPD/WS/JQM/20240509_210742/20240509_210742.0',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-001/JQ3/20240521_200540',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-004/JQ3/20240521_181105',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-002/JQ3/20240521_213115',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-005/JQ3/20240521_222747',
          'OPDs_PASSATA/OPD/WS/JQ3/JQ3-20240523T090047Z-003/JQ3/20240521_210334',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_190439/JQ3/20240527_190439',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_195648/JQ3/20240527_195648',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_204843/JQ3/20240527_204843',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_214043/JQ3/20240527_214043',
          'OPDs_PASSATA/OPD/WS/JQ3/20240527_223245/JQ3/20240527_223245',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_161522/JQ4/20240528_161522',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_163416/JQ4/20240528_163416',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_165414/JQ4/20240528_165414',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_171307/JQ4/20240528_171307',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_173157/JQ4/20240528_173157',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_175246/JQ4/20240528_175246',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_181244/JQ4/20240528_181244',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_183130/JQ4/20240528_183130',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_185018/JQ4/20240528_185018',
          'OPDs_PASSATA/OPD/WS/JQ4/20240528_190906/JQ4/20240528_190906')


#%%
for dir_nb in range(len(opds_dir)):
            
    opd_set = os.path.basename(fdir_res / opds_dir[dir_nb] ).split('.')[0]
    # opd_set='toto'
    print(fdir_res / opd_set)
    
    #%%
    """
    ### Read psf files & create profiles
    """
    
    #  TODO  trouver un moyen de securiser la selection des fichiers
    
    file_lst = sorted(
        recursive_search(fdir_res / opd_set), key=os.path.getmtime)
    # , reverse=True
    file_psf = [x for x in file_lst if 'ao_corr_psf_202406' in x]
    file_cro = [x for x in file_lst if 'ao_corr_coro_psf_202406' in x]
    
    print("psf :", file_psf)
    print("psf coro :", file_cro)

    base_psf = os.path.basename(file_psf[0]).split('.')[0]
    base_cro = os.path.basename(file_cro[0]).split('.')[0]
    
    Int_D0 = fits.getdata(fdir_res / opd_set / file_psf[0])
    Int_D  = fits.getdata(fdir_res / opd_set / file_cro[0])
    
    head_psf = fits.getheader(fdir_res / opd_set / file_psf[0])
    
    lam_min = head_psf['LMIN']
    lam_stp = head_psf['LSTP']
    lam_itv = head_psf['LITV']
    lam_lst = np.arange(lam_min,lam_min+(lam_itv+1)*lam_stp,lam_stp)
    nL = len(lam_lst)
    
    nImg = head_psf['NIMG']
    D = head_psf['DIAM']
    mB = head_psf['SFPM']
      
    # stackoveflow...
    a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
    b_ = a_.copy()
    aa, bb = np.meshgrid(a_, b_)
    rad_pix = np.sqrt(aa**2 + bb**2)
    rad_mas = rad_pix * (mas2rad * 58.393 / (D * 1e9) )
    
    #  angular separation
    aS = np.arange(nImg//2) * (mas2rad * 58.393 / (D * 1e9) )
    
    # if lambda of interest / D
    rW_mas = (lam_c / D) * mas2rad
            
    # filename of the plot
    fname_images_svg = 'ao_corrected_coro_psf_'+'.svg'
    fname_images_pdf = 'ao_corrected_coro_psf_'+'.pdf'
    
    # filepath for the direct and coronagraphic images
    fpath_images_svg = fdir_plt / opd_set / fname_images_svg
    fpath_images_pdf = fdir_plt / opd_set / fname_images_pdf
    
    # index of wvl to display
    iD = [0,nL//4,nL//2,nL*3//4,nL-1]
    
    # boundaries for the images in log scale
    vmin0 = -5
    vmax0 = 0
    
    fig = plt.figure(4, figsize=(16,6))
    plt.clf()
    plt.tight_layout()
    plt.suptitle('ao corrected psf (top) vs ao corrected coro. psf (bottom)')
    
    grid = AxesGrid(fig, 111,
            nrows_ncols=(2, 5),
            axes_pad=0.3,
            cbar_mode='single',
            cbar_location='right',
            cbar_pad=0.2
            )
    
    for i in range(5):
        
        im = grid[i].imshow(
            np.log10(Int_D0[iD[i],:]), vmin=vmin0, vmax=vmax0, cmap='inferno')
        grid[i].set_title(str(int(lam_lst[iD[i]]*1e9+.1))+'nm')
        
        im = grid[i+5].imshow(
            np.log10(Int_D[iD[i],:]), vmin=vmin0, vmax=vmax0, cmap='inferno')
        # grid[i+1+5].set_title('coro. psf')
        
    # colorbar
    cbar = grid[0].cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Intensity in log scale', rotation=270)
    
    # plt.savefig(fpath_images_svg)
    # plt.savefig(fpath_images_pdf)
    
    # if i==5:
    plt.show()
    # else : 
    # plt.close()
    # plt.close()

