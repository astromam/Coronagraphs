# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:59:22 2024

@author: asp, mndiaye, asimonnin
"""

# compute contrast profiles vs angular separation for multiple wavelength

#%%
"""
### Initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from pathlib import Path
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
    # dir name where to find results and plots of a common script run date
    was_donow = '20240729170321'
    fdir_res = fdir_res / was_donow
    fdir_plt = fdir_plt / was_donow

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
    file_psf = [x for x in file_lst if 'ao_corr_psf_202407' in x]
    file_cro = [x for x in file_lst if 'ao_corr_coro_psf_202407' in x]
    
    print("psf :", file_psf)
    print("psf coro :", file_cro)

    base_psf = os.path.basename(file_psf[0]).split('.')[0]
    base_cro = os.path.basename(file_cro[0]).split('.')[0]
    
    Int_D0_psf_avg = fits.getdata(fdir_res / opd_set / file_psf[0])
    Int_D_psf_avg  = fits.getdata(fdir_res / opd_set / file_cro[0])
    
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
    
    ratio_prf = np.zeros([nL, nImg//2])
    Int_D0_prf_avg = ratio_prf.copy()
    Int_D_prf_avg = ratio_prf.copy()
    
    for i in range(nL):
        
        if slc:
            rW_mas = (lam_lst[i] / D) * mas2rad
        
        for p in range(nImg//2):
            
            ring_val = np.where(np.abs(rad_mas-aS[p])<=rW_mas/2)
            Int_D0_prf_avg[i,p] = np.mean(Int_D0_psf_avg[i,:,:][ring_val])
            Int_D_prf_avg[i,p] = np.mean(Int_D_psf_avg[i,:,:][ring_val])
        
    if slc:
        
        fname_psf_contrast = ('contrast_profile_Lbd2D_' + opd_set + '_' +
                              base_psf + '.fits')
        fname_cro_contrast = ('contrast_profile_Lbd2D_' + opd_set + '_' +
                              base_cro + '.fits')

    else:
        
        fname_psf_contrast = ('contrast_profile_L0toD_' + opd_set + '_' +
                              base_psf + '.fits')
        fname_cro_contrast = ('contrast_profile_L0toD_' + opd_set + '_' +
                              base_cro + '.fits')
    
    fpath_psf_contrast =  fdir_res / opd_set / fname_psf_contrast 
    fits.writeto(fpath_psf_contrast, Int_D0_prf_avg, head_psf, overwrite=True)

    fpath_cro_contrast =  fdir_res / opd_set / fname_cro_contrast 
    fits.writeto(fpath_cro_contrast, Int_D_prf_avg, head_psf, overwrite=True)

    prf_as_oi_mas = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))
    
    file_psf_prf = [x for x in file_lst if 'ao_corr_psf_profile_202407' in x]
    file_cro_prf = [x for x in file_lst if 'ao_corr_coro_psf_profile_202407' in x]
    
    print("psf profile:", file_psf_prf)
    print("psf coro profile:", file_cro_prf)

    Int_D0_psf_prf = fits.getdata(fdir_res / opd_set / file_psf_prf[0])
    Int_D_psf_prf  = fits.getdata(fdir_res / opd_set / file_cro_prf[0])
    
            
    #%%
    """
    plot profiles
    """
    
    # plot of the gain vs wvl
    plt.figure(1, (8, 4.5))
    plt.tight_layout()
    plt.xlabel('wavelength (nm)')#[$\lambda$/D]')
    plt.ylabel('gain (log)')
    plt.yscale('log')
    plt.title('gain at 25 mas vs wvl')
    plt.grid(True)
    plt.ylim(9e-1, 2e3)

    g_val = Int_D0_prf_avg[:,prf_as_oi_mas]/Int_D_prf_avg[:,prf_as_oi_mas]
    print(np.min(g_val), np.max(g_val))    
    plt.plot(lam_lst*1e9, g_val)
    
    if slc:
        fname_gain_as_oi_mas = (
            'gain_'+str(int(as_oi))+'mas_Lbd2D_' + base_cro)
    else:
        fname_gain_as_oi_mas = (
            'gain_'+str(int(as_oi))+'25mas_L0toD_' + base_cro)
    
    fpath_gain_as_oi_mas_svg = (fdir_plt / opd_set /
                                (fname_gain_as_oi_mas + '.svg'))
    fpath_gain_as_oi_mas_pdf = (fdir_plt / opd_set /
                                (fname_gain_as_oi_mas + '.pdf'))
    plt.savefig(fpath_gain_as_oi_mas_svg)
    plt.savefig(fpath_gain_as_oi_mas_pdf)
    plt.show()
    
    #%%
    # fix mask radius before plots
    rW_mas = (lam_c / D) * mas2rad
    
    # plot of the contrast vs radial distance
    colors = plt.cm.rainbow(np.linspace(0,1,nL))
    plt.figure(2, (8, 4.5))
    plt.tight_layout()
    plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
    plt.ylabel('contrast')
    plt.yscale('log')
    plt.title('contrasts(wvl) vs radial distance')
    plt.grid(True)
    plt.ylim(1e-5, 2e0)

    for i in range(nL):
    
        # AO corrected psf 
        plt.plot(aS, Int_D0_prf_avg[i,:], color=colors[i], alpha=0.5)
        
        # AO corrected coronagraphic psf 
        plt.plot(aS, Int_D_prf_avg[i,:], label=str(int(lam_lst[i]*1e9))+'nm',
                  color=colors[i])
        
    # Focal plane mask boundary
    x = np.arange(0.0, mB/2, 0.01)
    plt.axvline(as_oi, color='k', ls='--')
    plt.legend(fontsize='xx-small', ncols=5)
    # Focal plane mask grey area
    plt.fill_between(x *rW_mas, 0, mB/2/ 38.54*lam_c/rad2mas, color='gray',
                      alpha=0.3)
    
    if slc:
        fname_contrast_profile = ('contrast_profile_Lbd2D_' + base_cro)
        
    else:
        fname_contrast_profile = ('contrast_profile_L0toD_' + base_cro)
    fpath_contrast_profile_svg = (fdir_plt / opd_set /
                                  (fname_contrast_profile + '.svg'))
    fpath_contrast_profile_pdf = (fdir_plt / opd_set /
                                  (fname_contrast_profile + '.pdf'))
    plt.savefig(fpath_contrast_profile_svg)
    plt.savefig(fpath_contrast_profile_pdf)
    plt.show()

     #%%   
    # plot of the radial profiles
    colors = plt.cm.rainbow(np.linspace(0,1,nL))
    plt.figure(3, (8, 4.5))
    plt.tight_layout()
    plt.ylim(1e-5, 2e0)
    plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
    plt.ylabel('intensity')
    plt.yscale('log')
    # plt.title('intensities(wvl) vs radial distance')
    plt.grid(True)
   
    for i in range(0,nL,2):
    
        # AO corrected psf 
        plt.plot(aS, Int_D0_psf_prf[i,1,:], color=colors[i], alpha=0.5,
                 ls='--')
        
        # AO corrected coronagraphic psf 
        plt.plot(aS, Int_D_psf_prf[i,1,:], label=str(int(lam_lst[i]*1e9))+'nm',
                 color=colors[i])
        
    # Focal plane mask boundary
    x = np.arange(0.0, mB/2, 0.01)
    plt.axvline(as_oi, color='k', ls='--')
    plt.legend(fontsize='xx-small', ncols=5)
    # Focal plane mask grey area
    plt.fill_between(x *rW_mas, 0, mB/2/ 38.54*lam_c/rad2mas, color='gray',
                     alpha=0.3)
    
    fname_contrast_profile = ('intensities_profiles_' + base_cro)
    fpath_contrast_profile_svg = (fdir_plt / opd_set /
                                  (fname_contrast_profile + '.svg'))
    fpath_contrast_profile_pdf = (fdir_plt / opd_set /
                                  (fname_contrast_profile + '.pdf'))
    plt.savefig(fpath_contrast_profile_svg)
    plt.savefig(fpath_contrast_profile_pdf)
    plt.show()
    
