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

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #  mdiaye 15!

avoid_k = True


#%%
"""
### scaling
"""
# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

# angular separation and ring width of interest in mas
as_oi = 20.
rW_mas = 7.


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots
    
# elif user == 'toto':

was_donow = '20251106160002'

fdir_res = fdir_res / was_donow
fdir_plt = fdir_plt / was_donow

# new ref 1 kHz 01/10/2025
# 0.89 0.35 3.9 20251001085445 ruane2018 YJH

# 0.89/0.35/3.9 + Lyot stop azimutal offset
# 20251107134453 2 pixels / 0.5% D 
# 20251107134538 4 pixels / 1.0% D 
# 20251107134615 8 pixels / 2.0% D 

# 0.89/0.35/3.9 + Lyot stop azimutal offset
# 20251106103751 2 pixels / 0.5% D
# 20251106103913 4 pixels / 1.0% D
# 20251106104005 8 pixels / 2.0% D

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

#%%
# root = 'OPDs_PASSATA/OPD/WS/'
# opds_dir=(root+'ocam2k','perfect')
# opds_dir=(root+'alice','perfect')
# opds_dir=(root+'cam_500us','perfect')

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
#           root+'20250707_114321.0',
#           'perfect')
root = 'OPDs_PASSATA/OPD/WS/1kHzVarWS/'
opds_dir=(root+'1',
          root+'2',
          root+'3',
          root+'4',
          root+'5',
          root+'6',
          root+'7',
          root+'8',
          root+'9',
          root+'10','perfect')


#%%

# """
# ### Read psf files & create profiles
# """

for dir_nb in range(len(opds_dir)):
        
    opd_set = os.path.basename(fdir_res / opds_dir[dir_nb] ).split('.')[0]

    print(fdir_res / opd_set)
    
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
        
        Int_D0_psf_avg = fits.getdata(fdir_res / opd_set / file_psf)
        Int_D_psf_avg  = fits.getdata(fdir_res / opd_set / file_cro)
        
        head_psf = fits.getheader(fdir_res / opd_set / file_psf)
        
        lam_min = head_psf['LMIN']
        lam_stp = head_psf['LSTP']
        lam_itv = head_psf['LITV']
        lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)
        if avoid_k:
            lam_lst = lam_lst[np.where(lam_lst < 1860e-9)]
        lmx = str(int(np.rint(lam_lst[-1]*1e9)))
        lmn = str(int(np.rint(lam_lst[0]*1e9)))
        nL = len(lam_lst)

        if np.median(lam_lst) > 1e-6:
            wvl = 'YJH'
        else:
            wvl = 'RIZ'
        
        nImg = head_psf['NIMG']
        pscale = head_psf['PSCL']
        D = head_psf['DIAM']
        mB = head_psf['SFPM'] # - 0.5
        lam_c = head_psf['LMBD']
        lamCD2mas = (lam_c / D) * mas2rad

        # print(lam_c)
        hlf_fov = nImg * pscale / 2.
        # stackoveflow...
        a_ = np.linspace(-(np.floor(nImg-1)/2), np.floor(nImg-1)/2, nImg)
        b_ = a_.copy()
        aa, bb = np.meshgrid(a_, b_)
        rad_pix = np.sqrt(aa**2 + bb**2)
        rad_mas = rad_pix * (mas2rad * hlf_fov / (D * 1e9) )
        
        #  angular separation
        aS = np.arange(nImg//2) * (mas2rad * hlf_fov / (D * 1e9) )
                
        ratio_prf = np.zeros([nL, nImg//2])
        Int_D0_prf_avg = ratio_prf.copy()
        Int_D_prf_avg = ratio_prf.copy()
        Int_D0_prf_std = ratio_prf.copy()
        Int_D_prf_std = ratio_prf.copy()
        Int_D0_prf_min = ratio_prf.copy()
        Int_D_prf_min = ratio_prf.copy()
        Int_D0_prf_max = ratio_prf.copy()
        Int_D_prf_max = ratio_prf.copy()

        Int_elt_prf_avg = ratio_prf.copy()
        Int_elt_prf_std = ratio_prf.copy()
        Int_elt_prf_min = ratio_prf.copy()
        Int_elt_prf_max = ratio_prf.copy()

        for i in range(nL):
                    
            for p in range(nImg//2):
                
                ring_val = np.where(np.abs(rad_mas-aS[p])<=rW_mas/2)
                Int_D0_prf_avg[i,p] = np.mean(Int_D0_psf_avg[i,:,:][ring_val])
                Int_D_prf_avg[i,p] = np.mean(Int_D_psf_avg[i,:,:][ring_val])
                Int_D0_prf_std[i,p] = np.std(Int_D0_psf_avg[i,:,:][ring_val])
                Int_D_prf_std[i,p] = np.std(Int_D_psf_avg[i,:,:][ring_val])
                Int_D0_prf_min[i,p] = np.min(Int_D0_psf_avg[i,:,:][ring_val])
                Int_D_prf_min[i,p] = np.min(Int_D_psf_avg[i,:,:][ring_val])
                Int_D0_prf_max[i,p] = np.max(Int_D0_psf_avg[i,:,:][ring_val])
                Int_D_prf_max[i,p] = np.max(Int_D_psf_avg[i,:,:][ring_val])
                

        fname_psf_contrast = ('contrast_profile_L0toD_' + opd_set + '_' +
                              base_psf +'_'+
                              str(int(as_oi))+'mas_'+str(int(rW_mas))+'mas_'+
                              lmn+'_'+lmx+ 'nm.fits')
        fname_cro_contrast = ('contrast_profile_L0toD_' + opd_set + '_' +
                              base_cro +'_'+
                              str(int(as_oi))+'mas_'+str(int(rW_mas))+'mas_'+
                              lmn+'_'+lmx+ 'nm.fits')
        
        fpath_psf_contrast =  fdir_res / opd_set / fname_psf_contrast 
        fits.writeto(fpath_psf_contrast, Int_D0_prf_avg, head_psf,
                     overwrite=True)
        fits.append(fpath_psf_contrast, Int_D0_prf_std, head_psf)
        fits.append(fpath_psf_contrast, Int_D0_prf_min, head_psf)
        fits.append(fpath_psf_contrast, Int_D0_prf_max, head_psf)
        
        fpath_cro_contrast =  fdir_res / opd_set / fname_cro_contrast 
        fits.writeto(fpath_cro_contrast, Int_D_prf_avg, head_psf,
                     overwrite=True)
        fits.append(fpath_cro_contrast, Int_D_prf_std, head_psf)
        fits.append(fpath_cro_contrast, Int_D_prf_min, head_psf)
        fits.append(fpath_cro_contrast, Int_D_prf_max, head_psf)

        prf_as_oi_mas = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))
       
        prf_as_oi_mas = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))
        
        if opd_set != 'perfect':
            
            file_psf_prf = [x for x in file_lst if 'ao_corr_psf_profile_' in x]
            file_cro_prf = [x for x in file_lst
                            if 'ao_corr_coro_psf_profile_' in x]
    
            if opd_set == 'perfect':
                
                file_psf_prf = [x for x in file_lst if 'wonoise_psf_profile_' in x]
                file_cro_prf = [x for x in file_lst
                                if 'wonoise_coro_psf_profile_' in x]
    
            
            print("psf profile:", file_psf_prf)
            print("psf coro profile:", file_cro_prf)
        
            Int_D0_psf_prf = fits.getdata(fdir_res / opd_set / file_psf_prf[0])
            Int_D_psf_prf  = fits.getdata(fdir_res / opd_set / file_cro_prf[0])
        
                
        #%%
        """
        plot profiles
        """
        
        # # plot of the gain vs wvl
        # plt.figure(1, (8, 4.5))
        # plt.tight_layout()
        # plt.xlabel('wavelength (nm)')#[$\lambda$/D]')
        # plt.ylabel('gain (log)')
        # plt.yscale('log')
        # plt.title('gain at 25 mas vs wvl')
        # plt.grid(True)
        # plt.ylim(9e-1, 2e3)
    
        # g_val = Int_D0_prf_avg[:,prf_as_oi_mas]/Int_D_prf_avg[:,prf_as_oi_mas]
        # print(np.min(g_val), np.max(g_val))    
        # plt.plot(lam_lst*1e9, g_val)
        
        # if slc:
        #     fname_gain_as_oi_mas = (
        #         'gain_'+str(int(as_oi))+'mas_Lbd2D_' + base_cro)
        # else:
        #     fname_gain_as_oi_mas = (
        #         'gain_'+str(int(as_oi))+'mas_L0toD_' + base_cro)
        
        # fpath_gain_as_oi_mas_pdf = (fdir_plt / opd_set /
        #                             (fname_gain_as_oi_mas + '.pdf'))
        # plt.savefig(fpath_gain_as_oi_mas_pdf)
        # plt.show()
        
        #%%
        # fix mask radius before plots
        
        # # plot of the contrast vs radial distance
        # colors = plt.cm.rainbow(np.linspace(0,1,nL))
        # plt.figure(2, (8, 4.5))
        # plt.tight_layout()
        # plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
        # plt.ylabel('contrast')
        # plt.yscale('log')
        # plt.title('contrasts(wvl) vs radial distance')
        # plt.grid(True)
        # plt.ylim(1e-5, 2e0)
    
        # for i in range(nL):
        
        #     # AO corrected psf 
        #     plt.plot(aS, Int_D0_prf_avg[i,:], color=colors[i], alpha=0.5)
            
        #     # AO corrected coronagraphic psf 
        #     plt.plot(aS, Int_D_prf_avg[i,:], label=str(int(lam_lst[i]*1e9))+'nm',
        #               color=colors[i])
            
        # # Focal plane mask boundary
        # x = np.arange(0.0, mB/2, 0.01)
        # plt.axvline(as_oi, color='k', ls='--')
        # plt.legend(fontsize='xx-small', ncols=5)
        # # Focal plane mask grey area
        # plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lam_c/rad2mas, color='gray',
        #                   alpha=0.3)
        
        # if slc:
        #     fname_contrast_profile = ('contrast_profile_Lbd2D_' + base_cro)
            
        # else:
        #     fname_contrast_profile = ('contrast_profile_L0toD_' + base_cro)

        # fpath_contrast_profile_pdf = (fdir_plt / opd_set /
        #                               (fname_contrast_profile + '.pdf'))
        # plt.savefig(fpath_contrast_profile_pdf)
        # plt.show()

        #%%
        # index of wvl to display
        iD = [0,nL//4-1,nL//2-1,nL*3//4-1,nL-1]

        if opd_set != 'perfect':
            
            # plot of the radial profiles
            colors = plt.cm.rainbow(np.linspace(0,1,nL))
            plt.figure(3, (8, 4.5))
            plt.tight_layout()
            plt.ylim(1e-5, 2e0)
            plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
            plt.ylabel('Intensity in log scale')
            plt.yscale('log')
            plt.title(f'Coronagraph configuration in {wvl} band')
            plt.grid(True)
           
            for i in range(5):
            # for i in range(0,nL,2):
            # for i in range(6):

                # AO corrected psf 
                # plt.plot(aS, Int_D0_psf_prf[iD[i],1,:], color=colors[iD[i]], alpha=0.5,
                #           ls='--')
                
                # AO corrected coronagraphic psf 
                plt.plot(aS, Int_D_psf_prf[iD[i],1,:],
                         label=str(int(np.rint(lam_lst[iD[i]]*1e9)))+'nm',
                          color=colors[iD[i]])
                
            # Focal plane mask boundary
            x = np.arange(0.0, mB/2, 0.01)
            plt.axvline(as_oi, color='k', ls='--')
            plt.legend(fontsize='xx-small', ncols=6)
            # Focal plane mask grey area
            plt.fill_between(x *lamCD2mas, 0, mB/2/ 38.54*lam_c/rad2mas, color='gray',
                              alpha=0.3)
            
            fname_contrast_profile = ('intensities_profiles_' + base_cro)
            fpath_contrast_profile_svg = (fdir_plt / opd_set /
                                          (fname_contrast_profile +'_'+lmn+'_'+lmx+ '.svg'))
            fpath_contrast_profile_pdf = (fdir_plt / opd_set /
                                          (fname_contrast_profile +'_'+lmn+'_'+lmx+ '.pdf'))
            plt.savefig(fpath_contrast_profile_svg)
            plt.savefig(fpath_contrast_profile_pdf, bbox_inches='tight', pad_inches=0.1)
            plt.show()
    
    
            # plot of the radial profiles
            colors = plt.cm.rainbow(np.linspace(0,1,nL))
            plt.figure(4, (8, 8))
            # plt.title('intensity@1600nm with/without coro. vs radial distance')
            plt.tight_layout()
            plt.xlim(1e0,1e2)
            plt.ylim(1e-5, 2e0)
            plt.xlabel('Angular separation [mas]')#[$\lambda$/D]')
            plt.ylabel('intensity')
            plt.yscale('log')
            plt.xscale('log')
            plt.title('intensities(wvl) vs radial distance')
            plt.grid(True)
           
            if True:
                
                i=np.argmin(np.abs(lam_lst-lam_c))
                # AO corrected psf 
                plt.plot(aS, Int_D0_psf_prf[i,1,:],
                          label='no coro', color=colors[0])
                
                # AO corrected coronagraphic psf 
                plt.plot(aS, Int_D_psf_prf[i,1,:],
                          label='with coro', color=colors[-1])
                
            # Focal plane mask boundary
            x = np.arange(0.0, mB/2, 0.01)
            plt.axvline(as_oi, color='k', ls='--')
            plt.legend(fontsize='xx-small', ncols=5)
            # Focal plane mask grey area
            fill_max=mB/2/ 38.54*lam_c/rad2mas
            plt.fill_between(x*lamCD2mas, 1e-5, fill_max, color='gray', alpha=0.3)
            
            fname_contrast_profile = ('intensities_profiles_1600nmOnly_'+base_cro)
            fpath_contrast_profile_pdf = (fdir_plt / opd_set /
                                          (fname_contrast_profile +'_'+lmn+'_'+lmx+ '.pdf'))
            plt.savefig(fpath_contrast_profile_pdf)
            plt.show()
        
