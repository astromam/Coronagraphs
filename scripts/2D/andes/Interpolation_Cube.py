"""
Interpolation of data Cube of ANDES
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import scipy 
import astropy.io.fits as fits

# Load data
data_cube_nyq = fits.open('/Users/asimonnin/Desktop/PhD/Andes/data/ANDES/results/Results_for_Interp/mode=MHHR_cube_nyquist_150_test_MHR_spatiale_COMP_ahdbéjbz.fits')

data_cube_elt = fits.open('/Users/asimonnin/Desktop/PhD/Andes/data/ANDES/results/Results_for_Interp/img_cube_elt_wv=0001_R=005000_nLam=003196_150_planet_test.fits')
data_cube_HR = fits.open('/Users/asimonnin/Desktop/PhD/Andes/data/ANDES/results/Results_for_Interp/img_cube_elt_wv=0001_R=100000_nLam=063909_150.fits')


data_cube_nyq = data_cube_nyq[1].data
data_cube_elt = data_cube_elt[1].data
data_cube_HR = data_cube_HR[0].data

data_cube_MHR_no_OPD = fits.open('/Users/asimonnin/Desktop/PhD/Andes/data/ANDES/results/Results_for_Interp/img_cube_elt_wv=0001_R=005000_nLam=003196_150_planet_test_HRSPAT_NO_OPD.fits')

data_cube_MHR_no_OPD = data_cube_MHR_no_OPD[1].data

#%%

plt.imshow(np.log10(data_cube_elt[1000,:,:]),vmin=-5)
plt.colorbar()
plt.show()
# %%
plt.imshow(np.log10(data_cube_nyq[1000,:,:]),vmin=-5)
plt.colorbar()
plt.show()
# %%

def wavelength_grid(wave_min, wave_max, wave_res):
    '''
    Create a constant resolution wavelength grid

    Parameters
    ----------
    wave_min : float
        Minimum wavelength, in meters

    wave_max : float
        Maximum wavelength, in meters

    wave_res : float
        Spectral resolution

    Returns
    -------
    wave : array
        Wavelength grid, in meters

    dwave : array
        Spectral bins grid, in meters
    '''

    print('Compute wavelength grid')
    
    wave  = [wave_min]
    dwave = []
    done  = False
    while not done:
        cw = wave[-1]
        dw = cw / wave_res
        nw = cw + dw

        dwave.append(dw)
        wave.append(nw)

        if nw > wave_max:
            done = True

    wave  = np.array(wave)[:-1]
    dwave = np.array(dwave)

    return wave, dwave
    
#%%
wave_min = 950e-9
wave_max = 1800e-9


wavelength_grid_LHR = wavelength_grid(wave_min, wave_max, 3703)
wavelength_grid_HR = wavelength_grid(wave_min, wave_max, 100000)

len(wavelength_grid_HR)

wavelength_grid_HR = np.array(wavelength_grid_HR)

for i in range(len(wavelength_grid_LHR[0])):
    ind = np.where(wavelength_grid_HR[0] == wavelength_grid_LHR[0][i])
    print(ind)
#%%

# #%%
# interp_cube_elt_2 = scipy.interpolate.interp1d(wavelength_grid_MHR[0], data_cube_nyq, axis=0, fill_value='extrapolate')(wavelength_grid_HR[0])

# # interp_cube_elt = scipy.interpolate.interp1d(wavelength_grid_MHR[0], data_cube_MHR_no_OPD, axis=0, fill_value='extrapolate')(wavelength_grid_HR[0])
# # %%

# mean_HR_2 = np.mean(interp_cube_elt_2, axis=(1, 2))
# mean_MHR = np.mean(data_cube_nyq, axis=(1, 2))
# mean_HR = np.mean(data_cube_HR, axis=(1, 2))
# #%%
# plt.plot(wavelength_grid_HR[0] * 1e9, mean_HR_2, label='HR interpolated')
# plt.plot(wavelength_grid_MHR[0] * 1e9, mean_MHR, label='MHR original')


# # plt.plot(wavelength_grid_HR[0] * 1e9, mean_HR/mean_HR_2, label='HR_interpolated')

# # plt.plot(wavelength_grid_MHR[0] * 1e9, mean_MHR, label='MHR')
# plt.xlabel('Wavelength [nm]')
# plt.ylabel('Mean intensity')
# plt.legend()
# plt.show()
# # %%

# plt.imshow(np.log10(interp_cube_elt_2[3000,:,:]),vmin=-5)
# plt.title('Image at wl = ' + str(wavelength_grid_HR[0][3000]*1e9) + 'nm')
# plt.colorbar()
# plt.show()

# plt.imshow(np.log10(data_cube_nyq[146,:,:]),vmin=-5)
# plt.title('Image at wl = ' + str(wavelength_grid_MHR[0][146]*1e9) + 'nm')
# plt.colorbar()
# plt.show()

# %%

diff_1 = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_1.npy')
diff_2 = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_2.npy')
diff_3 = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_3.npy')
diff_4 = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_4.npy')

diff_1_ELT = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_1_ELT.npy')
# diff_2_ELT = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_2.npy')
diff_3_ELT = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_3_ELT.npy')
diff_4_ELT = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_4_ELT.npy')
diff_5_new = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_5_new.npy')
diff_4_new = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_4_new.npy')
diff_3_new = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_3_new.npy')
diff_3_ELT_new = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_3_ELT_new.npy')
diff_4_ELT_new = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_4_ELT_new.npy')
diff_5_ELT_new = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Diff_5_ELT_new.npy')
# %%

mean_frac= np.load('/Users/asimonnin/Desktop/PhD/Andes/results/mean_frac.npy')
median_frac= np.load('/Users/asimonnin/Desktop/PhD/Andes/results/median_frac.npy')



## faire plot mediane et moyenne en fonction de la longueur d'onde 

#%%

plt.scatter(wavelength_grid_HR[0],abs(median_frac))
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Median fractional difference (in %)')
plt.xlabel('Wavelength [m]')
# plt.ylim(10**-20,10)
plt.show()

plt.scatter(wavelength_grid_HR[0],abs(mean_frac))
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Mean fractional difference  (in %)')
plt.xlabel('Wavelength [m]')
plt.show()
# %%

mean_HR = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Mean_HR.npy')
mean_LHR = np.load('/Users/asimonnin/Desktop/PhD/Andes/results/Mean_LHR_new.npy')

plt.plot(mean_HR)
plt.plot(mean_LHR)
plt.show()
# %%
