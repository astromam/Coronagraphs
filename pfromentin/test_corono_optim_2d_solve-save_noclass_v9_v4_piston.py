"""
Created on Mon Dec  9 12:52:33 2024

@author: mndiaye
"""
import numpy as np
import time
import os
from pathlib import Path
import sys
sys.path.insert(0, "/user/home/fromentp/Code")
import corono as coro


from astropy.io import fits
syst = sys.platform

from utils_corono_v4 import (
    sft_qrt, isft_qrt, sft_hlf, isft_hlf,
    compute_corono_field_2d_qrt,
    compute_response_matrices,
    compute_response_matrices_qrt,
    GUROBI_FIXED_PARAMS,
)

slvLogToConsole   = GUROBI_FIXED_PARAMS["slvLogToConsole"]
slvCrossover      = GUROBI_FIXED_PARAMS["slvCrossover"]
slvMethod         = GUROBI_FIXED_PARAMS["slvMethod"]
slvSparse         = GUROBI_FIXED_PARAMS["slvSparse"]
allLogToConsole   = GUROBI_FIXED_PARAMS["allLogToConsole"]
MinIsland         = GUROBI_FIXED_PARAMS["MinIsland"]
Binarity          = GUROBI_FIXED_PARAMS["Binarity"]
FirstDerGlobalLim = GUROBI_FIXED_PARAMS["FirstDerGlobalLim"]
BinarityReg       = GUROBI_FIXED_PARAMS["BinarityReg"]
nProgRef          = GUROBI_FIXED_PARAMS["nProgRef"]
CtrBtwnPix        = GUROBI_FIXED_PARAMS["CtrBtwnPix"]
CtrBtwnPix2       = GUROBI_FIXED_PARAMS["CtrBtwnPix2"]
Pupil2dSym        = GUROBI_FIXED_PARAMS["Pupil2dSym"]
ImPart            = GUROBI_FIXED_PARAMS["ImPart"]

# ============================================================================================
# GESTION ET IMPORTATION DES SOLVEURS D'OPTIMISATION
# ============================================================================================

try:
    from scipy.optimize import linprog
    scipy_lp_dispo = True
    print("Solveur SciPy (linprog) pret.")
except ImportError:
    scipy_lp_dispo = False
    print("Impossible d'importer SciPy linprog.")

# 2. Solveur Gurobi classique (gurobipy)
try:
    import gurobipy as gb
    gb_dispo = True
    print("Interface Gurobi (gurobipy) chargee avec succes.")
except ImportError:
    gb_dispo = False
    print("Gurobi n'est pas installe ou la licence est absente (gb is False).")

#%% parameters
t_start_global = time.perf_counter()
#Variables
RUN_ID = "0109-testv1"
SegRobustness = True
if SegRobustness:
    amplitude = 1e-3
rMask = 2.9       # Mask radius = rho0 - 1
cDarkHole = 7   # Contrast imposed in the dark region if problem_name = 'MaxTau'
OD_LS = 0.90
gray = 'False'
oversampling = 1
"""
Fixed Parameters
"""
nPup0 = 64       # PUPIL SIZE Default
corono_name  = 'APLC'
pupil_name   = 'hwo' # 'vlt' or 'sbr' or 'hwo'
problem_name = 'MaxTau' # 'MaxContrastL1' # 'MaxContrastLinf' # 'MaxTau'
solver       = 'gurobipy' #,'stdgrb' #  'gurobipy', 'scipy.linprog'
nFPM = 50                       # FPM DIAMETER IN PIXELS
Fmax2d = 35 #   45 #22.5
nImg2d = 140 #  90 #45         

rho0 =  3.0
rho1 = 10.0 
tau   = 0.6 # tau (integrated Pupil transmission) if problem_name = 'MaxContrast'
bw   = 0                    #bandwidth
nlam = 1                    #number of wavelengths

do_fits = True
print("\n" + "="*50)
print("PARAMETERS...")
print("="*50 + "\n")
print(f'Numero ID={RUN_ID}')
print(f'Robustesse aux aberrations de piston activee ? --> {SegRobustness}')
if SegRobustness:
    print(f'Amplitude des aberrations en piston = {amplitude} RMS')
print(f'Taille pupille=({nPup0},{nPup0})')
print(f'Taille Lyot Stop OD={OD_LS}')
print(f'IWA={rho0}')
print(f'OWA={rho1}')

print(f'Problem name={problem_name}')
print(f'Contraste voulu={cDarkHole}')
#%%
"""
### Functions
"""
dtype0 = 'float64'
if ImPart is True:
    dtype0 = 'complex128'

if solver != 'gurobipy' and solver != 'stdgrb':
    solver = 'scipy'

ncorono = 1
neps = 1 

# wavelengths
lam0 = 1.        
dlam = bw*lam0
lam_t = np.linspace(lam0-dlam/2*(nlam>1),lam0+dlam/2,nlam)
    
# Focal plane mask
mask2d = coro.utils.uniform_disk(nFPM, nFPM/2., CtrBtwnPix=CtrBtwnPix)
mask2d_qrt = mask2d[nFPM//2:,nFPM//2:]

# mask size at a given wavelength for SFT
mB_t  = 2.*rMask*(lam0/lam_t)


# ==========================================
# --- SAMPLING VERIFICATION ---
# ==========================================
sampling_DZ = nImg2d / Fmax2d
sampling_FPM = nFPM / (2.0 * rMask)
print(f"Dark Zone Grid (SFT Outer) : {sampling_DZ:.2f} px per lbd/D")
print(f"Mask Grid (SFT Inner)      : {sampling_FPM:.2f} px per lbd/D")


#%%
"""
File reading for Pupil and Lyot stop
"""
# ============================================================================================
# File reading for Pupil, Pupil AB and Lyot stop
# ============================================================================================

fdir = Path("/user/home/fromentp/Code/pupils").resolve()
fdir_ls = Path("/user/home/fromentp/Code/ls").resolve()
fdir_sav = Path("/user/home/fromentp/Code/Results").resolve() / pupil_name
sim_case = 'test' 
fdir_sav.mkdir(parents=True, exist_ok=True)

#%%
def get_filename(corono=None):
    if corono is None:
        pass
    else:
        params = corono.params.copy()
        params = coro.utils.update_params(params)
                
        if problem_name == 'MaxTau':
            str_opt = '_C={cDarkHole:.1f}'
        elif problem_name == 'MaxContrastL1' or problem_name == 'MaxContrastLinf':
            str_opt = '_tau={tau:.3f}'
        else:
            raise NameError('{0}: Not an existing optimization problem!'.format(problem_name))
            
        str_SegRobustness = ''
        if SegRobustness is True:
            str_SegRobustness = '_SegRobsutness=1'
        fname_corono = corono.get_filename()             
        str_Lyot_Specs = f'_ODLS={OD_LS:.2f}_over={oversampling}'
        fname_gen_optim  = '{problem_name}' \
        + str_opt \
        + str_SegRobustness \
        + str_Lyot_Specs \
        + '_{solver}'

        id_str = f"ID={RUN_ID}"
    
        filename = f"{id_str}_{pupil_name}_{fname_corono}{fname_gen_optim.format(**params)}"
        
        return filename

#%%
"""
### scale arrays
"""
Gray_2d = 0
Ones_2d = 0
ApoCst = 0
for k in range(nProgRef):
    nPup = nPup0*2**k
    fname_pup = f'pupil={pupil_name}_nPup={int(np.round(nPup))}.fits'
    fname_lys = f'pupil={pupil_name}_nPup={int(np.round(nPup))}_OD={OD_LS}_gray={gray}_oversampling={oversampling}.fits'    
    fpath_pup = fdir / fname_pup
    fpath_lys = fdir_ls / fname_lys

    Pupil2d = np.zeros((nPup, nPup))
    LyotStop2d = np.zeros((nPup, nPup))
    Pupil2d = fits.getdata(fpath_pup)
    LyotStop2d = fits.getdata(fpath_lys)

    Pupil2d_qrt = Pupil2d[nPup//2:,nPup//2:]
    LyotStop2d_qrt = LyotStop2d[nPup//2:,nPup//2:]



    list_Pupil2d_AB_qrt = []
    if SegRobustness:
        pupil_indexed = fits.getdata(fdir / f"pupil={pupil_name}_nPup={nPup}_index.fits")
        pupil_indexed_qrt = pupil_indexed[nPup//2:, nPup//2:]
        segment_ids_qrt = np.unique(pupil_indexed_qrt)
        segment_ids_qrt = segment_ids_qrt[segment_ids_qrt > 0] # On exclut le fond noir
        
        n_segments_qrt = len(segment_ids_qrt)
        print(f'Nombre de segments contraints (dans le quadrant) = {n_segments_qrt}')

        for seg_id in segment_ids_qrt:
            mask_qrt = (pupil_indexed_qrt == seg_id).astype(float)
            pupil_complex_qrt = Pupil2d_qrt * mask_qrt
            list_Pupil2d_AB_qrt.append(pupil_complex_qrt)
    

    
    params0 = coro.to_dict(nPup=nPup, Fmax2d = Fmax2d, nImg2d=nImg2d, nFPM = nFPM,
                     rho0=rho0, rho1=rho1, cDarkHole=cDarkHole, tau=tau, 
                     CtrBtwnPix=CtrBtwnPix, CtrBtwnPix2 = CtrBtwnPix2,
                     nlam=nlam, bw=bw,
                     Pupil2d = Pupil2d, LyotStop2d = LyotStop2d,
                     List_Pupil2d_AB_qrt = list_Pupil2d_AB_qrt,
                     Pupil2dSym = Pupil2dSym, rMask=rMask,
                     problem_name = problem_name, 
                     solver = solver, 
                     corono_name = corono_name, pupil_name = pupil_name,
                     slvLogToConsole = slvLogToConsole,
                     slvCrossover = slvCrossover, slvMethod = slvMethod,
                     slvSparse = slvSparse,
                     allLogToConsole = allLogToConsole,
                     MinIsland = MinIsland, FirstDerGlobalLim = FirstDerGlobalLim,
                     Binarity = Binarity, BinarityReg = BinarityReg,
                     ImPart = ImPart, SegRobustness = SegRobustness)
    
    #%%  
    """ 
    Coronagraph defintion
    """

    if corono_name == 'APLC':
        corono0 = coro.design.APLC2d(**params0)
    else:
        raise NameError('{0}: Not an existing coronagraph!'.format(corono_name))  
    
    
    #%%
    """
    Problem definition without class
    """
    print("\n" + "="*50)
    print('definition of the variables for the problem')
    t_a = time.perf_counter()

    
    dz2d, rad2d = corono0.generate_area()
    
    xx,yy  = np.meshgrid(np.arange(nImg2d)-nImg2d//2+1/2, 
                         np.arange(nImg2d)-nImg2d//2+1/2)
    dist2d = (Fmax2d/nImg2d)*np.hypot(yy,xx)
    
    xyp = (nPup/(2*nPup))*(np.arange(2*nPup)-2*nPup//2+1/2)
    xxp, yyp  = np.meshgrid(xyp, xyp)
    
    #%%
    """
    ### Point selection (Full Format)
    """
    dist2d_qrt = dist2d[nImg2d//2:,nImg2d//2:] 
    dz2d_qrt = dz2d[nImg2d//2:,nImg2d//2:] 
    dz       = np.reshape(dz2d_qrt, ((corono0.nImg2d//2)**2))
    aaa      = np.arange((corono0.nImg2d//2)**2)
    dist     = np.reshape(dist2d_qrt, (corono0.nImg2d//2)**2)

    idx_dz = np.where(dz == 1)[0].tolist() 
    ndz = len(idx_dz)

    Pupil_vec = np.reshape(Pupil2d_qrt, ((corono0.nPup//2)**2))
    pup     = (Pupil_vec > 0.)
    bbb     = np.arange((corono0.nPup//2)**2)
    
    idx_pup = list(bbb[pup])
    npp     = len(idx_pup)
    PupilLyotStop_vec = np.reshape(Pupil2d_qrt*LyotStop2d_qrt, ((corono0.nPup//2)**2))


#%%
    neps = 0
    if problem_name == 'MaxContrastLinf':
        neps = 1
    elif problem_name == 'MaxContrastL1':
        neps = ndz*1   
        
    #%%
    TR = np.sum(Pupil_vec)
    
    nI1 = 1
    nPsiD = nI1*corono0.nlam*ndz
    
    print(f"Etape de definition des variables finies en : {time.perf_counter() - t_a:.2f}s")
    print("="*50 + "\n")
    #%%
    """
    ### Matrices Computation
    """
    print("\n" + "="*50)
    print('Computing response matrices for nominal pupil and all segments...')
    t_b = time.perf_counter()
    PsiD_re = np.zeros((npp, nPsiD))
    PsiD_im = 0
    
    PsiD_re[0:npp,0:nPsiD] = compute_response_matrices_qrt(idx_pup=idx_pup, idx_dz=idx_dz, npp=npp, ndz=ndz, 
            ImPart=False, Pupil2d_qrt=Pupil2d_qrt, mask2d_qrt=mask2d_qrt, dtype0=dtype0, corono=corono0)

    if SegRobustness:
        list_PsiD_AB_re = []
        list_PsiD_AB_im = []

        for k in range(n_segments_qrt):
            if ImPart:
                re, im = compute_response_matrices_qrt(
                    idx_pup=idx_pup, idx_dz=idx_dz, npp=npp, ndz=ndz, 
                    ImPart=ImPart, Pupil2d_qrt=list_Pupil2d_AB_qrt[k], mask2d_qrt=mask2d_qrt, dtype0=dtype0, 
                    corono=corono0)
                list_PsiD_AB_re.append(re)
                list_PsiD_AB_im.append(im)
            else:
                # Calcul purement réel des sensibilités
                re = compute_response_matrices_qrt(
                    idx_pup=idx_pup, idx_dz=idx_dz, npp=npp, ndz=ndz, 
                    ImPart=ImPart, Pupil2d_qrt=list_Pupil2d_AB_qrt[k], mask2d_qrt=mask2d_qrt, dtype0=dtype0, 
                    corono=corono0)
                list_PsiD_AB_re.append(re)
                list_PsiD_AB_im.append(0)

        print(f"Calcul termine pour {n_segments_qrt} segments.")

    PsiD0bis = 0
    PsiD0bis_t = np.zeros((ncorono))
     
    print(f"Etape calcul des matrices finie en : {time.perf_counter() - t_b:.2f}s")
    print("="*50 + "\n")
    #%%
    """
    ### Dimensions
    """
    print("\n" + "="*50)
    print("DIMENSIONS DES MATRICES...")
    print(f'dimensions PsiD_re: {np.shape(PsiD_re)}')
    print(f'dimensions PsiD_im: {np.shape(PsiD_im)}')
    print(f'PsiD axis-0: {np.shape(PsiD_re)[0]}')
    print("="*50 + "\n")
    
    #%%
    """
    ### Computation of the gurobi model
    """
    print("\n" + "="*50)
    print('GENERATING GUROBI MODEL...')
    t_c = time.perf_counter()
    if problem_name == 'MaxContrastLinf' or problem_name == 'MaxContrastL1':
        
        # Create a new model  
        model = gb.Model("LP max C new")
        
        # Create variables
        Apo = model.addMVar(npp, lb=0.0, ub=1.0, name="Apo")
        if problem_name == 'MaxContrastLinf':
            Eps = model.addMVar(neps, lb=0.0, name="Eps")
        else:
            Eps = model.addMVar(neps*corono0.nlam, lb=0.0, name="Eps")
            
        # Set objective
        model.setObjective(Eps.sum(), gb.GRB.MINIMIZE)
        
        # Add constraint:
        if ImPart:    
            model.addConstr( (PsiD_re + PsiD_im).T @ Apo + (PsiD0bis.real + PsiD0bis.imag) - Eps <= 0)
            model.addConstr( (PsiD_re - PsiD_im).T @ Apo + (PsiD0bis.real - PsiD0bis.imag) - Eps <= 0)
            model.addConstr((-PsiD_re + PsiD_im).T @ Apo +(-PsiD0bis.real + PsiD0bis.imag) - Eps <= 0)
            model.addConstr((-PsiD_re - PsiD_im).T @ Apo +(-PsiD0bis.real - PsiD0bis.imag) - Eps <= 0)

        else:
            model.addConstr( PsiD_re.T @ Apo + PsiD0bis.real - Eps <= 0)
            model.addConstr(-PsiD_re.T @ Apo - PsiD0bis.real - Eps <= 0)

        
        model.addConstr(-Apo.sum() - ApoCst  <= -tau*TR)
        
    

    else:
        # Create a new model  
        model = gb.Model("LP max tau new")
        
        # Create variables
        Apo = model.addMVar(npp, lb=0.0, ub=1.0, name="Apo")
            
        # Set objective
        model.setObjective(-Apo.sum() - ApoCst, gb.GRB.MINIMIZE)
        
        cst = (10.**(-cDarkHole/2.)/np.sqrt(2.))*corono0.Fmax2d/(corono0.nImg2d*corono0.nPup)
        Psi0 = cst*np.sum(PupilLyotStop_vec)
        
        print(f'Constante Psi0:{cst}')
        print(f'Psi0:{Psi0}')
        
        # Add constraint (Nominal):
        if ImPart:
            model.addConstr( (PsiD_re + PsiD_im).T @ Apo - Psi0 <= 0)
            model.addConstr( (PsiD_re - PsiD_im).T @ Apo - Psi0 <= 0)
            model.addConstr((-PsiD_re + PsiD_im).T @ Apo - Psi0 <= 0)
            model.addConstr((-PsiD_re - PsiD_im).T @ Apo - Psi0 <= 0)    
        else:
            model.addConstr( PsiD_re.T @ Apo - Psi0 <= 0)
            model.addConstr(-PsiD_re.T @ Apo - Psi0 <= 0)
        
        if SegRobustness:
            n_segments_total = 19 
            
            Psigrad = Psi0 / (np.sqrt(n_segments_total) * amplitude)
            print(f'Constante Psigrad:{Psigrad}')
            
            # On boucle sur le quadrant
            for k in range(n_segments_qrt):
                if ImPart:
                    model.addConstr( (list_PsiD_AB_re[k] + list_PsiD_AB_im[k]).T @ Apo - Psigrad <= 0)
                    model.addConstr( (list_PsiD_AB_re[k] - list_PsiD_AB_im[k]).T @ Apo - Psigrad <= 0)
                    model.addConstr( (-list_PsiD_AB_re[k] + list_PsiD_AB_im[k]).T @ Apo - Psigrad <= 0)
                    model.addConstr( (-list_PsiD_AB_re[k] - list_PsiD_AB_im[k]).T @ Apo - Psigrad <= 0)
                else:
                    model.addConstr( list_PsiD_AB_re[k].T @ Apo - Psigrad <= 0)
                    model.addConstr(-list_PsiD_AB_re[k].T @ Apo - Psigrad <= 0)
            
    # Update model
    model.update()
    print(f'Gurobi model computation time: {time.perf_counter() - t_c:.2f}s')
    
    #%%
    """
    ### Solving of the model
    """
    print("\n" + "="*50)
    print("LAUNCHING GUROBI OPTIMIZER...")
    print('Solving problem with gurobipy package')
    print("="*50 + "\n")
    t_d = time.perf_counter()
    
    
    try:               
        model.Params.Method       = slvMethod
        model.Params.LogToConsole = slvLogToConsole
        model.Params.Crossover    = slvCrossover        
        model.optimize()
        
    except gb.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')

    print(f'Optimization time : {time.perf_counter() - t_d:.2f}s\n')
    
    
    #%% Display of the apodizer
    """
    ### Generation of full apodizer (Full Pupil)
    """
    print('Generation of the final apodizer')
    t_e = time.perf_counter()

    Apod_full_vec = np.zeros(((corono0.nPup//2)**2))
    Apod_full_vec[idx_pup] = Apo.x 
    
    Apod_full_2d = np.zeros((corono0.nPup, corono0.nPup))
    
    Apod_full_2dtmp =  np.reshape(Apod_full_vec, (corono0.nPup//2, corono0.nPup//2))
    Apod_full_2d[corono0.nPup//2:, corono0.nPup//2:] = Apod_full_2dtmp
    Apod_full_2d[:corono0.nPup//2, corono0.nPup//2:] = np.flip(Apod_full_2dtmp, axis=0)
    Apod_full_2d[:, :corono0.nPup//2]          = np.flip(Apod_full_2d[:, corono0.nPup//2:], axis=1)
            
    Apod_full_2d *= Pupil2d

    print(f'Apodizer generation time : {time.perf_counter() - t_e:.2f}s\n')
    
    #%%
    current_dim = Apod_full_2d.shape[0]
    nzp0 = (np.abs(Apod_full_2d) <= 1e-2)
    nzp1 = (np.abs(Apod_full_2d - 1.0) <= 1e-2)
    Gray_2d = np.ones((current_dim, current_dim))
    Gray_2d[nzp0] = 0.
    Gray_2d[nzp1] = 0.
    Ones_2d = np.zeros((current_dim, current_dim))
    Ones_2d[np.abs(Apod_full_2d) > 0.99] = 1.0

    #%%
    """
    Save apodizer
    """
    
    if not os.path.exists(fdir_sav):
        os.makedirs(fdir_sav)

    suffixe_array = f"_nPup{nPup0}_rMask{rMask:.3f}"

    fname_sav = get_filename(corono0) + '.fits'
    fpath_sav = fdir_sav / fname_sav
    
    if do_fits is True:
        fits.writeto(fpath_sav, Apod_full_2d, overwrite=True)
# Libération propre des ressources Gurobi
model.dispose()
gb.disposeDefaultEnv()

print("Session Gurobi fermée proprement. Jeton libéré.")
print(f"Temps de calcul total : {time.perf_counter() - t_start_global:.2f}s")