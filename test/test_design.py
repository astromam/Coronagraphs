#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

#%% Initialization
import numpy as np
import corono as coro
import os

#%%
"""
Function to check basics in the class and subclass
"""
def check_basics(corono0):
    # load and save params
    corono0.save_params('temp.file')
    corono0.load_params('temp.file')
    os.remove('temp.file')
    
    # print and filename
    print(corono0)
    print(corono0.get_filename())

    # print and filename
    varname = 'test'
    print(corono0.get_cache(varname))
    
    # test params
    assert 'nFPM' in corono0
    assert not 'mamadoupowa' in corono0  
#%%
"""
Function to check the corono methods for 1d design
"""
def check_1d_corono_methods(corono0):
    Apod = corono0.Pupil1d
    corono0.compute_direct_field_1d(Apod)
    corono0.compute_corono_field_1d(Apod)
    
    corono0.compute_direct_intensity_1d(Apod, poly=True)
    corono0.compute_direct_intensity_1d(Apod, poly=False)
    corono0.compute_corono_intensity_1d(Apod, poly=True)
    corono0.compute_corono_intensity_1d(Apod, poly=False)   

"""
Function to check the corono methods for 2d design
"""    
def check_2d_corono_methods(corono0):
    Apod = corono0.Pupil2d
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)
    
    corono0.compute_direct_intensity_2d(Apod, poly=True)
    corono0.compute_direct_intensity_2d(Apod, poly=False)
    corono0.compute_corono_intensity_2d(Apod, poly=True)
    corono0.compute_corono_intensity_2d(Apod, poly=False)
    
    corono0.compute_direct_field_2d_vec(Apod)
    corono0.compute_corono_field_2d_vec(Apod)
   
    params = coro.to_dict() 
    params2 = coro.update_params(params, Pupil2dSym=True)
    
    if corono0.corono_name == 'SP':
        corono0 = coro.design.SP2d(**params2)
    elif corono0.corono_name == 'APLC':
        corono0 = coro.design.APLC2d(**params2)
    elif corono0.corono_name == 'DZPM':
        corono0=coro.design.DZPM2d(**params2)        
    else:
        raise NameError('{0}: Not an existing coronagraph!'.format(corono0.corono_name))
        
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod) 

"""
Function to check the 1d corono class
"""
def check_corono_1d(class_corono):
    # define corono    
    corono0=class_corono()
    
    # test basics
    check_basics(corono0) 

    # test corono methods
    check_1d_corono_methods(corono0)

"""
Function to check the 2d corono class
"""
def check_corono_2d(class_corono):
    # define corono    
    corono0=class_corono()
    
    # test basics
    check_basics(corono0) 

    # test corono methods
    check_2d_corono_methods(corono0)     

    
#%%
"""
Test the Coronagraph class
"""
def test_Coronagraph():

    # define corono    
    corono0=coro.design.Coronagraph()

    # test basics
    check_basics(corono0)    
    
    # test values
    np.testing.assert_allclose(corono0.ClearPupil1d,1)
    
    # test function call
    assert 'Pupil1d' in corono0 
    corono0.compute_nostop_field_1d()
    
    corono0.compute_nostop_intensity_1d(poly=True)
    corono0.compute_nostop_intensity_1d(poly=False)
    
    
    corono0.generate_area()
    corono0.params['Pupil2dSym'] = True
    corono0.generate_area()
    
        
#%%    
"""
Test the APLC1d subclass
"""
def test_APLC1d():
    check_corono_1d(coro.design.APLC1d)

#%%    
"""
Test the SP1d subclass
"""
def test_SP1d():
    check_corono_1d(coro.design.SP1d)
    
#%%
"""
Test the DZPM1d subclass
"""
def test_DZPM1d():
    check_corono_1d(coro.design.DZPM1d)
    
#%%
"""
Test the HDZPM1d subclass
"""
def test_HDZPM1d():
    check_corono_1d(coro.design.HDZPM1d)    

#%%
"""
Test the HTZPM1d subclass
"""
def test_HTZPM1d():
    check_corono_1d(coro.design.HTZPM1d)        

#%%
"""
Test the APLC2d subclass
"""
def test_APLC2d():
    
    # define corono
    corono0=coro.design.APLC2d()
    
    # test basics
    check_basics(corono0)
    
    # test methods
    check_2d_corono_methods(corono0)

    Apod = corono0.Pupil2d    
    params = coro.to_dict() 

    params = coro.to_dict() 
    params2 = coro.update_params(params, Pupil2dSym=True)
    corono0 = coro.design.APLC2d(**params2) 
    corono0.compute_direct_lyot_field_2d(Apod)
    corono0.compute_corono_lyot_field_2d(Apod)
       
    OPDmap2d = corono0.Pupil2d
    params3 = coro.update_params(params, OPDmap2d=OPDmap2d)

    corono0 = coro.design.APLC2d(**params3)    
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)
    corono0.compute_direct_lyot_field_2d(Apod)
    corono0.compute_corono_lyot_field_2d(Apod)

    Ampmap2d = corono0.Pupil2d    
    params4 = coro.update_params(params, Ampmap2d=Ampmap2d)
    
    corono0 = coro.design.APLC2d(**params4)    
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)
    corono0.compute_direct_lyot_field_2d(Apod)
    corono0.compute_corono_lyot_field_2d(Apod)

    
    
#%%
"""
Test the SP2d subclass
"""
def test_SP2d():
    check_corono_2d(coro.design.SP2d)    

#%%
"""
Test the DZPM2d subclass
"""
def test_DZPM2d():
    check_corono_2d(coro.design.DZPM2d)    
