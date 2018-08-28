
import numpy as np
import corono.design as ds
import os



import gurobipy


def test_get_default_params():
    
    ds.get_default_params_Coronagraph()
    
    ds.get_default_params_APLC1d()
    
    ds.get_default_params_SP1d()
    
    ds.get_default_params_DZPM1d()
    
    ds.get_default_params_HDZPM1d()

    ds.get_default_params_APLC2d()
    
    ds.get_default_params_SP2d()
    
    ds.get_default_params_DZPM2d()
    


def test_Coronagraph():
    
    cor=ds.Coronagraph()
    
    # load and save params
    cor.save_params('temp.file')
    cor.load_params('temp.file')
    os.remove('temp.file')
    
    # print and filename
    print(cor)
    print(cor.get_filename())
    
    # test params
    assert 'nFPM' in cor
    assert not 'mamadoupowa' in cor
    
    # test values
    np.testing.assert_allclose(cor.ClearPupil1d,1)
    
    # test function call
    
    #f=cor.compute_nostop_field_1d()
    #assert f.shape[1]==cor.nPup and f.shape[0]==cor.nlam
    
    