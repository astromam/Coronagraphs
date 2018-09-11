#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:44:47 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

import corono as coro
import os
import pytest

try:
    import stdgrb
except ModuleNotFoundError:
    stdgrb = False

try:
    import gurobipy as gb
except ModuleNotFoundError:
    gb = False

#%%
"""
Function to check basics in the class and subclass
"""
def check_basics(problem0):
    
    assert 'cDarkHole' in problem0

    print(problem0)
    print(problem0.params['cDarkHole'])    
    print(problem0.params)
    
    # load and save params
    problem0.save_params('temp.file')
    problem0.load_params('temp.file')
    os.remove('temp.file')
    
    assert 'problem_name' in problem0
    problem0.compute_response_matrices()
    
    problem0.print_log('test')

"""
Function to check the corono methods for 1d design
"""
def check_2d_problem_methods(problem0):
    
    problem0.compute_problem_matrices()
    
    assert not 'compute_problem_matrices_1stder' in problem0
    problem0.compute_problem_matrices_MinIsland()
    problem0.compute_gurobi_model()
    
    problem0.compute_response_matrices()
    problem0.solve_model()
    
    problem0.compute_matrices()
    problem0.get_filename()
    
    assert not 'Binarity' in problem0
    assert not 'nbb' in problem0

def check_2d_problem_methods_constraints(class_optim):
    
    problem00 = class_optim()
    problem00.compute_problem_matrices_gurobi()
        
    problem03 = class_optim()
    problem03.compute_problem_matrices_MinIsland()    

def check_2d_problem_methods_parameters(class_optim, Lnorm='L1'):

    params = coro.to_dict(Lnorm=Lnorm)

    params1 = coro.update_params(params, MinIsland=True)    
    problem1 = class_optim(**params1)
    check_2d_problem_methods(problem1)    

    params2 = coro.update_params(params, allLogToConsole=True)
    problem2 = class_optim(**params2)
    check_2d_problem_methods(problem2)
        
    params6 = coro.update_params(params, ImPart='False')
    problem6 = class_optim(**params6)
    check_2d_problem_methods(problem6)

    params7 = coro.update_params(params, LSRobustness='True')
    problem7 = class_optim(**params7, corono=[coro.design.APLC2d(), coro.design.APLC2d()])
    check_2d_problem_methods(problem7)

    params8 = coro.update_params(params, LSRobustness='True')
    problem8 = class_optim(**params8, corono=coro.design.APLC2d())
    check_2d_problem_methods(problem8)

    params = coro.to_dict() 
    params2 = coro.update_params(params, Pupil2dSym=False)    
    corono0 = coro.design.APLC2d(**params2)
    params9 = coro.update_params(params, LSRobustness='True')
    problem9 = class_optim(**params9, corono=corono0)
    check_2d_problem_methods(problem9)

def check_2d_problem_methods_parameters_stdgrb(class_optim, Lnorm='L1'):
    
    params = coro.to_dict(Lnorm=Lnorm)
    
    params1 = coro.update_params(params, solver='stdgrb')
    problem1 = class_optim(**params1)
    check_2d_problem_methods(problem1)        


def check_2d_problem_methods_parameters_gurobipy(class_optim, Lnorm='L1'):
    
    params = coro.to_dict(Lnorm=Lnorm)
    
    params1 = coro.update_params(params, solver='gurobipy')
    problem1 = class_optim(**params1)
    check_2d_problem_methods(problem1)
    
    
def check_2d_problem_methods_parameters_scipy(class_optim, Lnorm='L1'):
    
    params = coro.to_dict(Lnorm=Lnorm)
    
    params1 = coro.update_params(params, solver='xxx')
    problem1 = class_optim(**params1)
    check_2d_problem_methods(problem1)    

    
"""
Check the problem class
"""
def check_problem(class_optim):
    problem0 = class_optim()
    check_basics(problem0)
    check_2d_problem_methods(problem0)    

    
#%%
"""
Test the ProblemMatrix class
"""
def test_ProblemMatrix_2d():
    
    problem0 = coro.optim_2d.ProblemMatrix()
    check_basics(problem0)
    
#%%
"""
Test the MaxTau subclass
"""
def test_MaxTau_2d():
    check_problem(coro.optim_2d.MaxTau)

#%%
"""
Test the MaxTau subclass methods
"""
def test_MaxTau_2d_methods():
    problem0 = coro.optim_2d.MaxTau()
    problem0.update_cDarkHole()
    
    check_2d_problem_methods_constraints(coro.optim_2d.MaxTau)    

        
#%%
"""
Test the MaxContrast subclass
"""
def test_MaxContrast_2d():
    check_problem(coro.optim_2d.MaxContrast)
    check_2d_problem_methods_parameters(coro.optim_2d.MaxTau)
    
#%%
"""
Test the MaxContrast subclass methods
"""
def test_MaxContrast_2d_methods():
    problem0 = coro.optim_2d.MaxContrast()
    problem0.update_tau()
    
    check_2d_problem_methods_constraints(coro.optim_2d.MaxContrast)
    check_2d_problem_methods_parameters(coro.optim_2d.MaxContrast)
    check_2d_problem_methods_parameters(coro.optim_2d.MaxContrast, Lnorm='Linf')

#%%
"""
Test solvers
"""
@pytest.mark.skipif(not gb, reason="Missing gurobipy")
def test_optim_gurobipy():
    check_2d_problem_methods_parameters_gurobipy(coro.optim_2d.MaxTau)
    check_2d_problem_methods_parameters_gurobipy(coro.optim_2d.MaxContrast)
    check_2d_problem_methods_parameters_gurobipy(coro.optim_2d.MaxContrast, Lnorm='Linf')

@pytest.mark.skipif(not stdgrb, reason="Missing stdgrb")
def test_optim_stdgrb():
    check_2d_problem_methods_parameters_stdgrb(coro.optim_2d.MaxTau)
    check_2d_problem_methods_parameters_stdgrb(coro.optim_2d.MaxContrast)    
    check_2d_problem_methods_parameters_stdgrb(coro.optim_2d.MaxContrast, Lnorm='Linf')
    
def test_optim_scipy():
    check_2d_problem_methods_parameters_scipy(coro.optim_2d.MaxTau)
    check_2d_problem_methods_parameters_scipy(coro.optim_2d.MaxContrast)    
    check_2d_problem_methods_parameters_stdgrb(coro.optim_2d.MaxContrast, Lnorm='Linf')
        