#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:49:45 2018

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
def check_1d_problem_methods(problem0):
    
    problem0.compute_problem_matrices()
    
    assert not 'compute_problem_matrices_1stder' in problem0
    problem0.compute_problem_matrices_1stDer()
    problem0.compute_problem_matrices_2ndDer()
    problem0.compute_problem_matrices_MinIsland()
    problem0.compute_gurobi_model()
    
    problem0.compute_response_matrices()
    problem0.solve_model()
    
    problem0.compute_matrices()
    problem0.get_filename()
    
    assert not 'Binarity' in problem0
    assert not 'nbb' in problem0
 
def check_1d_problem_methods_constraints(class_optim):
    
    problem00 = class_optim()
    problem00.compute_problem_matrices_gurobi()
    
    problem01 = class_optim()
    problem01.compute_problem_matrices_1stDer()
    
    problem02 = class_optim()
    problem02.compute_problem_matrices_2ndDer()
    
    problem03 = class_optim()
    problem03.compute_problem_matrices_MinIsland()    

def check_1d_problem_methods_parameters(class_optim, Lnorm='L1'):

    params = coro.to_dict(Lnorm=Lnorm)

    params1 = coro.update_params(params, MinIsland=True)    
    problem1 = class_optim(**params1)
    check_1d_problem_methods(problem1)    

    params2 = coro.update_params(params, allLogToConsole=True)
    problem2 = class_optim(**params2)
    check_1d_problem_methods(problem2)
    
    params3 = coro.update_params(params, FirstDer=True)
    problem3 = class_optim(**params3)
    check_1d_problem_methods(problem3)
    
    params4 = coro.update_params(params, SecondDer=True)
    problem4 = class_optim(**params4)
    check_1d_problem_methods(problem4)
    
def check_1d_problem_methods_parameters_gurobipy(class_optim, Lnorm='L1'):
    
    params = coro.to_dict(Lnorm=Lnorm)
    
    params5 = coro.update_params(params, solver='gurobipy')
    problem5 = class_optim(**params5)
    check_1d_problem_methods(problem5)
    
    
def check_1d_problem_methods_parameters_scipy(class_optim, Lnorm='L1'):
    
    params = coro.to_dict(Lnorm=Lnorm)
    
    params6 = coro.update_params(params, solver='xxx')
    problem6 = class_optim(**params6)
    check_1d_problem_methods(problem6)    
    
def check_1d_problem_methods_parameters_stdgrb(class_optim, Lnorm='L1'):
    
    params = coro.to_dict(Lnorm=Lnorm)
    
    params6 = coro.update_params(params, solver='stdgrb')
    problem6 = class_optim(**params6)
    check_1d_problem_methods(problem6)        

"""
Check the problem class
"""
def check_problem(class_optim):
    problem0 = class_optim()
    check_basics(problem0)
    check_1d_problem_methods(problem0)    

#%%
"""
Test the ProblemMatrix class
"""
def test_ProblemMatrix_1d():
    
    problem0 = coro.optim_1d.ProblemMatrix()
    check_basics(problem0)
        
#%%
"""
Test the MaxTau subclass
"""
def test_MaxTau_1d():
    check_problem(coro.optim_1d.MaxTau)
        
#%%
"""
Test the MaxContrast subclass
"""
def test_MaxContrast_1d():
    check_problem(coro.optim_1d.MaxContrast)
        
#%%
"""
Test the MaxTau subclass methods
"""
def test_MaxTau_1d_methods():
    check_1d_problem_methods_constraints(coro.optim_1d.MaxTau)    
    check_1d_problem_methods_parameters(coro.optim_1d.MaxTau)
        
#%%
"""
Test the MaxContrast subclass methods
"""
def test_MaxContrast_1d_methods():
    check_1d_problem_methods_constraints(coro.optim_1d.MaxContrast)
    check_1d_problem_methods_parameters(coro.optim_1d.MaxContrast)
    check_1d_problem_methods_parameters(coro.optim_1d.MaxContrast, Lnorm='Linf')


"""
Test solvers
"""
@pytest.mark.skipif(not gb, reason="Missing gurobipy")
def test_optim_gurobipy():
    check_1d_problem_methods_parameters_gurobipy(coro.optim_1d.MaxTau)
    check_1d_problem_methods_parameters_gurobipy(coro.optim_1d.MaxContrast)
    
@pytest.mark.skipif(not stdgrb, reason="Missing stdgrb")
def test_optim_stdgrb():
    check_1d_problem_methods_parameters_stdgrb(coro.optim_1d.MaxTau)
    check_1d_problem_methods_parameters_stdgrb(coro.optim_1d.MaxContrast)    
    
def test_optim_scipy():
    check_1d_problem_methods_parameters_scipy(coro.optim_1d.MaxTau)
    check_1d_problem_methods_parameters_scipy(coro.optim_1d.MaxContrast)    
    