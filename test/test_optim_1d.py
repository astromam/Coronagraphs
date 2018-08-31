#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:49:45 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

import corono as coro
import os

#%%
"""
Function to check basics in the class and subclass
"""
def check_basics(problem0):
    
    assert 'cDarkHole' in problem0

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
    problem0.compute_problem_matrices_gurobi()
    
    assert not 'compute_problem_matrices_1stder' in problem0
    problem0.compute_problem_matrices_1stDer()
    problem0.compute_problem_matrices_2ndDer()
        
#    problem0.compute_problem_matrices_Binarity()
#    assert 'npp_bis' in problem0
#    problem0.compute_problem_matrices_MinIsland()
    problem0.compute_gurobi_model()
    
    problem0.compute_response_matrices()
    problem0.solve_model()
    
    problem0.compute_matrices()
#    problem0.get_filename()
    

#%%
"""
Test the ProblemMatrix class
"""
def test_ProblemMatrix():
    
    problem0 = coro.optim_1d.ProblemMatrix()
    check_basics(problem0)
    
    problem1 = coro.optim_1d.ProblemMatrix(corono=coro.design.APLC1d())
    

#    assert 'corono_name' in problem0
#    problem0.get_filename()

#    problem0.compute_problem_matrices()
#
#    problem0.solve_model()

#%%
"""
Test the MaxTau subclass
"""
def test_MaxTau():

    problem0 = coro.optim_1d.MaxTau()
    check_basics(problem0)
    
    check_1d_problem_methods(problem0)
    
    params = coro.to_dict()
    params1 = coro.update_params(params, MinIsland=True)
    
    problem1 = coro.optim_1d.MaxTau(**params1)
    check_1d_problem_methods(problem1)    
    
    
#%%
"""
Test the MaxTau subclass
"""
#def test_MaxContrast():
#
#    problem0 = coro.optim_1d.MaxContrast()
#    check_basics(problem0)
#    
#    check_1d_problem_methods(problem0)

    