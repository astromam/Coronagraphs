# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:37:22 2024

@author: asp
"""

import os

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
