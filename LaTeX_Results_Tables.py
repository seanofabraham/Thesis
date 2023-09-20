#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:01:29 2023

@author: seanabrahamson
"""

import os
import pickle
import pandas as pd

def get_files(start_string, directory, file_type=None):
    """
    This function returns a list of file paths for all files in the specified directory that start with the given start_string and have the specified file_type.
    :param start_string: The string that the file names should start with
    :param directory: The directory to search for the files
    :param file_type: The file extension to filter by (e.g. '.txt'). If None, all file types are included.
    :return: A list of file paths
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(start_string):
                if file_type is None or file.endswith(file_type):
                    file_paths.append(os.path.join(root, file))
    return file_paths


def load_files_into_dict(start_string, directory, file_type=None):
    """
    This function loads all files that start with the given start_string and have the specified file_type into a dictionary where the key is the file name.
    :param start_string: The string that the file names should start with
    :param directory: The directory to search for the files
    :param file_type: The file extension to filter by (e.g. '.txt'). If None, all file types are included.
    :return: A dictionary where the key is the file name and the value is the contents of the file
    """
    files_dict = {}
    file_paths = get_files(start_string, directory, file_type)
    for path in file_paths:
        with open(path, 'rb') as handle:
            files_dict[os.path.basename(path)] = pickle.load(handle)
    return files_dict


"""
Script for organizing results into Latex Tables
"""
#%%
RESULTS = load_files_into_dict('Results','./Results',file_type = '.pkl')


#%% Loop through Results

Results_Dict = {}
Results_List = []

# Open a file to write the LaTeX code
with open('../LaTeXThesis/Latex-Thesis/ResultsTables.tex', 'w') as f:
    # Loop through the dataframes
    f.write(r"\documentclass{article}" )
    f.write('\n')
    f.write(r"\usepackage{graphicx}") 
    f.write('\n')
    f.write(r"\usepackage{booktabs}")
    f.write('\n\n')

    f.write(r'\begin{document}')
    f.write('\n\n')

    for value in RESULTS:
        # Convert the dataframe to a LaTeX table and write it to the file
        
        sectionTitle = value.replace('_', ' ')
        
        f.write('\section{' + sectionTitle + '} \n')
        
        f.write(RESULTS[value]['Coeff: K_1-K_5']['coefficientDF'].to_latex(index=False, float_format="%.3E"))
        f.write('\n')
        

    f.write(r'\end{document}')
        
