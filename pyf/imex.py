# functions handling operations with additional files  

import pandas as pd
import numpy as np
import datetime
import math
import random
import re
import glob


# import multiple csv files with same amount of columns into one df concatenated along axis 0. Gets all csv files from specified path
# innputs: file path (format <path>/), file name with desired col names (opt)
# output: df
def import_concat_csv(path, columns_from=''):
    # glob all files
    all_files = glob.glob(f'{path}' + '*.csv')

    # case: pick a file with col names
    if columns_from != '':
        # import csv with col names
        for f in all_files:
            if f == (path + columns_from + '.csv'):
                df_all = pd.read_csv(f)
                column_names = list(df_all.columns.values)
                
        # import all other csv 
        for fl in all_files:
            if fl != (path + columns_from + '.csv'):
                df = pd.read_csv(fl)
                df.columns = column_names
                df_all = pd.concat([df_all, df], axis=0, ignore_index=True)
                   
    # case: dont pick file with col names
    else:
        df_files = (pd.read_csv(ff) for ff in all_files)
        df_all = pd.concat(df_files, axis=0, ignore_index=True)
        
    df_all.reset_index(inplace=True, drop=True)

    return df_all


# function to export df to csv with var naming
# inputs: df, stings for constant and variable part of file name
# outputs: df
def df_to_csv_format(df, name_cst, name_var, index=False):
    file_name = name_cst + '_' + name_var
    df.to_csv(f'{file_name}.csv', sep=',', index=index)



