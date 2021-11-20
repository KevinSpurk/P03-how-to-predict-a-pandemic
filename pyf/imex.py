# functions handling operations with additional files  

import pandas as pd
import numpy as np
import datetime
import math
import random
import re


# function to export df to csv with var naming
# inputs: df, stings for constant and variable part of file name
# outputs: df
def df_to_csv_format(df, name_cst, name_var, index=False):
    file_name = name_cst + '_' + name_var
    df.to_csv(f'{file_name}.csv', sep=',', index=index)
    return df


