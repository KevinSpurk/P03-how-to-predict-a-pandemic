# data transformation functions incl. scaling, box cox, encoding

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import datetime
import math
import random
import re


# encoding categorical col with get dummies
# inputs: df, in_columns=col to encode (list, default=all object col), skip_columns=col not to encode (list, default=empty), 
# drop_first= value for pd.get_dummies
# outputs: df
def encoding_get_dummies(df, in_columns=[], skip_columns=[], drop_first=True):
    # set col to all object col by default
    if in_columns == []:
        in_columns = list(df.select_dtypes(np.object).columns)
        # list col not to include in encoding
        for col in skip_columns:
            in_columns.remove(col)
            
    # split df into df to encode and df with other col    
    df_encode = df[in_columns]
    df_rest = df.drop(in_columns, axis=1)
    
    # encode
    df_encode = pd.get_dummies(df_encode, drop_first=drop_first)
    
    # concat encoded and other col
    df = pd.concat([df_encode, df_rest], axis=1)
    df.reset_index(inplace=True, drop=True)
    
    return df





