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

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, LabelEncoder


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


# decode a pd.get_dummies encoded df
# works only if get_dummies was done with 'drop_first=False' and if col names of dummified col did not appear as part of other col names
# inputs: df, columns= col to decode (list), keep_dummies=option to keep or drop encoded col for output df (bool, default=False)
# outputs: decoded df
def invert_getdummies(df, columns, keep_dummies=False):
    # new df for all undummified columns
    df_undummify = df.copy()
    
    # loop though col to undummify
    for col in columns:
        # list get dummies columns
        undummify = [dummy_col for dummy_col in df.columns if dummy_col.startswith(col) == True]
        
        # create df with dummy col only
        df_temp = df[undummify]
        
        # get df with original values from col names
        original_val = [c.split('_', maxsplit=1)[1] for c in df_temp.columns]
        df_temp.columns = original_val
        results = pd.DataFrame({col: df_temp.idxmax(axis='columns')})
        
        # add to return df
        df_undummify = pd.concat([df_undummify, results], axis=1)
        
        # case: drop get dummies col 
        if keep_dummies==False:
            df_undummify = df_undummify.drop(undummify, axis=1)
            
    return df_undummify



from pyf.preprocessing import _df_split_columns_num

# function to choose a feature transformation like StandardScaler, MinMaxScaler, Normalizer with the option to apply an existing fit 
# take inputs: df, transformer_type=normaization function, load_transformer=transformer of a previous transformation, 
# in_columns=col to transform (list, set to all num col inside func by default), skip=col not to transform if in_columns is default (list)
# outputs: df with transformations, transformer
def select_feature_scaling(df, in_columns=[], skip=[], transformer_type='', load_transformer=''):    
    # select col for transformation
    df_in, df_other = _df_split_columns_num(df=df, in_columns=in_columns, skip=skip)

    # case: new transformation fit
    if load_transformer == '':
        transformer = transformer_type().fit(df_in)
    # case: use existing fit
    else:
        transformer = load_transformer
    
    # transformation
    transformed = transformer.transform(df_in)
    df_transformed = pd.DataFrame(transformed, columns=list(df_in.columns))
    
    # formating transformed data
    df_transformed.index = df_in.index
    df_transformed.columns = df_in.columns
    try:
        df_transformed = pd.concat([df_transformed, df_other], axis=1)
    except:
        pass
     
    return df_transformed, transformer


# function to choose a feature transformation like StandardScaler, MinMaxScaler, Normalizer and apply to a model dataset dict 
# take inputs: dataset, transformer_type=normaization function, in_columns=col to transform (list, set to all num col inside func by default), 
# skip=col not to transform if in_columns is default (list), target=choose btw. scaling features or target var (bool, default=False), title=pick name for dict key to add scaler (str)
# outputs: df with transformations
def scaling_model_dataset(dataset, title, in_columns=[], skip=[], transformer_type='', target=False):
    # create dataset copy for results
    dataset_scaled = dataset.copy()
    
    # case: scaling features
    if target == False:
        # scaling training set
        dataset_scaled['X_train'], _trfm = select_feature_scaling(df=dataset_scaled['X_train'], in_columns=in_columns, skip=skip, transformer_type=transformer_type)
        # scaling test set
        dataset_scaled['X_test'], _trfm = select_feature_scaling(df=dataset_scaled['X_test'], in_columns=in_columns, skip=skip, load_transformer=_trfm)
    
    # case: scaling target var
    else:
        # scaling training set
        dataset_scaled['y_train'], _trfm = select_feature_scaling(df=dataset_scaled['y_train'], in_columns=in_columns, skip=skip, transformer_type=transformer_type)
        # scaling test set
        dataset_scaled['y_test'], _trfm = select_feature_scaling(df=dataset_scaled['y_test'], in_columns=in_columns, skip=skip, load_transformer=_trfm)

    # add sckler to dict
    dataset_scaled[title] = _trfm
        
    return dataset_scaled
        

# function to invert feature scaling transformation like StandardScaler, MinMaxScaler, Normalizer
# take inputs: df, load_transformer=transformer to invert, in_columns=col to transform (list, set to all num col inside func by default), skip=col not to transform if in_columns is default (list)
# outputs: df with original unscaled data
def invert_feature_scaling(df, load_transformer, in_columns=[], skip=[]):
     # select col for transformation
    df_in, df_other = _df_split_columns_num(df=df, in_columns=in_columns, skip=skip)
    
    # invert sclaing
    inverted = load_transformer.inverse_transform(df_in)
    
     # creating df from inverted data
    df_inverted = pd.DataFrame(inverted, columns=list(df_in.columns))
    df_inverted.index = df_in.index
    try:
        df_inverted = pd.concat([df_inverted, df_other], axis=1)
    except:
        pass
     
    return df_inverted    


# function to invert feature scaling transformation like StandardScaler, MinMaxScaler, Normalizer
# take inputs: dataset, load_transformer=transformer to invert, in_columns=col to transform (list, set to all num col inside func by default), 
# skip=col not to transform if in_columns is default (list), target=choose btw. unscaling features or target var (bool, default=False), trainingset=choose to unscaling trainingset too (bool, default=False)
# outputs: dataset with original unscaled data    
def invert_scaling_model_dataset(dataset,  load_transformer, in_columns=[], skip=[], target=False, trainingset=False):
    # create dataset copy for results
    dataset_inverted = dataset.copy()
    # case: invert feature scaling
    if target == False:
        # invert scaling of test data
        dataset_inverted['X_test'] = invert_feature_scaling(df=dataset_inverted['X_test'], in_columns=in_columns, skip=skip, load_transformer=load_transformer)
        # invert scaling of training data
        if trainingset == True:
            dataset_inverted['X_train'] = invert_feature_scaling(df=dataset_inverted['X_train'], in_columns=in_columns, skip=skip, load_transformer=load_transformer)
    
    # case: invert target var scaling
    else:
        # invert scaling of test data
        dataset_inverted['y_test'] = invert_feature_scaling(df=dataset_inverted['y_test'], in_columns=in_columns, skip=skip, load_transformer=load_transformer)
        # invert scaling of training data
        if trainingset == True:
            dataset_inverted['y_train'] = invert_feature_scaling(df=dataset_inverted['y_train'], in_columns=in_columns, skip=skip, load_transformer=load_transformer)
    
    return dataset_inverted


# function to display distribution plots and boxplots of df columns side by side
# inputs: df, columns=col to plot (list, optional, default=all num col), skip=col not to plot(list, optional, default=[])
# outputs: none
from pyf.preprocessing import _df_split_columns_num

def plots_continuous_var(df, in_columns=[], skip=[]):
    df_plot, df_rest = _df_split_columns_num(df=df, in_columns=in_columns, skip=skip)
    for col in df_plot.columns:
        print('\n')
        custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False}
        sns.set_theme(style="whitegrid", rc=custom_params)
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        sns.distplot(df_plot[col], ax=axes[0])
        sns.boxplot(df_plot[col], ax=axes[1])
        plt.show()


