# functions for eda

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import datetime
from datetime import date, timedelta
import math
import random
import re



# display correlation heatmap 
# inputs: df, width annd height of heatmap (int), correlation calc method (str), plot title (str)
def show_heatmap(df, w, h, method='pearson', title=''):
    corr_matrix=df.corr(method=method)
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True 

    fig, ax = plt.subplots(figsize=(w, h))
    ax = sns.heatmap(corr_matrix, mask=mask, annot=True)
    
    if title != '':
        ax.set_title('\n' + title + '\n', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.show()


# transform df with time series data to df with dates as indexes
# inputs: df, column with dates in datetime format, outputs: df
def df_to_timeseries(df, timeframe):
    df = df.sort_values(by=[timeframe], ignore_index=True)
    df = df.set_index(timeframe, drop=True)
    return df

# plot a time series df
# inputs: df, col with dates in datetime format, plot title (str)
def timeseries_plot(df, timeframe, title=''):
    # convert df to time series
    plot_df = df_to_timeseries(df=df, timeframe=timeframe)
    
    # create time series plot
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False}
    sns.set(rc={'figure.figsize':(20,16)})
    sns.set_theme(style="whitegrid", rc=custom_params)
    # option for title
    if title != '':
        sns.set_title(title)
    plot_df.plot()
    plt.xticks(rotation=45)
    plt.show()
    return df


# show correlactions of 2 columns above a certain treshold in a df
# inputs: df, method=correlation calc method (str, values that work with pd.corr), limit=corr threshold (float)
# outputs: df
def correlation_check(df, method='pearson', limit=0.8):
    # build corr matrix
    corr_matrix=df.corr(method=method)
    # build dict for results
    results = {'feature_1': [], 'feature_2': [], 'correlation': []}
    
    for col in range(corr_matrix.shape[1]-1):
        row_range = col+1
        for row in range(row_range):
            if (corr_matrix.iloc[row, col+1] >= limit) | (corr_matrix.iloc[row, col+1] <= (-1*limit)):
                # fill dict values with data for corr above 0.8
                results['feature_1'].append(corr_matrix.columns[col+1])
                results['feature_2'].append(corr_matrix.index[row])
                results['correlation'].append(round(corr_matrix.iloc[row, col+1], 3))
    
    # df with results from dict
    df_results = pd.DataFrame(results)
                
    return df_results


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


        

