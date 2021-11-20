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
    return 'heatmap'


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



