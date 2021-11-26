# functions for modeling and evaluation operations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import datetime
from datetime import date, timedelta
import math
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



# create a x-y split and train-test split based on a df with target var in the last col. 
# inputs: df, timeframe=name of col with timeseries data (str, opt. in case df contains time col), test_size=fraction of data to use in test set (float, default=0.2), 
# random_state=used in train_test_split (int, default=84), as_timeseries=choice to separate timeseries into 2 parts at a certain date instead of random train test split(bool, default=False)
# outputs: original df, X_train, X_test, y_train, y_test
def train_test_split_items(df, test_size=0.2, random_state=84, timeframe='', as_timeseries=False):
    # clean index
    df.reset_index(inplace=True, drop=True)
    
    # case: random sample test set
    if as_timeseries == False:
        results = _train_test_split_items_random(df=df, test_size=test_size, random_state=random_state, timeframe=timeframe)
            
    # case: latest data of timeseries as test set
    else:
        results = _train_test_split_items_chronological(df=df, test_size=test_size, random_state=random_state, timeframe=timeframe)

    return results

# sub func of train_test_split_items
def _train_test_split_items_random(df, test_size, random_state, timeframe):
    # x-y-split
    X = df.iloc[:, :-1]
    y = df.drop(df.iloc[:, :-1], axis=1)
    
    # train test split    
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=test_size, random_state=random_state)   
    
    # move col with timeseries data to separate df
    if timeframe != '':
        time_trainset = X_trainset[[timeframe]]
        X_trainset = X_trainset.drop([timeframe], axis=1)
        time_testset = X_testset[[timeframe]]
        X_testset = X_testset.drop([timeframe], axis=1)
    else:
        time_trainset, time_testset = [], []

    # results dict
    results = {'X_train': X_trainset, 'X_test': X_testset, 'y_train': y_trainset, 'y_test': y_testset, 'time_train': time_trainset, 'time_test': time_testset}

# sub func of train_test_split_items
def _train_test_split_items_chronological(df, test_size, random_state, timeframe):
    df = df.sort_values(by=[timeframe], ignore_index=True)
    
    # find cut off date to separate train and test set
    cut_index = round(len(df)*(1-test_size))
    cut_date = df[timeframe].iloc[cut_index]
    
    # create dfs for train and test set
    trainset = df[df[timeframe] < cut_date]
    testset = df[df[timeframe] >= cut_date]
    
    # create x y split
    X_trainset = trainset.iloc[:, :-1].drop([timeframe], axis=1)
    X_testset = testset.iloc[:, :-1].drop([timeframe], axis=1)
    y_trainset = trainset.drop(trainset.iloc[:, :-1], axis=1)
    y_testset = testset.drop(testset.iloc[:, :-1], axis=1)
    
    # save timeseries col
    time_trainset = trainset[[timeframe]]
    time_testset = testset[[timeframe]]
        
    # results dict
    results = {'X_train': X_trainset, 'X_test': X_testset, 'y_train': y_trainset, 'y_test': y_testset, 'time_train': time_trainset, 'time_test': time_testset}








