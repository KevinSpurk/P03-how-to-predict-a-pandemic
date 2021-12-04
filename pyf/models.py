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
    
    return results


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
    
    return results


# create model based on a dataset dict as created with train_test_split_items function
# inputs: dataset=dict with train test split items, title for the model (str), model type=model to use, kwargs= additional arguments for model
# outputs: dict with train test split items from input dict and model data
def model_dataset(dataset, title, model_type, **kwargs):
    # fit model
    model = model_type(**kwargs)
    model.fit(dataset['X_train'], dataset['y_train'])
    
    # add title and model to dict
    dataset['title'] = title
    dataset['model'] = model
    return dataset

# get model evaluation metrics based on a dataset dict as created with model_dataset function
# inputs: dataset=dict with train test split items and model data
# outputs: dict with train test split items and model data from input dict and evaluation metrics, df with model metrics
def regression_model_evaluation_metrics(dataset):
    # model prediction
    predictions = dataset['model'].predict(dataset['X_test'])
    
    # model scores
    r2_train = round(dataset['model'].score(dataset['X_train'], dataset['y_train']), 3)
    r2_test = round(dataset['model'].score(dataset['X_test'], dataset['y_test']), 3)
    r2_adj = round(1-(1-r2_test)*((len(dataset['X_test'])-1)/(len(dataset['X_test'])-len(dataset['X_test'].columns)-1)), 3)
    mae = round(mean_absolute_error(dataset['y_test'], predictions), 3)
    rmse = round(mean_squared_error(dataset['y_test'], predictions, squared=False), 3)
    mse = round(mean_squared_error(dataset['y_test'], predictions, squared=True), 3)
    
    # collect metrics in df
    metrics = pd.DataFrame({'model title': [dataset['title']], 'R2 (train)': [r2_train], 'R2 (test)': [r2_test], 'R2 adj.': [r2_adj], 'MAE': [mae], 'RMSE': [rmse], 'MSE': [mse]})

    # add eval results to model dict
    evaluation = {'predictions': predictions, 'r2_train': r2_train, 'r2_test': r2_test,'r2_adj': r2_adj, 'MAE': mae, 'RMSE': rmse, 'MSE': mse}
    dataset.update(evaluation)
    
    return dataset, metrics

# create model and evaluation metrics based on a dataset dict as created with train_test_split_items function
# inputs: dataset=dict with train test split items, title for the model (str), model type=model to use, kwargs= additional arguments for model
# outputs: dict with train test split items from input dict, model data and evaluation metrics, df with model metrics
def regression_model_and_evaluation(dataset, title, model_type, **kwargs):
    # apply model to dataset
    model_dict = model_dataset(dataset=dataset, title=title, model_type=model_type, **kwargs)
    
    # get model evaluation metrics
    model_dict, model_metrics = regression_model_evaluation_metrics(dataset=model_dict)
    
    return model_dict, model_metrics


# aggregate model metrics into a single df
# inputs: metrics_sets=dict of dfs with metrics, 1 row per df and identical col.
# outputs: df
def compare_model_metrics(metrics_sets):
    # df for results
    results = pd.DataFrame({})
    
    # loop through metrics tables
    for table in metrics_sets.values():
        results = pd.concat([results, table], axis=0)
    
    results.reset_index(inplace=True, drop=True)
    results.style.hide_index()
    
    return results

# automates the creation of regression models for multiple datasets and the comparison of their model metrics
# using functions regression_model_and_evaluation and compare_model_metrics
# inputs: datasets=dict of model datasets as created with train_test_split_items function, titles=titles for the models (list), model type=model to use, kwargs= additional arguments for model
# outputs: input dict with model datasets and added items for model and evaluation metrics, df with model metrics comparison
def regression_model_comparison(datasets, titles, model_type, **kwargs):
    # dict to store all metrics tables
    metrics_allmodels = {}
    
    # iterator for list of titles
    i = 0
    
    # loop through model datasets in dict
    for data in datasets.values():
        # model title
        model_title = titles[i]
        # create model and evaloation
        lr_data, lr_metrics = regression_model_and_evaluation(dataset=data, title=model_title, model_type=model_type, **kwargs)
        # add evaluation to dict
        metrics_allmodels[model_title] = lr_metrics
        # go to next model title in list
        i += 1
    
    # combine all evaluations into single df
    results = compare_model_metrics(metrics_allmodels)
    
    return datasets, results


# get model evaluation metrics based on a dataset dict as created with model_dataset function
# metrics for a subset of the test data defined by a col to cluster by 
# inputs: dataset=dict with train test split items and model data, cluster_by=name cluster col (str), 
# undummify=option to use a prev get_dummies encoded col a cluster col (bool, default=False)
# outputs: df with evaluation metrics
from pyf.transform import invert_getdummies

def regression_model_evaluation_clustered(dataset, cluster_by, undummify=False):
    # new df to gather metrics for all clusters
    results = pd.DataFrame({})
                                                   
    X_test = dataset['X_test'].copy()
    
    # case: get original cluster col if prev encoded with pd.get_dummies
    if undummify == True:
        X_test = invert_getdummies(df=X_test, columns=[cluster_by], keep_dummies=True)                                         

    # rejoin x test and y test
    testset = pd.concat([X_test, dataset['y_test']], axis=1)
                        
    # loop through clusters
    for c in testset[cluster_by].unique():
        # create df for one cluster
        df_temp = testset[testset[cluster_by] == c]
        # x-y-split of test set
        X_test_cluster = df_temp.iloc[:, :-1]
        y_test_cluster = df_temp.drop(df_temp.iloc[:, :-1], axis=1)
        # case: remove original cluster col if prev encoded with pd.get_dummies
        if undummify == True:
            X_test_cluster = X_test_cluster.drop(cluster_by, axis=1)
        # create dataset dict for cluster for metrics function
        dataset_temp = dataset.copy()
        dataset_temp['X_test'] = X_test_cluster
        dataset_temp['y_test'] = y_test_cluster
        
        # get model evaluation metrics for cluster
        cluster_dict, cluster_metrics = regression_model_evaluation_metrics(dataset=dataset_temp)

        # add cluster name to metrics table
        cluster_metrics.insert(1, cluster_by, [c])

        # add cluster metrics to results df
        results = pd.concat([results, cluster_metrics], axis=0)
        results.reset_index(inplace=True, drop=True)
    
    return results



