# data pre-processing functions incl. cleaning, feature engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import datetime
from datetime import timedelta
import math
import random
import re

from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import TomekLinks


# function to standardize column names
# inputs: df
# outputs: df
def clean_headers(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    return df


# function to replace strings in a df
# inputs: df, dict with column names as keys and list(2) as values, value 0 - regex replace pattern, value 1 - replacement str
# outputs: df
def clean_data(df, cleaner):
    for column in cleaner:
        print('Cleaning', column)
        for x in range(len(df[column])):
            df[column].iloc[x] = re.sub(cleaner[column][0], cleaner[column][1], df[column].iloc[x])
    print('Done')
    return df


# function to create a df with an overview of the amount of NaNs, empty fields, zeros and negative values in all columns of a df
# option to only show columns with amount of unwelcome values above a chosen threshold
# inputs: df, threshold percentage (float), bool. ind. to exclude negative numbers from the threshold
# outputs: overview df, list of columns shown
def value_overview(df, limit=0, neg_allowed=False):
    # creating columns for data type, no of unique values, NaNs
    df_dtypes = pd.DataFrame(df.dtypes, columns=['type'])
    no_uniques = pd.DataFrame(df.nunique(), columns=['unique values'])
    nulls = pd.DataFrame(df.isna().sum(), columns=['NaN'])
    nulls_percent = pd.DataFrame(round(df.isna().sum()*100/len(df), 2), columns=['NaN %'])

    # lists for of no of zeros, empty cells, neg. values 
    zeros = [len(df[df[column] == 0]) if df_dtypes['type'].loc[column] != 'object' else 0 for column in df.columns]
    negative_values = [len(df[df[column] < 0]) if df_dtypes['type'].loc[column] != 'object' else 0 for column in df.columns]

    emptys_1 = [len(df[df[column] == '']) for column in df.columns]
    emptys_2 = [len(df[df[column] == ' ']) for column in df.columns]
    emptys = [(emptys_1[i] + emptys_2[i]) for i in range(len(emptys_1))]

    zeros_percent, negative_percent, emptys_percent = [], [], []

    zeros_percent = [round(zeros[j]*100/len(df), 2) for j in range(len(zeros))]
    negative_percent = [round(negative_values[j]*100/len(df), 2) for j in range(len(negative_values))]
    emptys_percent = [round(emptys[j]*100/len(df), 2) for j in range(len(emptys))]

    # df for of zeros, empty cells, neg. values 
    special_values = pd.DataFrame({'empty': emptys, 'empty %': emptys_percent, 'zeros': zeros, 'zeros %': zeros_percent, 'negative': negative_values, 'negative %': negative_percent})
    special_values.index = df_dtypes.index
    
    # concat
    overview = pd.concat([df_dtypes, no_uniques, nulls, nulls_percent, special_values], axis=1)

    # option to filter overview with threshold value
    if limit != 0:
        overview['removable %'] = overview['NaN %']
    
        for x in overview['removable %']:
            if neg_allowed == False:
                overview['removable %'] = overview['NaN %'] + overview['empty %'] + overview['negative %']
            else:
                overview['removable %'] = overview['NaN %'] + overview['empty %']
    
        overview = overview[overview['removable %'] > limit].sort_values(by='removable %', ascending = False)
        print(len(overview), 'columns are missing at least', limit, '% of the data')
    
    return overview, list(overview.index)


# given a column to group entries of a df, ckeck if all entries are provided for the same range of dates
# inputs: df, col to group by, col with dates
# output: print statement if timeframe is complete, df with details about timeframe
def timeframe_check_by_group(df, groupby, timeframe):
    # lists for results df
    group_item, dt_min, dt_max, date_count, group_len = [], [], [], [], []
    # loop throught groups 
    for x in df[groupby].unique():
        # fill lists with info about dates present
        group_item.append(x)
        dt_min.append(min(df[df[groupby] == x][timeframe]))
        dt_max.append(max(df[df[groupby] == x][timeframe]))
        date_count.append(df[df[groupby] == x][timeframe].nunique())
        group_len.append(len(df[df[groupby] == x]))
    
    # check if range of dates is identical and complete for all groups
    if (min(dt_min) == max(dt_min)) & (min(dt_max) == max(dt_max)) & (min(date_count) == max(date_count)) & (min(group_len) == max(group_len)) & (min(date_count) == min(group_len)):
        print('timeframe:', datetime.datetime.strftime(min(dt_min), "%Y-%m-%d"), '-', datetime.datetime.strftime(min(dt_max), "%Y-%m-%d"), '\n', 'no. of dates:', min(date_count), '\n', 'timeframe complete!')
    else:
        print('timeframe incomplete. Check details.')
    
    # create df for result details
    results = pd.DataFrame({groupby: group_item, 'min_date': dt_min, 'max_date': dt_max, 'no_of_dates': date_count, 
                            'expected_no_of_dates': (max(df[df[groupby] == x][timeframe]) - min(df[df[groupby] == x][timeframe])).days + 1})
    
    return results


# limit df with date column to entries within a select timeframe
# inputs: df, column of dates (datetime format), start and end date for chosen periode (str) with YYYY-MM-DD format
# outputs: df
def df_timeframe_limit(df, col, start_date, end_date):
    # convert inputs to datetime
    dt_a = str_to_date(start_date)
    dt_b = str_to_date(end_date)

    # drop rows with dates before and after chosen timeframe
    df = df.drop(df[df[col] < dt_a].index)
    df = df.drop(df[df[col] > dt_b].index)
    
    return df
    

# function to deal with class imbalances in binary classification
# inputs: dfs for training set x, y, sampling method identification string, targeted ratio for maj./min. class for up-/down-/mix-sampling
# outputs: balanced training set x, y 
def class_balancing(imb_X, imb_Y, method='down', ratio=1.5): 
    # values to navigate target variable
    target_col = list(imb_Y.columns.values)[0]
    values_sorted = imb_Y[target_col].value_counts()
    train_imbalanced = pd.concat([imb_X, imb_Y], axis=1)    
    # SMOTE
    if method == 'smo':
        smote = SMOTE()
        bal_X, bal_Y = smote.fit_resample(imb_X, imb_Y) 
    #TomekLinks
    elif method == 'tl':
        undersample = TomekLinks()
        bal_X, bal_Y = undersample.fit_resample(imb_X, imb_Y)
    else:    
        # Downsampling
        if method == 'down':
            sample_size = values_sorted[1]
            maj_class = train_imbalanced[train_imbalanced[target_col] == values_sorted.index[0]].sample(round(sample_size*ratio))
            min_class = train_imbalanced[train_imbalanced[target_col] == values_sorted.index[1]].sample(sample_size)
        # Upsampling    
        elif method == 'up':
            sample_size = values_sorted[0]
            maj_class = train_imbalanced[train_imbalanced[target_col] == values_sorted.index[0]].sample(sample_size)
            min_class = train_imbalanced[train_imbalanced[target_col] == values_sorted.index[1]].sample(round(sample_size/ratio), replace=True)
        # Mix Up-/Downsampling
        elif method == 'mix':
            sample_size = values_sorted[1]
            imbalance_factor = math.floor(values_sorted[0]/values_sorted[1])
            sample_factor = math.floor(imbalance_factor/3)
            maj_class = train_imbalanced[train_imbalanced[target_col] == values_sorted.index[0]].sample(round(sample_size*sample_factor*ratio))
            min_class = train_imbalanced[train_imbalanced[target_col] == values_sorted.index[1]].sample(round(sample_size*sample_factor), replace=True)
          
        train_sampled = pd.concat([maj_class, min_class]).sample(frac=1)
        # creating dfs of balanced classes to return
        bal_X = train_sampled.drop([target_col], axis=1)
        bal_Y = pd.DataFrame(train_sampled[target_col])
    return bal_X, bal_Y


# converts string into date format
# inputs: date string of most common formats, outputs: date object of date
def str_to_date(date_str):
    try:
        date_object = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    except:
        try:
            date_object = datetime.datetime.strptime(date_str, '%y-%m-%d')
        except:
            try:
                date_object = datetime.datetime.strptime(date_str, '%m/%d/%Y')
            except:
                try:
                    date_object = datetime.datetime.strptime(date_str, '%m/%d/%y')
                except:
                    try:
                        date_object = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except:
                        try:
                            date_object = datetime.datetime.strptime(date_str, '%m/%d/%y %H:%M:%S')
                        except:
                            try:
                                date_object = datetime.datetime.strptime(date_str, '%Y %m %d')
                            except:
                                date_object = datetime.datetime.strptime('0000-00-00', '%Y-%m-%d')
                                print('date format unknown.')
                                   
    date_object = date_object.date()
    return date_object


# list the missing dates in a time series df
# inputs: df, col with dates in datetime format, astart and end date of period to check (str)
# outputs: missing dates (list)
def timeseries_missing_dates(df, timeframe, start_date, end_date):
    # set start and end date for loop
    start_date = str_to_date(start_date)
    end_date = str_to_date(end_date)
    
    # items for missing days
    missing = []
    
    # loop to find missing dates in df
    current = start_date
    while current <= end_date:
        date_check = False
        for d in df[timeframe]:
            if d == current:
                date_check = True
        # append missing date to lists
        if date_check == False:
            missing.append(current)
        current += timedelta(days=1)
    return missing
    
# check a time series df for missing dates and add rows for them 
# inputs: df, col with dates in datetime format, astart and end date of period to check (str), col with constant values to adopt for rows with misssing dates (list)
# outputs: df with NaNs for missing values
def complete_timeseries(df, timeframe, start_date, end_date, constant_col=[]):
    # list missing dates of time series
    missing_dates = timeseries_missing_dates(df=df, timeframe=timeframe, start_date=start_date, end_date=end_date)
    # dict for missing dates
    missing_data = {timeframe: missing_dates}
    
    # create lists with data for dict
    for col in df.columns:
        if col != timeframe:
            # list to fill columns with known constant values
            if col in constant_col:
                missing_values = [df[col].iloc[0] for m in missing_dates]
            # list to fill columns with NaN
            else:
                missing_values = [np.NaN for m in missing_dates]
            # fill dict with data for column
            missing_data[col] = missing_values
    
    # put missing data dict into df
    columns_ordered = df.columns.tolist()
    df_dates_missing = pd.DataFrame(missing_data)
    df_dates_missing = df_dates_missing[columns_ordered]
    
    # concat existing df with missing dates df 
    df_completed = pd.concat([df, df_dates_missing], axis=0)
    df_completed[timeframe] = pd.to_datetime(df_completed[timeframe], errors='coerce')
    df_completed[timeframe] = df_completed[timeframe].dt.date
    df_completed = df_completed.sort_values(by=[timeframe], ignore_index=True)
    
    return df_completed   


# interpolation in a timeseries df where dates appear multiple times, but are unique if df is clustered by a specific column
# inputs: df, name of culster column (str), method of interpolation (str), additional kwargs for pd.interpolate
# outputs: df
def timeseries_interpolation_clustered(df, timeframe, cluster_by, method='linear', **kwargs):
    # df for output
    df_full = pd.DataFrame({})
    
    # loop through clusters
    for c in df[cluster_by].unique():
        # create datetimeindex
        df_temp = df[df[cluster_by] == c]
        datetime_index = pd.DatetimeIndex(df_temp[timeframe].values)
        df_temp.set_index(datetime_index, inplace=True)
        # interpolation
        df_temp = df_temp.interpolate(method=method, **kwargs)
        # concat with full df
        df_full = pd.concat([df_full, df_temp], axis=0)
    
    df_full[timeframe] = pd.to_datetime(df_full[timeframe], errors='coerce')
    df_full.reset_index(inplace=True, drop=True)
    
    return df_full


# add columns with simple moving average to timeseries df
# inputs: df, timeframe column name (str), nod=number of days to calc srm from, columns=list of col names (default=all num. col)
# outputs: df
def timeseries_sma(df, timeframe, nod, columns=[]):
    # use all columns by default
    if columns == []:
        columns = df.select_dtypes(np.number).columns
    
    # sort timeseries
    df[timeframe] = pd.to_datetime(df[timeframe], errors='coerce')
    df = df.sort_values(by=[timeframe], ignore_index=True)

    # loop through columns
    for col in columns:
        # def sma column
        columnname_new = col + '_sma_' + str(nod) + 'd'
        df[columnname_new] = df[col]
        
        # loop through rows to add sma
        for i in range(len(df)):
            # case: days with not enogh prev days to have sma
            if i < nod-1:
                df[columnname_new].iloc[i] = np.NaN
            # case: days to calc sma
            else:
                # list to create sum for averaging
                row_sum = [df[col].iloc[i-n] for n in range(nod-1)]
                # add sma
                df[columnname_new].iloc[i] = np.round((sum(row_sum)/nod),3)
    return df

# add columns with simple moving average to timeseries df with clustered data
# inputs: df, timeframe column name (str), nod=number of days to calc srm from, 
# columns=list of col names (default=all num. col), cluster_by=col name of cluster col(str)
# outputs: df
def timeseries_clustered_sma(df, timeframe, nod, cluster_by, columns=''):
    # df for output
    df_full = pd.DataFrame({})
    
    # loop through clusters
    for c in df[cluster_by].unique():
        # create datetimeindex
        df_temp = df[df[cluster_by] == c]
        # add sma 
        df_temp = timeseries_sma(df=df_temp, timeframe=timeframe, nod=nod, columns=columns)
        df_full = pd.concat([df_full, df_temp], axis=0)
    
    df_full[timeframe] = pd.to_datetime(df_full[timeframe], errors='coerce')
    df_full.reset_index(inplace=True, drop=True)
    
    return df_full


# extract season from date column
# inputs: df, dates=col of dates (str), drop_date=drop original date col (bool, default=False)
# output: df
def df_date_to_season(df, dates, drop_dates=False):
    df[dates] = pd.to_datetime(df[dates], errors='coerce')  
    
    # create col for month and season
    df['month'] = df[dates].dt.month
    df['season'] = pd.Series(['' for x in range(len(df))])
    
    # get season
    check_season = lambda s : 'winter' if (s >= 1 and s <= 2) else ('spring' if s >= 3 and s <= 5 else ('summer' if s >= 6 and s <= 8 else ('fall' if s >= 9 and s <= 11 else ('winter' if s == 12 else np.NaN))))
    
    # loop to fill season col
    for i in range(len(df)):
        df['season'].iloc[i] = check_season(df['month'].iloc[i])
    
    # drop temp col    
    df = df.drop(['month'], axis=1)
    if drop_dates == True:
        df = df.drop([dates], axis=1)
    
    return df


# function to drop columns
# inputs: df, list of columns
# outputs: df
def drop_features(df, drop):
    df = df.drop(columns=drop)
    
    return df


