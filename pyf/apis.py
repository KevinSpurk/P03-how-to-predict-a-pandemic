# functions to manage tasks that use APIs

import pandas as pd
import numpy as np
import datetime
import math
import random
import re
import time
from translate import Translator
from pytrends.request import TrendReq
from pytrends import dailydata
from datetime import date, timedelta
from pyf.imex import df_to_csv_format
import tweepy as tw 


# DeepL

# translate a list of words into various languages
# inputs: DeepL authentication key string, list of words (str) for translation, dict with 2-char id of countries as keys and 2-char id of language as values for all target languages
# outputs: dict and df with translation results
def deepl_translate_list(auth, keywords, lang_out, lang_in='en'):
    # dict to store translation results
    keywords_translated = {}
    
    # loop through target languages
    for country, lang in lang_out.items():
        if lang != lang_in:
            lang_translator = Translator(provider='deepl', from_lang=lang_in, to_lang=lang, secret_access_key=auth)
            
            # list to store the translated keywords
            keywords_results = []
            
            # loop through keyword list to translate
            for keyword in keywords:
                result = lang_translator.translate(keyword)
                keywords_results.append(result)
            
            # add translaltions for one lang to dict
            keywords_translated[country] = keywords_results
        
        # add original lang
        else:
            keywords_translated[country] = keywords
    
    # df with translaltions
    df_keywords_translated = pd.DataFrame(keywords_translated)
            
    return keywords_translated, df_keywords_translated



# Google

# get the search trend data for a series of keywords in multiple languages
# inputs: dict with 2-char id of starget countries as keys and list of keywords in target language as values ('GB' has to be the first lang),
# start and end date in format YYYY-MM, sleep timer in sec
# outputs: dict with search trend results, creates csv files of results for each language
def g_trends_ml(kw_dict, dt_start, dt_end, sleep_time=60):
    # df for all results
    trends_all = pd.DataFrame({})
    
    # loop through lang
    for geo in kw_dict.keys():
        # df for results of one loc
        search_trend = pd.DataFrame({})
        
        # loop through keywords
        for k in kw_dict[geo]:
            # get df with search trend for kw
            k_trend = dailydata.get_daily_data(k, int(dt_start.split('-')[0]), int(dt_start.split('-')[1]), int(dt_end.split('-')[0]), int(dt_end.split('-')[1]), geo=geo)
            time.sleep(sleep_time)
           
            # transform df
            k_trend = g_trends_df_transform(k_trend)
            
            # add missing dates
            k_trend_completed = g_trends_complete_timeframe(df=k_trend, kw=k)
            
            # add df with search trend for kw to df for results of one loc
            search_trend['date'] = k_trend_completed['date']
            search_trend[k] = k_trend_completed[k]
        
        # add loc to df 
        search_trend['location'] = geo
        # export df to csv
        search_details = dt_start + '_' + dt_end + '_' + geo
        df_to_csv_format(df=search_trend, name_cst='g_trend', name_var=search_details, index=False)
        
        # concat df with df holding all results
        if len(trends_all) > 0:
            search_trend.columns = trends_all.columns
        trends_all = pd.concat([trends_all, search_trend], axis=0)
    
    trends_all.reset_index(inplace=True)
    trends_all = trends_all.sort_values(by=['date'], ignore_index=True)

    return trends_all


# create data pionts in df for dates where search trend data is missing, sub func of g_trends_multilang
# inputs: df of search trend values for one keyword in one language, string of keyword
# outputs: df of search trend values with added NaN for missing dates 
def g_trends_complete_timeframe(df, kw):
    # set start and end date for loop
    dt_s = df['date'][0]
    pos_end = len(df) - 1
    dt_e = df['date'][pos_end]
    
    # lists for missing days
    dates_missing = []
    trend_missing = []
    
    # loop to find missing dates in df
    while dt_s <= dt_e:
        date_check = False
        for d in df['date']:
            if d == dt_s:
                date_check = True
        # append missing date to lists
        if date_check == False:
            dates_missing.append(dt_s)
            trend_missing.append(np.NaN)
        dt_s += timedelta(days=1)
    
    # df for missing dates
    df_dates_missing = pd.DataFrame({'date': dates_missing, kw: trend_missing})
    df_completed = pd.concat([df, df_dates_missing], axis=0)
    df_completed['date'] = pd.to_datetime(df_completed['date'], errors='coerce')
    df_completed = df_completed.sort_values(by=['date'], ignore_index=True)
    
    return df_completed   


# transform the search trend fetch into a usable df, sub func of g_trends_multilang
# inputs: df with results from 'get_daily_data'
# outputs: df reformated 
def g_trends_df_transform(df):
    # drop unwanted col
    if len(df.columns) > 1:
        df = df.drop(df.iloc[:, :-1], axis=1)
            
    # get date as col
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.date
    return df



# Twitter

# get authentification for twitter API
# inputs: txt file with credentials (l.2 api key, l.4 api key secret, l.6 token, l.8 token secret, l.10 bearer token)
# outputs: api and client api access points
def tweepy_auth(credentials):
    # open cred file
    with open(credentials) as t:
        cred = t.readlines()
    
    # read lines with cred
    tw_api_key = cred[1][:-1]
    tw_api_key_sec = cred[3][:-1]
    tw_token = cred[5][:-1]
    tw_token_sec = cred[7][:-1]
    tw_bearer_token = cred[9]

    # twitter API authentication
    tw_auth = tw.OAuthHandler(tw_api_key, tw_api_key_sec)
    tw_auth.set_access_token(tw_token, tw_token_sec)
    client = tw.Client(bearer_token=tw_bearer_token)

    # authentication check
    api = tw.API(tw_auth, wait_on_rate_limit=True)
    try:
        api.verify_credentials()
        print("Authentication OK")
    except:
        print("Error during authentication")
    
    return api, client
