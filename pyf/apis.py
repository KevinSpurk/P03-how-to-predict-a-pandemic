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

    