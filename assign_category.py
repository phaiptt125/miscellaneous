import requests, re, os, json, sys, csv, time, datetime
import operator, curl
from io import StringIO
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)
from scipy import spatial
from pprint import pprint

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions

# sklearn 
from sklearn.metrics.pairwise import cosine_similarity

# gensim
import gensim
import gensim.utils

def assign_category(df_to_match, df_category, topn):
    
    '''
    This function takes in a list of embeddings from "df_to_match" 
    and assign the topn closest categories from "df_category". 
    '''
    
    numbered_columns = ['v' + str(w) for w in range(0,300)]
    
    # the second column of "df_business_need" is the business need cluster
    category_column_name = df_category.columns[1]
        
    category_list = df_category[category_column_name].tolist()
    category_embedding = np.array(df_category[numbered_columns])

    df_output = df_to_match.copy()
    df_output = df_output[['company_id', 'article_id']]
    
    for n in range(0,topn):
        df_output['top' + str(n)] = '' 
        df_output['top' + str(n) + '_sim'] = 0

    for category in category_list:
        df_output[category] = 0

    for to_match_indx, to_match in df_to_match.iterrows():
    
        to_match_embedding = np.array(to_match[numbered_columns]).reshape(1,-1)
    
        similarity_matrix = cosine_similarity(to_match_embedding, category_embedding)

        assert np.shape(similarity_matrix) == (1, len(category_list))
    
        similarity = similarity_matrix[0]

        '''
        Note: "similarity_matrix" can actually contains multiple vectors, which will
        be the case if we put cosine_similarity(X,Y) where X and Y, on
        its own, is also contains multiple vectors.

        Let's call X = [x1, x2, x3, ...] and Y = [y1, y2, y3, ...]
        Then we have Z = cosine_similarity(X,Y)

        where Z = [z1, z2, z3, ...]

        Here z1 = [cos(x1,y1), cos(x1,y2), cos(x1, y3),...]
        and z2 = [cos(x2,y1), cos(x2,y2), cos(x2, y3),...]
        and z3 = [cos(x3,y1), cos(x3,y2), cos(x3, y3),...]        
        '''
        
        # get indices of the topn highest scores
        argmax_similarity = similarity.argsort()[::-1][:topn]
    
        # get score values and corresponding catagory
        max_similarity = [similarity[w] for w in argmax_similarity]
        max_category = [category_list[w] for w in argmax_similarity]
    
        for n in range(0,topn):
            df_output.loc[to_match_indx, 'top' + str(n) + '_sim'] = max_similarity[n]
            df_output.loc[to_match_indx, 'top' + str(n)] = max_category[n]
        
    return df_output

