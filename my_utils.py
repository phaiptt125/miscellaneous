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

'''
not using gensim stopword -> too many words. For example, "call" in "call center" will be removed.
'''

# nltk
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))

# scacy
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load( disable=['parser', 'tagger','ner'] )

'''

Functions related to text preprocessing


'''

def main_preprocess(text, extra_remove = None):
    
    if len(str(text)) == 0:
        return ""
    else:
        # initial replacement
        clean_text = text
        clean_text = clean_text.replace('e-Commerce','eCommerce')
        # tokenize
        tokens = [str(w.lemma_.lower()) for w in nlp(clean_text)]
        # remove stopwords and other selected words
        selected_tokens = [w for w in tokens if not w in STOPWORDS]
    
        more_remove_words = ['will','would','could','also','might','including','not','say','per',
                             'includes','include','included','still','your','yours','that','this','it',
                             'get','go','like','be','have','make','follow','much','many','among','well',
                             'use','understand','find', 'identify','ensure','explore','improve','across',
                             'drive','mr','ms','mrs']

        selected_tokens = [w for w in selected_tokens if not w in more_remove_words]
        selected_tokens = [re.sub('[^a-z0-9]','',w) for w in selected_tokens]

        if extra_remove is not None:
            selected_tokens = [w for w in selected_tokens if not w in extra_remove]

        #3.) final replacement 
        replacement_tokens = []

        for token in selected_tokens:
            if token == 'datum':
                replacement_tokens.append('data')
            else:
                replacement_tokens.append(token)

        join_tokens = ' '.join(replacement_tokens) 

        return ' '.join([w for w in re.split(' ',join_tokens) if not w=='']) 

def clean_from_special_symbols_and_lower(string):
    """regex for cleaning from special symbols, leaving only letters
    """
    regex = "[^a-zA-Z \n]"
    return re.sub(regex, "", string).lower()

def clean_cognet(string):
    """regex for cleaning from special symbols, 
    leaving only letters, numbers, ., ,
    """
    # regex for cleaning from special symbols, 
    # leaving only letters and numbers
    regex = "[^a-zA-Z0-9 \n,.!?/()-:;]"
    return re.sub(regex, "", string)


'''

Functions related to embedding models


'''

def load_model(model_filename, model_path = ""):

    model = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join(model_path, model_filename), binary=True)
    
    print(model_filename + ' downloaded.')

    return model

def get_embedding_matrix(list_text, model):
    
    '''
    This function takes a list of (cleaned) text and compute embedding.
    
    input:
        - list_text: a list of tokens (python list, each element is one document)
        - model: word embedding model
    output: a numpy matrix of size (n_doc, n_model) 
        - n_doc: number of documents = len(list_text)
        - n_model: model dimension
    '''
    
    #initialize output matrix
    output_matrix = np.zeros((len(list_text), model.vector_size))
    
    for i, text in enumerate(list_text): #loop by document 
        
        # split into tokens and keep only words in the model
        tokens = [w for w in re.split(' ',text) if w in model.vocab.keys() if not w=='']
        
        if len(tokens) == 0: # return a row of zeros if no word is in the model
            vector_this_text = np.zeros((1,model.vector_size))
        else:
            
            #initialize temporary matrix of this document of size (n_tokens, n_model)
            matrix_this_text = np.zeros((len(tokens),model.vector_size))  

            for j, word in enumerate(tokens): # loop by word
                matrix_this_text[j,:] = model[word] # record the vector
                
            # sum over words to get a vector representation of the whole document
            # note: np.sum(x,0) means sum over row
            vector_this_text = np.sum(matrix_this_text,0).reshape(1, -1)
            
            # assert that the vector has the correct dimension
            assert np.shape(vector_this_text) == (1,model.vector_size)
            
        # record the final vector for each document
        output_matrix[i,:] = vector_this_text
            
    return output_matrix

def cosine_similarity_score(vec1, vec2):
    return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)).squeeze())




def cluster_need(x):
    need_dict = {'benefits administration' : 'human resources', 
                 'compensation management' : 'human resources', 
                 'employee engagement' : 'human resources', 
                 'payroll' : 'human resources', 
                 'talent management' : 'human resources', 
                 'training and development' : 'human resources',
                 'applicant tracking system' : 'human resources',
                 'mobile workforce' : 'human resources',
                 'call center' : 'customer service', 
                 'customer experience' : 'customer service', 
                 'customer relationship management' : 'customer service', 
                 'customer service' : 'customer service', 
                 'help desk' : 'customer service',
                 'loyalty management' : 'customer service', 
                 'personalization' : 'customer service',
                 'content management system' : 'content management',
                 'digital content management' : 'content management',
                 'digital asset management' : 'content management', 
                 'document management' : 'content management', 
                 'email and calendar' : 'content management', 
                 'web content management' : 'content management',
                 'order management' : 'supply chain management', 
                 'procurement' : 'supply chain management', 
                 'supply chain management' : 'supply chain management',
                 'campaign management' : 'sales and marketing',
                 'configure, price, quote' : 'sales and marketing', 
                 'content marketing' : 'sales and marketing', 
                 'digital marketing' : 'sales and marketing', 
                 'sales' : 'sales and marketing', 
                 'e-commerce' : 'sales and marketing',
                 'ecommerce' : 'sales and marketing',
                 'lead management' : 'sales and marketing',
                 'business process management' : 'business performance', 
                 'enterprise performance management' : 'business performance', 
                 'enterprise resource planning' : 'business performance',
                 'enterprise asset management' : 'business performance',
                 'project management' : 'business performance',
                 'accounting' : 'accounting',
                 'automation' : 'automation',
                 'disaster recovery' : 'disaster recovery',
                 'fraud protection' : 'fraud protection',
                 'risk and compliance' : 'risk and compliance',
                 'financial reporting' : 'financial reporting',
                 'fixed asset management' : 'fixed asset management'}

    return need_dict[x.lower()]


def assign_week(date):
    #example: 2018-09-30
    regex = re.findall('\d+',date)
    YYYY = int(regex[0])
    MM = int(regex[1])
    DD = int(regex[2])
    WW = int(datetime.date(YYYY, MM, DD).strftime("%V"))
    
    if YYYY == 2018:
        week = WW
    elif YYYY == 2019:
        week = WW + 52
    elif YYYY == 2020:
        week = WW + 52 + 52
                
    return week 

def assign_weekday(date):
    #example: 2018-09-30
    regex = re.findall('\d+',date)
    YYYY = int(regex[0])
    MM = int(regex[1])
    DD = int(regex[2])
    
    weekday = int(datetime.date(YYYY, MM, DD).weekday())
                
    return weekday



