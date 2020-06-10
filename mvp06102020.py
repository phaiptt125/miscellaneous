# -*- coding: utf-8 -*-
# Phai Phongthiengtham
# 06/10/2020

import requests
import re 
import os
import json
import sys
import csv
import time
import datetime
import difflib
import operator
from io import StringIO
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

# hashing packages
import hashlib
from simhash import Simhash, SimhashIndex
from snapy import MinHash, LSH

# nltk package
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk_stopwords = list(set(stopwords.words('english')))

# spacy package
import spacy
import en_core_web_sm

# sentence tokenization 
nlp_sentence = en_core_web_sm.load()

# word tokenization (disable unused components for faster implementation)
nlp_word = en_core_web_sm.load( disable=['parser', 'tagger','ner'] )

# Sentence Transformers: Multilingual Sentence Embeddings using BERT
from sentence_transformers import SentenceTransformer

'''
'''

class ClauseAnalyzer(object):
    def __init__(self, abbrevation = None, stopwords = nltk_stopwords):
        self.stopwords = stopwords
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def remove_space(self, text):
        return ' '.join([w for w in re.split(' ',text) if not w==''])

    def clean_sentence(self, sentence):
        if str(len(sentence)) == 0:
            sentence_processed = ''
        else:
            tokens = [w.text.lower() for w in nlp_word(sentence)]
            tokens = [re.sub('[^a-z0-9]+', ' ', w) for w in tokens if not w in ['','a','an','the']]
            sentence_processed = self.remove_space(' '.join(tokens))
        return sentence_processed 

    def clean_before_tokenization(self, text):
        text = text.replace('’',"'")
        text = text.replace('_',' ')
        text = text.replace('\t',' ')
        
        match_fullstop_nospace = re.findall('\w+\.[A-Z]+\w+' ,text)
    
        if match_fullstop_nospace:
            for match_pattern in match_fullstop_nospace:
                if match_pattern in ['ibm.com','www.ibm']:
                    pass
                else:
                    #print(match_pattern + ' => ' + re.sub('\.','. ', match_pattern))
                    text = text.replace(match_pattern, re.sub('\.','. ', match_pattern))
                
        return self.remove_space(text)

    def main_preprocess(self, text):
        s_original = [self.remove_space(w.text) for w in nlp_sentence(self.clean_before_tokenization(text)).sents]
        s_clean = [self.clean_sentence(w) for w in s_original]
        
        s_original_out = []
        s_clean_out = []
        
        for s_indx in range(0, len(s_clean)):
            this_s_clean = s_clean[s_indx]
            this_s_original = s_original[s_indx]
            
            if not this_s_clean in s_clean_out:
                s_clean_out.append(this_s_clean)
                s_original_out.append(this_s_original)
                
        return {'s_original': s_original_out, 's_clean': s_clean_out}
    
    def geo_entity_detection(self, sentence):
        geo_entity = []
        
        doc = nlp_sentence(sentence)
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                geo_entity.append(ent.text)
        return geo_entity
    
    def clause_similarity(self, s_clean1, s_clean2, method = 'cosine'):
        if method == 'jaccard':
            t1 = set([w for w in re.split(' ', ' '.join(s_clean1)) if not w=='']) 
            t2 = set([w for w in re.split(' ', ' '.join(s_clean2)) if not w==''])
            score = len(t1.intersection(t2)) / len(t1.union(t2))
        elif method == 'cosine':
            t1 = np.mean(self.model.encode(s_clean1), 0).reshape(1,-1)
            t2 = np.mean(self.model.encode(s_clean2), 0).reshape(1,-1)
            score = float(cosine_similarity(t1,t2)[0])
        return score
    
    def get_ngrams(self, s, width = 3):
        if len(str(s)) <= width:
            out = str(s)
        else:
            out = [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]
        return out
    
    def ngram_similarity(self, s1, s2):
        t1 = set(self.get_ngrams(s1, width = 3)) 
        t2 = set(self.get_ngrams(s2, width = 3))
        score = len(t1.intersection(t2)) / len(t1.union(t2))
        return score
    
    def analyze_difference(self, list1, list1_clean, list2, list2_clean):
    
        all_element = list(set(list1_clean + list2_clean))
        elem2hash = dict()
    
        for elem in all_element:
            hash_val = hashlib.md5(str(elem).encode()).hexdigest()
            elem2hash[elem] = hash_val
    
        list1_hash = [elem2hash[w] for w in list1_clean]
        list2_hash = [elem2hash[w] for w in list2_clean]
    
        diff = difflib.Differ().compare(list1_hash, list2_hash)
    
        diff1 = []
        diff2 = []
        diff1_clean = []
        diff2_clean = []
    
        for obj in diff:
            obj_hash = obj.replace('+','').replace('-','').replace(' ','')
            if '+' in obj: # "+" means the hash is in list 2 but not list 1
                h_indx = list2_hash.index(obj_hash)
    
                diff1.append('')
                diff2.append(list2[h_indx])
                diff1_clean.append('')
                diff2_clean.append(list2_clean[h_indx])
    
            elif '-' in obj: # "-" means the hash is in list 1 but not list 2
                h_indx = list1_hash.index(obj_hash)
    
                diff1.append(list1[h_indx])
                diff2.append('')
                diff1_clean.append(list1_clean[h_indx])
                diff2_clean.append('')
    
        assert(len(diff1) == len(diff2) == len(diff1_clean) == len(diff2_clean))
        
        return diff1, diff1_clean, diff2, diff2_clean
    
    def realign_difference(self, diff1, diff1_clean, diff2, diff2_clean):
    
        realign1 = diff1[:]
        realign2 = diff2[:]
        realign1_clean = diff1_clean[:]
        realign2_clean = diff2_clean[:]
    
        for indx_d1, d1_clean in enumerate(diff1_clean):
            if not d1_clean == '':
                sim_d2 = [self.ngram_similarity(d1_clean, w) for w in diff2_clean]
                indx_d2 = np.array(sim_d2).argsort()[::-1][0]
                max_sim_d2 = sim_d2[indx_d2]
                if max_sim_d2 >= 0.4 and abs(indx_d1 - indx_d2) <= 2:
                    d2_clean = diff2_clean[indx_d2]
                    d2 = diff2[indx_d2]
                    if realign2_clean[indx_d1] == '':
                        realign2_clean[indx_d1] = d2_clean
                        realign2_clean[indx_d2] = ''
                        realign2[indx_d1] = d2
                        realign2[indx_d2] = ''
                        
        assert(len(realign1)==len(realign2)==len(realign1_clean)==len(realign2_clean))
        
        realign1_out = []
        realign2_out = []
        realign1_clean_out = []
        realign2_clean_out = []
        
        for indx_r in range(0, len(realign1)):
            r1 = realign1[indx_r]
            r2 = realign2[indx_r]
            r1_clean = realign1_clean[indx_r]
            r2_clean = realign2_clean[indx_r]
            
            if (r1 == '') & (r2 == '') & (r1_clean == '') & (r2_clean == ''):
                pass
            else:
                realign1_out.append(r1)
                realign2_out.append(r2)
                realign1_clean_out.append(r1_clean)
                realign2_clean_out.append(r2_clean)
        
        return  realign1_out, realign1_clean_out, realign2_out, realign2_clean_out
        
    def highlight_difference(self, sent1_token, sent2_token):
        sent1 = []
        sent2 = []
        
        diff = difflib.Differ().compare(sent1_token, sent2_token)
        for obj in diff:
            obj_value = re.sub('^\-\s|^\+\s','',obj)
            if re.findall('^\+',obj):
                sent2.append(' <highlight>' + obj_value + '</highlight> ')
            elif re.findall('^\-',obj):
                sent1.append(' <highlight>' + obj_value + '</highlight> ')
            elif re.findall('^\?',obj):
                pass
            else:
                sent1.append(obj_value)
                sent2.append(obj_value)
                
        sent1 = ' '.join([w for w in re.split(' ',''.join(sent1)) if not w==''])
        sent2 = ' '.join([w for w in re.split(' ',''.join(sent2)) if not w==''])
        
        sent1 = sent1.replace(' ,',',')
        sent2 = sent2.replace(' ,',',') 

        sent1 = sent1.replace(' .','.')
        sent2 = sent2.replace(' .','.')
        
        return sent1, sent2

    
    def clause_comparison(self, clause1_text, clause2_text):
        
        clause1 = self.main_preprocess(clause1_text)
        clause2 = self.main_preprocess(clause2_text)
        
        difference1 = []
        difference2 = []
        
        if ' '.join(clause1['s_clean']) == ' '.join(clause2['s_clean']):
            score = 1
            is_identical = True
        else:
            score = self.clause_similarity(clause1['s_clean'], clause2['s_clean'], method = 'jaccard')
            is_identical = False
            
            diff1, diff1_clean, diff2, diff2_clean = self.analyze_difference(clause1['s_original'], clause1['s_clean'], clause2['s_original'], clause2['s_clean'])
            realign1, realign1_clean, realign2, realign2_clean = self.realign_difference(diff1, diff1_clean, diff2, diff2_clean)
        
            diff1_out = []
            diff2_out = []
            
            for indx in range(0, len(realign1)):
                sent1_token = [w.text for w in nlp_word(realign1[indx])]
                sent2_token = [w.text for w in nlp_word(realign2[indx])]
                sent1_out, sent2_out = self.highlight_difference(sent1_token, sent2_token)
                
                diff1_out.append(sent1_out)
                diff2_out.append(sent2_out)
                
            assert(len(diff1_out) == len(diff2_out))
            
            for indx in range(0, len(diff1_out)):
                obj1 = {'sentence' : diff1_out[indx]}
                obj2 = {'sentence' : diff2_out[indx]}
                difference1.append(obj1)
                difference2.append(obj2)

        json_dump = {}
        json_dump['is_identical'] = is_identical
        json_dump['score'] = score
        json_dump['difference1'] = difference1
        json_dump['difference2'] = difference2

        json_out = json.dumps(json_dump)
                       
        return json_out

#...........................................#
# code testing    
    
'''
clause1_text = 'I have a cat in my apartment. This sentence is identical. This sentence is not supposed to be here. This sentence is also identical.'
clause2_text  = 'I have a dog in my apartment. This sentence is identical. This sentence is also identical.'
print('PRINTING SAMPLE TEXT 1')
print(clause1_text)
print('------')
print('PRINTING SAMPLE TEXT 2')
print(clause2_text)

analyzer = ClauseAnalyzer()

json_out = json.loads(analyzer.clause_comparison(clause1_text, clause2_text))

print(json_out)

for d1 in json_out['difference1']:
    print(d1)
    print('')
    
for d2 in json_out['difference2']:
    print(d2)
    print('')
'''