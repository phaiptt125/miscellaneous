import os
import re
import platform
import sys
import spacy
import pandas as pd
import en_core_web_sm

nlp = en_core_web_sm.load(disable=['parser', 'tagger','ner'] )

# string replace
def cleanup(text):
    if text == '': # allows for possibility of being empty 
        output = ''
    else:
        text = text.replace("'s", " ")
        text = text.replace("n't", " not ")
        text = text.replace("'ve", " have ")
        text = text.replace("'re", " are ")
        text = text.replace("'m","  am ")
        text = text.replace("'ll","  will ")
        text = text.replace("-"," ")
        text = text.replace("/"," ")
        text = text.replace("("," ")
        text = text.replace(")"," ")
        text = re.sub(r'[^A-Za-z ]', '', text) #remove all characters that are not A-Z, a-z or 0-9
        output = ' '.join([w for w in re.split(' ',text) if not w=='']) #remove extra spaces 
    return output  

# pre-process text
def main_preprocess(text):
    text = str(text) # make sure the input is actually string
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    if text == '': # allows for possibility of being empty 
        output = ''
    else:
        tokens = [w.lemma_.lower() for w in nlp(cleanup(text))] # cleanup and tokenize
        output = ' '.join([w for w in tokens if not w==''])
    return output


df = pd.read_csv("Occupation Data.txt", sep = '\t', header = 0)
df['CleanText'] = df['Description'].apply(lambda x: main_preprocess(x))

print( df.head() )

df.to_csv("ONET_preprocessed.txt", sep = '\t', index = False)




print('-----DONE-----')
