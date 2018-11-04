import os
import re
import platform
import sys
import spacy
import pandas as pd
import en_core_web_sm

nlp = en_core_web_sm.load(disable=['parser', 'tagger','ner'] )

df_onet = pd.read_csv("Occupation Data.txt")
print( df_onet.head() )


print('-----DONE-----')
