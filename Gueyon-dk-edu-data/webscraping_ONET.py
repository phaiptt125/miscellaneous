import os
import re
import sys
import urllib.request
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd

def reset_space(input_string):
    # This function removes unnecessary white spaces.
    string_reset_space = [w for w in re.split(' ',input_string) 
                          if not w=='']
    return ' '.join(string_reset_space)

def remove_markup(input_string):
    # This function removes html merkups.
    content = [w for w in re.split('\n',input_string) if not w=='']
    content = [reset_space(w) for w in content 
               if not reset_space(w)=='']
    content = [w for w in content if not '>' in w]
    return ' '.join(content)

def clean_phrase(input_string):
    temp = input_string.lower()
    
    remove_character = [',', '.', ':', ';', '/', '(', ')', '?', '!']
    for i in remove_character:
        temp = temp.replace(i,' ')
        
    list_words = [w for w in re.split(' ',temp) if not w=='']
    
    remove_words = ['and', 'in', 'of', 'a', 'an', 'the']
    list_words = [w for w in list_words if not w in remove_words]
    
    return ' '.join(list_words)
    
def search_ONET_connector(phrase):
    original_phrase = phrase
    clean = clean_phrase(phrase)

    # (1.) form an URL
    URL_part1 = "http://www.onetcodeconnector.org/find/result?s="
    URL_part2 = re.sub(' ','+',clean) # replace space ' ' with '+'
    URL = URL_part1 + URL_part2
    # For example: 
    # https://www.onetcodeconnector.org/find/result?s=plastic+operator

    # (2.) submit and request the ONET webpage
    req = urllib.request.Request(URL)
    resp = urllib.request.urlopen(req)
    respData = resp.read()    
    soup = BeautifulSoup(respData,"html.parser") 

    # (3.) extract relevant text
    html = re.findall('Activities.*end content', 
                      str(soup.prettify()), 
                      re.DOTALL)[0]

    html = reset_space(html)

    pattern_1 = re.escape('<td class="right"')
    pattern_2 = re.escape('<td class="left"')
    pattern_3 = re.escape('<td class="center"')
    pattern = pattern_1 + '|' + pattern_2 + '|' + pattern_3
    split_html = re.split(pattern,html)[1:]

    # (4.) extract search table
    count_column = 0
    table = list()
    result = ['']*8

    for elem in split_html:
        count_column += 1

        if 1 <= count_column <= 3:
            content = str(remove_markup(elem))
            result[count_column-1] = str(content)
        elif 4 <=count_column <= 8:

            if re.findall('check\.gif', elem):
                content = 1
            else:
                content = 0

            result[count_column-1] = str(content)

        if count_column == 8:
            count_column = 0
            table.append( '\t'.join(result)  )
            result = ['']*8
    
    # (5.) generate dataframe
    names =['score','occupation','soc','check1','check2','check3','check4','check5']
    df = pd.read_csv(StringIO('\n'.join(table)),sep='\t',names=names)
    df['original_phrase'] = original_phrase
    df['clean'] = clean
    
    return df
    
    
    

