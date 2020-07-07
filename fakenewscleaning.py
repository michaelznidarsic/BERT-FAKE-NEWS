# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:58:10 2019

@author: mznid
"""

# DATA CLEANING

import pandas as pd
import numpy as np
import math
import os
import re
import csv
import random
import seaborn as sb
import matplotlib.pyplot as plt

from tqdm import tqdm 

import ctypes

csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
csv.field_size_limit()


os.chdir('D:\\FAKENEWSOUTPUT')

raw = pd.read_csv('longform.csv')

print(raw.columns.values)

cleanarray = []
for index, row in tqdm(raw.iterrows()):
    if row[4] != '' and isinstance(row[4], str):                 # CONTENT NOT EMPTY STRING                                         #  DATE NOT EMPTY STRING
        if row[7] != '':                                     # PUBLICATION NOT EMPTY STRING
            if row[7] in [ 'New York Times' , 'Washington Post' ,'Reuters' , 'National Review' , 'Los Angeles Times' , 'The Verge'] :
                cleanarray.append(list(row[0:5]) + [row[7]])


del raw

cleandf = pd.DataFrame(cleanarray)
print(cleandf.shape)

print(cleandf.columns.values)

del cleanarray

cleandf.columns = ['id', 'title', 'author', 'date', 'content', 'publication']


subsetdf = cleandf.loc[:, ['publication', 'content'] ]

del cleandf


subsetdf.publication.unique()


# RELIABLE AUGMENTATION

exceptions = ['cato.org/blog', 'nationalreview.com', 'www.forbes.com', 'online.wsj.com', 'www.nbcnews.com', 'www.wsj.com', 'www.cbsnews.com', 'www.msn.com',  'www.npr.org',  'www.politico.com']

fakeid = []
fakesite = [] 
faketype = []
fakecontent = []
fakeauthor = []
with open('D:\\news_cleaned_2018_02_13\\news_cleaned_2018_02_13.csv', encoding="utf8") as source:
    read = csv.reader(source)
    counter = 0
    for row in tqdm(read):
        if counter >= 500000:
            break
        if len(row) >= 6:
            if len(row[5]) > 500:         # MINIMUM CHARLENGTH
                if row[2] in exceptions:                           # or row[3] == 'bias'    'junksci'    'conspiracy'   'hate'     'fake'                       'reliable'
                    counter += 1
                    fakeid.append(row[1])
                    fakesite.append(row[2])
                    faketype.append(row[3])
                    fakecontent.append(re.sub(r'[\n\r]+', ' ', row[5]))
                    fakeauthor.append(row[10])

len(fakesite)

uniquesourcelist = []

for each in fakesite:
    if not each in uniquesourcelist:
        uniquesourcelist.append(each)


fakedf = pd.DataFrame({'publication': fakesite, 'content': fakecontent, 'author': fakeauthor})


newpublisherset = set()
with open('D:\\news_cleaned_2018_02_13\\news_cleaned_2018_02_13.csv', encoding="utf8") as source:
    read = csv.reader(source)
    counter = 0
    for row in tqdm(read):
        if len(row) >= 6:
            if type(row[2]) == str:
                if row[3] == 'reliable':
                    newpublisherset.add(row[2])
                
                
                
# subdf = fakedf.loc[:,['publication', 'content']]
# del fakedf
# subsetdf.append(subdf, ignore_index = True).to_csv('D:\\FAKENEWSOUTPUT\\BERT FAKE NEWS DATA\\RELIABLE-true.csv')

#####
#subsetdf.to_csv('D:\\FAKENEWSOUTPUT\\BERT FAKE NEWS DATA\\RELIABLE-true.csv')
#####



# POLITICAL

fakeid = []
fakesite = [] 
faketype = []
fakecontent = []
fakeauthor = []
with open('D:\\news_cleaned_2018_02_13\\news_cleaned_2018_02_13.csv', encoding="utf8") as source:
    read = csv.reader(source)
    counter = 0
    for row in read:
        if counter >= 500000:
            break
        if len(row) >= 6:
            if len(row[5]) > 500:  # MINIMUM CHARLENGTH
                if row[3] == 'political' and row[2] not in exceptions:                           # or row[3] == 'bias'    'junksci'    'conspiracy'   'hate'     'fake'                       'reliable'
                    counter += 1
                    fakeid.append(row[1])
                    fakesite.append(row[2])
                    faketype.append(row[3])
                    fakecontent.append(re.sub(r'[\n\r]+', ' ', row[5]))
                    fakeauthor.append(row[10])

len(fakesite)

uniquesourcelist = []

for each in fakesite:
    if not each in uniquesourcelist:
        uniquesourcelist.append(each)


fakedf = pd.DataFrame({'publication': fakesite, 'content': fakecontent, 'author': fakeauthor})


# fakedf.to_csv('D:\\FAKENEWSOUTPUT\\BERT FAKE NEWS DATA\\POLITICAL-mostlytrue.csv')
# del fakedf


# BIAS

fakeid = []
fakesite = [] 
faketype = []
fakecontent = []
fakeauthor = []
with open('D:\\news_cleaned_2018_02_13\\news_cleaned_2018_02_13.csv', encoding="utf8") as source:
    read = csv.reader(source)
    counter = 0
    for row in read:
        if counter >= 500000:
            break
        if len(row) >= 6:
            if len(row[5]) > 500:        # MINIMUM CHARLENGTH
                if row[3] == 'bias' and row[2] not in exceptions:                           # or row[3] == 'bias'    'junksci'    'conspiracy'   'hate'     'fake'                       'reliable'
                    counter += 1
                    fakeid.append(row[1])
                    fakesite.append(row[2])
                    faketype.append(row[3])
                    fakecontent.append(re.sub(r'[\n\r]+', ' ', row[5]))
                    fakeauthor.append(row[10])

len(fakesite)

uniquesourcelist = []

for each in fakesite:
    if not each in uniquesourcelist:
        uniquesourcelist.append(each)


fakedf = pd.DataFrame({'publication': fakesite, 'content': fakecontent, 'author': fakeauthor})



# fakedf.to_csv('D:\\FAKENEWSOUTPUT\\BERT FAKE NEWS DATA\\BIAS-mostlyfalse.csv')
# del fakedf




# FAKE 


fakeid = []
fakesite = [] 
faketype = []
fakecontent = []
fakeauthor = []
with open('D:\\news_cleaned_2018_02_13\\news_cleaned_2018_02_13.csv', encoding="utf8") as source:
    read = csv.reader(source)
    counter = 0
    for row in read:
        if counter >= 500000:
            break
        if len(row) >= 6:
            if len(row[5]) > 500:     # MINIMUM CHARLENGTH
                if row[3] == 'fake' and row[2] not in exceptions:                           # or row[3] == 'bias'    'junksci'    'conspiracy'   'hate'     'fake'                       'reliable'
                    counter += 1
                    fakeid.append(row[1])
                    fakesite.append(row[2])
                    faketype.append(row[3])
                    fakecontent.append(re.sub(r'[\n\r]+', ' ', row[5]))
                    fakeauthor.append(row[10])

len(fakesite)

uniquesourcelist = []

for each in fakesite:
    if not each in uniquesourcelist:
        uniquesourcelist.append(each)


fakedf = pd.DataFrame({'publication': fakesite, 'content': fakecontent, 'author': fakeauthor})



# fakedf.to_csv('D:\\FAKENEWSOUTPUT\\BERT FAKE NEWS DATA\\FAKE-false.csv')
# del fakecontent


