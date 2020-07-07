# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:32:54 2020

@author: mznid
"""

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import re
import math

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, AveragePooling2D, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding,GRU, LSTM,Bidirectional,TimeDistributed,AveragePooling1D,GlobalMaxPool1D,Reshape,Input,Concatenate,concatenate,Attention,Permute, Lambda,RepeatVector,Add, Input
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

import os
import bert
from bert.tokenization import bert_tokenization
import random

import matplotlib.pyplot as plt
import seaborn as sb



# UNLIST GPU OPTION FOR TENSORFLOW 2.1. SOMETIMES GPU RUNS OUT OF MEMORY WITH THIS NN MODEL 
# tf.config.set_visible_devices([], 'GPU')


####################################################

# BALANCE ORDINAL RELIABILITY LEVELS TO ENSURE A NON-POLARIZED MIX OF RELIABILITIES AND BIASES

# Will be labelled "reliable"
RELIABLESAMPLESIZE = 15000

# Will be labelled "unreliable"
POLITICALSAMPLESIZE = 5000
BIASSAMPLESIZE = 5000
FAKESAMPLESIZE = 5000


####################################################

# BALANCE POLITICAL BIAS VIA LEFT-CENTER-RIGHT APPROACH


RELIABLE = pd.read_csv('D:\\FAKENEWSOUTPUT\\BERT FAKE NEWS DATA\\RELIABLE-true.csv') #.dropna()

set(RELIABLE.loc[:,'publication'])

RELIABLE.groupby(by = 'publication').count()

CENTER = RELIABLE.loc[RELIABLE['publication'].isin(['Reuters', 'www.politico.com']) ,:]
LEFT = RELIABLE.loc[RELIABLE['publication'].isin(['Los Angeles Times', 'New York Times', 'Washington Post', 'www.cbsnews.com','www.msn.com','www.nbcnews.com','www.npr.org']) ,:]
RIGHT = RELIABLE.loc[RELIABLE['publication'].isin(['National Review', 'www.wsj.com','www.forbes.com','online.wsj.com']) ,:]
natreviewdownsample = RELIABLE.loc[RELIABLE['publication'].isin(['nationalreview.com']) ,:].reset_index()
natreviewdownsample = natreviewdownsample.loc[random.sample(range(0,len(natreviewdownsample.iloc[:,0])), round(0.02 * len(natreviewdownsample.iloc[:,0]))) ,:]
RIGHT = RIGHT.append(natreviewdownsample, ignore_index = True)

LEFT = LEFT.iloc[random.sample(range(0,len(LEFT.iloc[:,0])), 16500),:]
RIGHT = RIGHT.iloc[random.sample(range(0,len(RIGHT.iloc[:,0])), 16500),:]
CENTER = CENTER.iloc[random.sample(range(0,len(CENTER.iloc[:,0])), 12000),:]

newreliable = CENTER.loc[:,['publication','content']].append(LEFT.reset_index().loc[:,['publication','content']], ignore_index = True)
RELIABLE = newreliable.append(RIGHT.reset_index().loc[:,['publication','content']], ignore_index = True)


reliablesampler = random.sample(range(0,len(RELIABLE.iloc[:,0])), RELIABLESAMPLESIZE)


subsetdf = RELIABLE.reset_index().loc[reliablesampler,['publication', 'content']]
del RELIABLE


POLITICAL = pd.read_csv('D:\\FAKENEWSOUTPUT\\BERT FAKE NEWS DATA\\POLITICAL-mostlytrue.csv') #.dropna()
politicalsampler = random.sample(range(0,len(POLITICAL.iloc[:,0])), POLITICALSAMPLESIZE)
subsetdf = subsetdf.append(POLITICAL.reset_index().loc[politicalsampler,['publication', 'content']], ignore_index = True)
del POLITICAL


BIAS = pd.read_csv('D:\\FAKENEWSOUTPUT\\BERT FAKE NEWS DATA\\BIAS-mostlyfalse.csv') #.dropna()
biassampler = random.sample(range(0,len(BIAS.iloc[:,0])), BIASSAMPLESIZE)
subsetdf = subsetdf.append(BIAS.reset_index().loc[biassampler,['publication', 'content']], ignore_index = True)
del BIAS

FAKE = pd.read_csv('D:\\FAKENEWSOUTPUT\\BERT FAKE NEWS DATA\\FAKE-false.csv') #.dropna()
fakesampler = random.sample(range(0,len(FAKE.iloc[:,0])), FAKESAMPLESIZE)
subsetdf = subsetdf.append(FAKE.reset_index().loc[fakesampler,['publication', 'content']], ignore_index = True)
del FAKE


####################################################

#     LABELS TURN TO BINARY OR 4-LEVEL. FOR BINARY, FIRST TWO LOOPS ARE 't' AND LAST TWO ARE 'f', 4-LEVEL: r,p,b,f

labelcolumn = []                                
for each in range(0,RELIABLESAMPLESIZE):
    labelcolumn.append('t')
for each in range(0,POLITICALSAMPLESIZE):
    labelcolumn.append('f')
for each in range(0,BIASSAMPLESIZE):
    labelcolumn.append('f')
for each in range(0,FAKESAMPLESIZE):
    labelcolumn.append('f')


scrambler = random.sample(range(0, len(subsetdf.iloc[:,0])), sum([RELIABLESAMPLESIZE,POLITICALSAMPLESIZE,BIASSAMPLESIZE,FAKESAMPLESIZE]))      # could be LABELSAMPLESIZE * 4 or less
subsetdf = subsetdf.iloc[scrambler,:]
labelcolumn = list(np.array(labelcolumn)[scrambler])



# Remove Publication names and acronyms from the text, preventing contamination of features by words that map to reliability.

cheatwords = ['wsj', 'wall street journal','new york times','national review', 'reuters', 'los angeles times', 'washington post', 'msn', 'npr', 'politico', 'forbes','nyt', 'cbs', 'nbc', 'la times', 'national public radio', 'WSJ', 'Wall Street Journal','New York Times','National Review', 'Reuters', 'Los Angeles Times', 'Washington Post', 'MSN', 'NPR', 'Politico', 'Forbes','NYT', 'CBS', 'NBC', 'LA Times', 'National Public Radio', 'POLITICO', 'REUTERS', 'NEW YORK TIMES', 'FORBES','LOS ANGELES TIMES', 'WASHINGTON POST', 'WALL STREET JOURNAL', 'breitbart', 'Breitbart', 'BREITBART','fox', 'FOX', 'Fox', 'Blaze', 'blaze', 'BLAZE']

urlartifacts = '|'.join(['.com', 'www.','.org','.ru','.us'])
derivedcheats = []
for each in list(set(subsetdf.loc[:,'publication'])):
    derivedcheats.append(re.sub(urlartifacts,'', each))


exclusions = '|'.join(cheatwords + derivedcheats)

for i in tqdm(range(0, len(subsetdf.loc[:,'content']))):
    subsetdf.iloc[i,1] = re.sub(exclusions,' ' , subsetdf.iloc[i,1], flags=re.IGNORECASE)



templabel = []
for each in labelcolumn:
    if each == 't':
        templabel.append(1)
    else:
        templabel.append(0)

y = np.array(templabel)



####################################################

# BERT TOKENIZATION
 
BertTokenizer = bert_tokenization.FullTokenizer
from bert.loader import params_from_pretrained_ckpt     #  these are necessary because of weird  ImportError: cannot import name 'BertModelLayer' from 'bert' (unknown location) errors
from bert.model import BertModelLayer

bert_params = params_from_pretrained_ckpt('D:\\uncased_L-4_H-256_A-4')   # from google, not tensorflow hub
bert_layer1 = BertModelLayer.from_params(bert_params, name="bert")    # # hidden_dropout = 0.1,


model_name = 'uncased_L-4_H-256_A-4'

vocabulary_file = os.path.join('D:\\uncased_L-4_H-256_A-4\\vocab.txt')
to_lower_case = not (model_name.find("cased") == 0 or model_name.find("multi_cased") == 0)
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)



max_seq_length = 256
train_tokens = map(tokenizer.tokenize, list(subsetdf.loc[:,'content']))    # go all the way back to a list of raw strings
train_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], train_tokens)
train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

train_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), train_token_ids)
train_token_ids = np.array(list(train_token_ids))



tokenized_content = train_token_ids

total = 0
for each in tokenized_content:
    total += len(each)
print( total / len(tokenized_content) )


####################################################

# TRAIN/TEST SPLIT

samplesize = len(subsetdf.iloc[:,0])
allindex = range(0,samplesize)
trainindex = random.sample(allindex, round(0.7 * samplesize))
testindex = list(set(allindex) - set(trainindex))

train1 = tokenized_content[trainindex]
test1 = tokenized_content[testindex]

labeltrain = y[trainindex]
labeltest = y[testindex]

maxlen1 = 320         # 512 is max, but scale down to avoid memory death

train1 = pad_sequences(train1, maxlen = maxlen1, value = 0 , padding = 'post')  # maxlen
test1 = pad_sequences(test1, maxlen = maxlen1, value = 0 , padding = 'post')  # maxlen

labeltrainhot = to_categorical(labeltrain)
labeltesthot = to_categorical(labeltest)


#  Deal with rounding for batch

BATCH_SIZE = 32

trainslice = math.floor(len(train1)/BATCH_SIZE) * BATCH_SIZE
testslice = math.floor(len(test1)/BATCH_SIZE) * BATCH_SIZE

train1 = train1[:trainslice]
test1 = test1[:testslice]


# for classification
labeltrainhot = labeltrainhot[:trainslice]
labeltesthot = labeltesthot[:testslice]

labeltrainhot = np.array([np.argmax(i) for i in labeltrainhot])   # for binary_crossentropy
labeltesthot = np.array([np.argmax(i) for i in labeltesthot])


unpacked = [i for i in tokenized_content]
type(unpacked)
setlist = set()
for each in unpacked:
    for every in each:
        setlist.add( every )

vocab_size = max(setlist) + 1       #  len(setlist)     


del subsetdf


####################################################

# NEURAL NETWORK MODEL

def Create_Model(length1, vocab_size):
    model = Sequential([
        Input(shape=(length1,)),
        bert_layer1,
        Bidirectional(LSTM(maxlen1)),   #   , kernel_regularizer = l2(0.05)
        Flatten(),
        Dense(4096, activation='relu', kernel_regularizer = l2(0.1)),
        Dense(4096, activation='relu', kernel_regularizer = l2(0.1)),
        Dense(4096, activation='relu', kernel_regularizer = l2(0.1)),
        Dense(512, activation='relu', kernel_regularizer = l2(0.1)),
        Dense(1, activation='sigmoid')     # Dense(2, activation='softmax')     with 'categorical_crossentropy'
        ])
    
    model.compile(Adam(lr = 0.0005), loss='binary_crossentropy', metrics=['accuracy'])   #  categorical_crossentropy   loss = 'mean_absolute_error'    metrics = [MeanAbsoluteError()]  ??    metrics = [RootMeanSquaredError()]
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='FAKE_NEWS_BERT_ARCHITECTURE.png')
    
    return model


#################################################### 

# TRAIN MODEL

model = Create_Model(maxlen1, vocab_size)   #vocab_size

model.fit(train1, labeltrainhot, epochs = 3, batch_size = BATCH_SIZE, validation_data = (test1, labeltesthot))  # , callbacks=[tensorboard_callback]




# model.save_weights('/FAKE NEWS MODEL/FAKENEWSdeployment.model')
# model.save('REALfakenewsdeploy.model')



####################################################

# CREATE MODEL PREDICTIONS FOR CONFUSION MATRIX

nnpredictions = model.predict([test1])

correct = []
for ind, each in enumerate(nnpredictions):
    if np.argmax(each) == np.argmax(labeltesthot[ind]):
        correct.append(1)
    else:
        correct.append(0)

print('Validation Accuracy: ', sum(correct)/len(correct))

cf = pd.crosstab(np.array([np.argmax(i) for i in nnpredictions]), np.array( [np.argmax(j) for j in labeltesthot] ))

cf.index = ['Unreliable', 'Reliable']
cf.columns = ['Unreliable', 'Reliable']

sb.heatmap(cf, annot = True, cmap = "Blues", fmt='g')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Dev Confusion Matrix')


