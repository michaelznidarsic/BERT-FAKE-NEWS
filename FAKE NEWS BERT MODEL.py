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

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, AveragePooling2D, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding,GRU, LSTM,Bidirectional,TimeDistributed,AveragePooling1D,GlobalMaxPool1D,Reshape,Input,Concatenate,concatenate,Attention,Permute, Lambda,RepeatVector,Add
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



# UNLIST GPU OPTION FOR TENSORFLOW 2.1. GPU RUNS OUT OF MEMORY WITH THIS NN MODEL 
# tf.config.set_visible_devices([], 'GPU')


####################################################

# BALANCE ORDINAL RELIABILITY LEVELS TO ENSURE A NON-POLARIZED MIX OF RELIABILITIES AND BIASES

# Will be labelled "reliable"
RELIABLESAMPLESIZE = 10000

# Will be labelled "unreliable"
POLITICALSAMPLESIZE = 3333
BIASSAMPLESIZE = 3333
FAKESAMPLESIZE = 3334


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


# del subsetdf


# Remove Publication names and acronyms from the text, preventing contamination of features by words that map to reliability.

cheatwords = ['wsj', 'wall street journal','new york times','national review', 'reuters', 'los angeles times', 'washington post', 'msn', 'npr', 'politico', 'forbes','nyt', 'cbs', 'nbc', 'la times', 'national public radio', 'WSJ', 'Wall Street Journal','New York Times','National Review', 'Reuters', 'Los Angeles Times', 'Washington Post', 'MSN', 'NPR', 'Politico', 'Forbes','NYT', 'CBS', 'NBC', 'LA Times', 'National Public Radio', 'POLITICO', 'REUTERS', 'NEW YORK TIMES', 'FORBES','LOS ANGELES TIMES', 'WASHINGTON POST', 'WALL STREET JOURNAL', 'breitbart', 'Breitbart', 'BREITBART','fox', 'FOX', 'Fox', 'Blaze', 'blaze', 'BLAZE']

urlartifacts = '|'.join(['.com', 'www.','.org','.ru','.us'])
derivedcheats = []
for each in list(set(subsetdf.loc[:,'publication'])):
    derivedcheats.append(re.sub(urlartifacts,'', each))


exclusions = '|'.join(cheatwords + derivedcheats)

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # html tags
    sentence = remove_tags(sen)
    # CHEAT WORDS
    sentence = re.sub(exclusions,'' , sentence, flags=re.IGNORECASE)
    # punctuation and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # single characters
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    return sentence


articles = [sent_tokenize(i) for i in subsetdf.loc[:,'content']]    # not perfect sentence splitting , e.g. : '(Remember one Bitcoin is currently worth many U.S.', 'dollars.)',

content = []
for art in articles:
    sentences = []
    for sentence in art:
        sentences.append(preprocess_text(sentence))
    content.append(sentences)
    

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

bert_layer = hub.KerasLayer('D:\\bert_en_uncased_L-12_H-768_A-12_1', trainable=False)       # I just gave up and used the suggested (worse) tfhub layer      # error without specifying signature = 'tokens'
# "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
# 'D:\\bert_en_uncased_L-12_H-768_A-12_1'

# "D:\\bert_multi_cased_L-12_H-768_A-12_1"
# https://github.com/tensorflow/tensorflow/issues/34775

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


#tokenized = [[tokenize_udf(i) , y[ind]] for ind, i in enumerate(together)]


def tokenize_udf(text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


tokenized_content =  []
for each in tqdm(content):
    concatenation = []
    for every in each:
        concatenation += tokenize_udf(every)
    tokenized_content.append(concatenation)
        

tokenized_content = np.array(tokenized_content)

total = 0
for each in tokenized_content:
    total += len(each)
print( total / len(tokenized_content) )



####################################################

# TRAIN/TEST SPLIT

samplesize = len(subsetdf.iloc[:,0])
allindex = range(0,samplesize)
trainindex = random.sample(allindex, round(0.9 * samplesize))
testindex = list(set(allindex) - set(trainindex))

train1 = tokenized_content[trainindex]
test1 = tokenized_content[testindex]

labeltrain = y[trainindex]
labeltest = y[testindex]

maxlen1 = 512

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


unpacked = [i for i in tokenized_content]
type(unpacked)
setlist = set()
for each in unpacked:
    for every in each:
        setlist.add( every )

vocab_size =   max(setlist) + 1       #  len(setlist)     #   this is BS, I have to make vocab length include all the unused IDs


####################################################

# NEURAL NETWORK MODEL

def Create_Model(length1, vocab_size):
    
	# query
    inputs1 = Input(shape=(length1,))
    embedding1 = Embedding(vocab_size, 256)(inputs1)
    
    conv1 = Conv1D(filters=256, kernel_size=2, activation='relu')(embedding1)
    drop1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)     # GlobalMaxPool1D()  MaxPooling1D(pool_size=2)
    conv2 = Conv1D(filters=256, kernel_size=3, activation='relu')(pool1)
    drop2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat1 = Flatten()(pool2)
    
    dense = Dense(4096, activation='relu', kernel_regularizer = l2(0.01))(flat1) # merged
    drop0 = Dropout(0.1)(dense)
    dense0 = Dense(1024, activation='relu', kernel_regularizer = l2(0.01))(drop0) # (drop0)
    dropl = Dropout(0.1)(dense0)
    dense1 = Dense(256, activation='relu', kernel_regularizer = l2(0.01))(dropl)
    dropm = Dropout(0.1)(dense1)
    dense2 = Dense(64, activation='relu', kernel_regularizer = l2(0.01))(dropm)
    dropn = Dropout(0.1)(dense2)
    dense3 = Dense(16,activation='relu', kernel_regularizer = l2(0.01))(dropn)
    dropo = Dropout(0.1)(dense3)
    outputs = Dense(2, activation='softmax')(dropo)                                   # number of classes   4  activation='softmax'
    model = Model(inputs=inputs1, outputs=outputs)

    model.compile(Adam(lr = 0.0005), loss='categorical_crossentropy', metrics=['accuracy'])   #  categorical_crossentropy   loss = 'mean_absolute_error'    metrics = [MeanAbsoluteError()]  ??    metrics = [RootMeanSquaredError()]

    print(model.summary())
    plot_model(model, show_shapes=True, to_file='FAKE_NEWS_BERT_ARCHITECTURE.png')
    return model


#################################################### 

# TRAIN MODEL

model = Create_Model(maxlen1, vocab_size)   #vocab_size

model.fit(train1, labeltrainhot, epochs=6, batch_size=BATCH_SIZE, validation_data=(test1, labeltesthot))  # , callbacks=[tensorboard_callback]


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

