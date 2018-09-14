# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:41:11 2017

@author: CTTC
"""

import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import nltk
import  sklearn.manifold 
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
#from keras.layers.merge import concatenate
#from keras.models import Model
#from keras.layers.normalization import BatchNormalization
#from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
#rseload(sys)
#sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = 'embeddings/'
EMBEDDING_FILE = BASE_DIR + 'SBW-vectors-300-min5.bin.gz'

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)


df = pd.read_csv('./embeddings/intents.csv', names = ['example', 'intent'])
allClasses = df['intent'].unique()

#example1= ['Quiero enviar dinero a mi padre', 'Quiero hacer una transferencia', 'transferir a mi primo mil euros', 'hacer una transferencia']
#example2= ['mi perro ladra mucho', 'los gatos son rapidos', 'el perro esta comiendo','Los gatos comen pienso']
#example3= ['quiero ver mi saldo', 'cuanto dinero tengo', 'cuantos euros tengo','ver saldo']
#example4= ['un coche rojo', 'un coche rapido', 'el automovil iva por la carretera','el policia multo el automovil verde']
#example5 = ['Necesito un prestamo de doscientos euros', 'Quiero dinero', 'Quiero una hipoteca']
##example1 = ['España', 'Madrid']
##example2 = ['Francia', 'Paris']
##example3 = ['Portugal', 'Lisboa']
##example4 = ['Italia', 'Roma']
#
#
#exampleClass = []
#exampleClass.append(example1)
#exampleClass.append(example2)
#exampleClass.append(example3)
#exampleClass.append(example4)
#exampleClass.append(example5)
#allClasses = range(len(exampleClass))   


w2v_sen_all = []
labels=[]

for classes in allClasses:
    print(classes)
    for example in df[df['intent']==classes]['example']:
        tokens = nltk.word_tokenize(example)
#        example_tokens1.append(tokens)
        w2v_sentence = []
        for token in tokens:
            try:
                word = word2vec.word_vec(token)
                w2v_sentence.append(word)
            except:
                
                print('word "'+token+'" not in vocabulary')
            
        w2v_sencence_final = np.array(w2v_sentence).sum(axis=0)/len(w2v_sentence)
#        w2v_sen_class1.append(w2v_sencence_final)
        w2v_sen_all.append(w2v_sencence_final)
        labels.append(classes)
        
#from sklearn.feature_extraction.text import TfidfVectorizer       
#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(df.example)        
#X.transform(df[df['intent']=='transferencia'])

#Compute cosine distances between pairs of examples.
#n_sentences = len(w2v_sen_all)
dist_matrix = cosine_similarity(w2v_sen_all)
plt.imshow(dist_matrix,  interpolation='nearest')
plt.show()        
    
x= np.array(w2v_sen_all)   
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y=model.fit_transform(x) 


#x = linspace(-0.001, 0.001, 10)
plt.scatter(Y[:, 0], Y[:, 1], c='blue')

for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    
    
plt.xlim([-0.0005,0.0004])
plt.ylim([-0.0003,0.0003])
plt.show()