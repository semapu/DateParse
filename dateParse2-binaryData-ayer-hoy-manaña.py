# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:27:54 2017

@author: CTTC
"""
#librería algebre lineal
import numpy as np
#librería embedding
from gensim.models import KeyedVectors
#keras permite dos APIs --> functional/sequential
from keras.models import Sequential
#keras afrece muchos tipos de capas. En nuetro caso DENSE
from keras.layers.core import Dense


################################
###Embedding
################################

#Importación del fichero con los embeddings de google
EMBEDDING_FILE = "C:/Users/CTTC/Documents/mariaGregori/python/embeddings/" + 'SBW-vectors-300-min5.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,   binary=True)


################################
###Training dataset
################################

ayer = np.array(word2vec.word_vec('ayer'))
hoy =  word2vec.word_vec('hoy')
mañana =  word2vec.word_vec('mañana')

#input
data_input = np.empty((93,331,), "float32")
data_input[:] = np.NAN
#ayer
for i in range(31):
    zeros = np.zeros(31, "float32")
    zeros[i]=1
    data_input[i] = np.concatenate((ayer, zeros), axis=0)
#hoy
#for i in range(31):
    zeros = np.zeros(31, "float32")
    zeros[i]=1
    data_input[i+31] = np.concatenate((hoy, zeros), axis=0) 
#mañana
#for i in range(31):
    zeros = np.zeros(31, "float32")
    zeros[i]=1
    data_input[i+62] = np.concatenate((mañana, zeros), axis=0) 

#output
data_output = np.empty((93,31,), "float32")
data_output[:] = np.NAN
#ayer
for i in range(31):
    zeros = np.zeros(31, "float32")
    index = np.mod(30+i,31)
    zeros[index]=1
    data_output[i] = zeros
#hoy
#for i in range(31):
    zeros = np.zeros(31, "float32")
    zeros[i]=1
    data_output[i+31] = zeros
#mañana
#for i in range(31):
    zeros = np.zeros(31, "float32")
    index = np.mod(i+1,31)
    zeros[index]=1
    data_output[i+62] = zeros


################################
###Neural Network - 2 hidden layer
################################

#inicialización del modelo    
model = Sequential()
#hidden layer 1
model.add(Dense(500, input_dim=331, activation='relu'))
#hidden layer 2
model.add(Dense(150, input_dim=500, activation='relu'))
#output layer sin especificar la dimensión de la entrada
model.add(Dense(31, activation='softmax'))
#características del modelo que estamos utilizando
model.compile(loss='categorical_crossentropy',
              optimizer='adam')

#entrenamiento
model.fit(data_input, data_output, nb_epoch=7000, verbose=2)



########################
###Test Neural Network
########################

#función nativa de predict --> Test de la red neuronal
test = np.array(word2vec.word_vec('mañana'))
zeros = np.zeros(31, "float32")
zeros[1]=1
set7 = np.concatenate((test, zeros), axis=0)
temp = np.array([set7])
tmp = model.predict(temp).round()
tmp = model.predict(temp)
print (tmp) #En la realidad seria el test-set no del trainning

