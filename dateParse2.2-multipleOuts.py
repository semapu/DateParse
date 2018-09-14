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
#from keras.models import Sequential

#keras afrece muchos tipos de capas. En nuetro caso DENSE
#from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Input, Dense


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

inputs = Input(shape=(331,))
x = Dense(500, activation='relu')(inputs) # this is your network, let's say you have 2 hidden layers of 64 nodes each (don't know if that's enough for you)
x = Dense(150, activation='relu')(x)

output1 = Dense(31, activation='softmax')(x) # now you create an output layer for each of your K groups. And each output has M elements, out of which because of 'softmax' only 1 will be activated. (practically this is of course a distribution, but after sufficient training, this usually makes one element close to one and the other elements close to zero)
#output2 = Dense(12, activation='softmax')(x)


model = Model(input=inputs, output=output1)
#model = Model(input=inputs, output=[output1, output2])


model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

model.fit(data_input, data_output, nb_epoch=7000, batch_size=2)
#model.fit(data_input, [outputData1, outputData2], nb_epoch=7000, batch_size=64)



########################
###Test Neural Network
########################

#función nativa de predict --> Test de la red neuronal
test = np.array(word2vec.word_vec('mañana'))
zeros = np.zeros(31, "float32")
zeros[30]=1
set7 = np.concatenate((test, zeros), axis=0)
temp = np.array([set7])
tmp = model.predict(temp).round()
print (tmp) #En la realidad seria el test-set no del trainning
