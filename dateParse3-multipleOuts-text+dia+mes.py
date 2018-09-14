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

#keras ofrece muchos tipos de capas. 
from keras.models import Model
from keras.layers import Input, Dense

"""
Multiple outs: https://groups.google.com/forum/#!topic/keras-users/cpXXz_qsCvA
"""
################################
###Embedding
################################

#Importación del fichero con los embeddings de google
EMBEDDING_FILE = "C:/Users/CTTC/Documents/mariaGregori/python/embeddings/" + 'SBW-vectors-300-min5.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,   binary=True)


#########################################
###Neural Network - 2 hidden layer
#########################################

inputs = Input(shape=(343,))
x = Dense(550, activation='relu')(inputs) # this is your network, let's say you have 2 hidden layers of 64 nodes each (don't know if that's enough for you)
x = Dense(200, activation='relu')(x)

output1 = Dense(31, activation='softmax')(x) # now you create an output layer for each of your K groups. And each output has M elements, out of which because of 'softmax' only 1 will be activated. (practically this is of course a distribution, but after sufficient training, this usually makes one element close to one and the other elements close to zero)
output2 = Dense(12, activation='softmax')(x)


#model = Model(input=inputs, output=output1)
model = Model(input=inputs, output=[output1, output2])


model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])



################################
###Training dataset
################################

ayer = np.array(word2vec.word_vec('ayer'))
hoy =  word2vec.word_vec('hoy')
mañana =  word2vec.word_vec('mañana')

#Inicialització
data_input = np.empty((93,343,), "float32")
data_input[:] = np.NAN
#output
data_output = np.empty((93,31,), "float32")
data_output[:] = np.NAN
#output
data_output2 = np.empty((93,12,), "float32")
data_output2[:] = np.NAN


for j in range(12):
    mes = np.zeros(12, "float32")
    mes[j]=1

    #inout
    
    for i in range(31):
    #ayer
        #input
        dia = np.zeros(31, "float32")
        dia[i]=1
        data_input[i] = np.concatenate((ayer, dia, mes), axis=0)
        
        #output
        dia = np.zeros(31, "float32")
        index = np.mod(30+i,31)
        dia[index]=1
        data_output[i] = dia
        
        #output2
        mes2 = np.zeros(12, "float32")
        
        if i == 0:
            index = np.mod(j-1,12)
            mes2[index] = 1
            data_output2[i] = mes2                       
        else:
            data_output2[i] = mes
       
    #hoy
        #input
        dia = np.zeros(31, "float32")
        dia[i]=1
        data_input[i+31] = np.concatenate((hoy, dia, mes), axis=0) 
        
        #output
        zeros = np.zeros(31, "float32")
        zeros[i]=1
        data_output[i+31] = zeros
        
        #output2
        data_output2[i+31] = mes
    
    #mañana
        #input
        dia = np.zeros(31, "float32")
        dia[i]=1
        data_input[i+62] = np.concatenate((mañana, dia, mes), axis=0) 
        
        #output
        zeros = np.zeros(31, "float32")
        index = np.mod(i+1,31)
        zeros[index]=1
        data_output[i+62] = zeros
        
        #output2
        mes2 = np.zeros(12, "float32")
        
        if i == 30:
            index = j+1
            mes2[index] = 1
            data_output2[i+62] = mes2                       
        else:
            data_output2[i+62] = mes



###############################
###Training Neural Network
###############################
  
    #model.fit(data_input, data_output, nb_epoch=7000, batch_size=2)
    
    model.fit(data_input, [data_output, data_output2], nb_epoch=250, batch_size=2)


#entrenamiento
#model.fit(data_input, data_output, nb_epoch=7000, verbose=2)


########################
###Test Neural Network
########################

#función nativa de predict --> Test de la red neuronal
embedding = np.array(word2vec.word_vec('ayer'))
dia = np.zeros(31, "float32")
mes = np.zeros(12, "float32")
dia[30]=1
mes[2] = 1
set7 = np.concatenate((embedding, dia, mes), axis=0)
temp = np.array([set7])
tmp = model.predict(temp)
dia = tmp[0]
mes = tmp[1]
print (dia.round()) #En la realidad seria el test-set no del trainning
print (mes.round())
