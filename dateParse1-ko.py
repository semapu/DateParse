# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:14:44 2017

@author: CTTC
"""


import numpy as np
from gensim.models import KeyedVectors

#keras permite dos APIs --> functional/sequential
from keras.models import Sequential
#keras afrece muchos tipos de capas. En nuetro caso DENSE
from keras.layers.core import Dense


EMBEDDING_FILE = "C:/Users/CTTC/Documents/mariaGregori/python/embeddings/" + 'SBW-vectors-300-min5.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,   binary=True)


#inicialización del modelo    
model = Sequential()
#hidden layer 1
model.add(Dense(31, input_dim=331, activation='relu'))

#hidden layer 12
#model.add(Dense(60, input_dim=100, activation='relu'))

#output layer sin especificar la dimensión de la entrada
model.add(Dense(31, activation='sigmoid'))

#proceso de aprendizaje
model.compile(loss='mean_squared_error',
              optimizer='adam')




ayer = np.array(word2vec.word_vec('ayer'))
hoy =  word2vec.word_vec('hoy')
mañana =  word2vec.word_vec('mañana')

for j in range(30):
    for i in range(31):
        
        #input
        zeros = np.zeros(31, "float32")
        zeros[i]=1
        set1 = np.concatenate((ayer, zeros), axis=0)
    
        """zeros = np.zeros(31, "float32")
        zeros[i]=1
        set2 = np.concatenate((hoy, zeros), axis=0)
      
        zeros = np.zeros(31, "float32")
        zeros[i]=1
        set3 = np.concatenate((mañana, zeros), axis=0)"""
    
        data_input = np.array([set1], "float32")
    
        #output
        #ayer
        zeros = np.zeros(31, "float32")
        index = np.mod(30+i,31)
        zeros[index]=1
        set4 = zeros
        """#hoy
        zeros = np.zeros(31, "float32")
        zeros[i]=1
        set5 = zeros
        #mañana
        zeros = np.zeros(31, "float32")
        index = np.mod(i+1,31)
        zeros[index]=1
        set6 = zeros"""
        
        data_output = np.array([set4], "float32")
        
        #entrenamiento
        model.fit(data_input, data_output, nb_epoch=100, verbose=2)








########################
###Test Neural Network
########################

#función nativa de predict --> Test de la red neuronal
test = np.array(word2vec.word_vec('ayer'))
zeros = np.zeros(31, "float32")
zeros[0]=1
set7 = np.concatenate((test, zeros), axis=0)
temp = np.array([set7])
tmp = model.predict(temp).round()
print (tmp) #En la realidad seria el test-set no del trainning








