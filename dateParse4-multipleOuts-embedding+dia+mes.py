# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:00:02 2017

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
x = Dense(500, activation='relu')(inputs) # this is your network, let's say you have 2 hidden layers of 64 nodes each (don't know if that's enough for you)
x = Dense(700, activation='relu')(x)
x = Dense(900, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
x = Dense(700, activation='relu')(x)
x = Dense(500, activation='relu')(x)
x = Dense(300, activation='relu')(x)
x = Dense(150, activation='relu')(x)

output1 = Dense(31, activation='softmax')(x) # now you create an output layer for each of your K groups. And each output has M elements, out of which because of 'softmax' only 1 will be activated. (practically this is of course a distribution, but after sufficient training, this usually makes one element close to one and the other elements close to zero)
output2 = Dense(12, activation='softmax')(x)


#model = Model(input=inputs, output=output1)
model = Model(input=inputs, output=[output1, output2])


model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy')



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

#for z in range(5000):
for j in range(12):
    j
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
            if j == 11:  
                mes2[0] = 1
                data_output2[i+62] = mes2
            else:
                index = j+1
                mes2[index] = 1
                data_output2[i+62] = mes2                       
        else:
            data_output2[i+62] = mes


    
###############################
###Training Neural Network
###############################
  
    #model.fit(data_input, data_output, nb_epoch=7000, batch_size=2)
    if j ==0:
         data_all = np.array(data_input)
         data_out_all = np.array(data_output)
         data_out_all2 = np.array(data_output2)
    else:        
        data_all = np.concatenate((data_all, data_input),axis=0)
        data_out_all = np.concatenate((data_out_all, data_output),axis=0)
        data_out_all2 = np.concatenate((data_out_all2, data_output2),axis=0)
        
model.fit(data_all, [data_out_all, data_out_all2], nb_epoch=10000, batch_size=1116)



"""
, batch_size=7


one epoch = one forward pass and one backward pass of all the training examples
batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
"""

#entrenamiento
#model.fit(data_input, data_output, nb_epoch=7000, verbose=2)


########################
###Test Neural Network
########################

#función nativa de predict --> Test de la red neuronal
embedding = np.array(word2vec.word_vec('mañana'))
dia = np.zeros(31, "float32")
mes = np.zeros(12, "float32")
dia[3]=1
mes[3] = 1
set7 = np.concatenate((embedding, dia, mes), axis=0)
temp = np.array([set7])
tmp = model.predict(temp)
dia = tmp[0]
mes = tmp[1]
aux = dia.round()
aux1 = mes.round()
print (np.argmax(aux)) #En la realidad seria el test-set no del trainning
print (np.argmax(aux1))
