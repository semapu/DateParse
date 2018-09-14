# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:46:57 2017

@author: CTTC
"""

#Librerias a importar
import numpy as np
#keras permite dos APIs --> functional/sequential
from keras.models import Sequential
#keras afrece muchos tipos de capas. En nuetro caso DENSE
from keras.layers.core import Dense

#input
data_input = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

#output
data_output = np.array([[0],[1],[1],[0]], "float32")

#inicialización del modelo
model = Sequential()
#hidden layer
model.add(Dense(16, input_dim=2, activation='relu'))
#output layer sin especificar la dimensión de la entrada
model.add(Dense(1, activation='sigmoid'))

#proceso de aprendizaje
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

#entrenamiento
model.fit(data_input, data_output, nb_epoch=800040, verbose=2)

#evaluamos el modelo
loss_and_metrics = model.evaluate(data_input, data_output)

#función nativa de predict

print (model.predict(data_input).round()) #En la realidad seria el test-set no del trainning

#obtención de los pesos
for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays
#print(weights)
   
weights1 = np.array(model.layers[0].get_weights())
print("\nPesos entre la INPUT y la HIDDEN layer")
print(weights1)

weights2 = np.array(model.layers[1].get_weights())
print("\nPesos entre la HIDDEN y la OPUTPUT layer")
print(weights2)
