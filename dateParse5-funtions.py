"""
!!!LA TABLA ES EL ÚNICO ELEMENTO QUE SE DEBE INTRODUCIR. !!!
DEBERÍA SE UN 'CSV' CON LAS DIFERENTES VARIACIONES. (importar fichero excel)
    
    name    offset
    hoy     0
    ayer    -1
    mañana  1
    
EN PRINCIPIO SOLO HAY QUE EJECUTAR DE FORMA GENERICA EL CÓDIGO (F6)
"""


###Librerias a importar necesarias
import numpy as np #linear algebra
import pandas as pd
from gensim.models import KeyedVectors #embedding
from keras.models import Model
from keras.layers import Input, Dense



################################
###Embeddings 
################################u

#PATH del fichero para los embeddings y creación de la variable asociada
EMBEDDING_FILE = "C:/Users/CTTC/Documents/mariaGregori/python/embeddings/" + 'SBW-vectors-300-min5.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,   binary=True)




################################
###Dataset creation & Learning
################################

###LA TABLA ES EL ÚNICO ELEMENTO QUE SE DEBE INTRODUCIR. DEBERÍA SE UN 'CSV' CON LAS DIFERENTES VARIACIONES.
#La table hace de símil a un fechero excel en el cual se recogeran todos los posibles casos/variaciones
table = pd.DataFrame(columns=['name', 'offset'])

table.loc[0,'name']='hoy'
table.loc[0,'offset']=0

table.loc[1,'name']='ayer'
table.loc[1,'offset']=-1

table.loc[2,'name']='mañana'
table.loc[2,'offset']=1

L = table.shape[0]

#Establecer los días de cada mes. Tener en cuenta los años bisiestos
dias_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def embedding(word):
    #Permite obtener el embedding de una palabra
    #Inputs: 
        #string a convertir    
    #Outpet: 
        #vector de dimensión 300
        
    embedding = np.array(word2vec.word_vec("word"))
    return embedding

def dataIn(length, day, month):
    #Creación del dataset(input). Replica del calendaria (se contempla la diferencia de días entre meses).
    #Inputs: 
        #length: langitud de la tabla introducida
        #day: indice(vector) del dia que se quiere incorporar a los datos
        #month: indice(vector) del mes que se quiere incorporar a los datos    
    #Outpet: 
        #matriz 3x300+díasMes+12
        #Incluye ayer, hoy, mañana más el dia que se quiere incorporar.
        
    mes = np.zeros(12, "float32")
    mes[month-1]=1
    dia = np.zeros(31, "float32")
    dia[day-1]=1    
    
    data_in = np.zeros([length,343])

    for i in range(length):
        data_in[i] = np.concatenate((embedding(table.values[i,0]), dia, mes), axis=0)
    return data_in

def dataOut(day, month, days_month):
    #Creación del dataset(output) - Con las variaciones temporales en función del estudio realizado.
    #Inputs: 
        #day: indice del dia que se quiere incorporar a los datos
        #month: indice del mes que se quiere incorporar a los datos   
        #days_month: vector completo con los días del mes.
    #Outpet: 
        #data_output: matriz correspomndiente a los días (para ayer, hoy i mañana)
        #data_output2: matriz correspomndiente a los meses (para ayer, hoy i mañana)

    data_output = np.empty((L,31,), "float32")
    data_output2 = np.empty((L,12,), "float32")

    for i in range(L):            
        dia = np.zeros(31, "float32")
        index_dia = np.mod(day+table.values[i,1]-1,days_month[month-1]-1)
        print(days_month[month-1]-1)
        
        offset_mes = np.floor((day-1+table.values[i,1])/days_month[month-1])
        print(offset_mes)
        mes2 = np.zeros(12, "float32")
        index_mes = int(np.mod(month-1+offset_mes,12))
        print('a')
        print(index_dia)
        if index_dia> days_month[index_mes]-1:
            index_dia = days_month[index_mes]-1
            print('b')
            print(index_dia)
            
        dia[index_dia]=1
        data_output[i,:] = dia         
       
        mes2[index_mes] = 1
        data_output2[i,:] = mes2  
    
    return data_output, data_output2
 
    
def dataset(table):
    #Creación del dataset de entrenamiento a partir de la tabla generada o introducida. IMPORTANTE RESPETAR LA ESTRUCTURA.
    #Inputs: 
        #table: tabla 'dataFrame' con dos columnas
            #primera columna: expresion temporal
            #segunda columna: offset asociado a la expresión temporal
    #Outpet: 
        #data_all: Matriz que contine todos los datos de input de la red neuronal (año completo)
        #data_out_all; Matriz que contine todos los datos de output (días) de la red neuronal. Traslación del día en función de si se pregunta por: hoy, ayer, mañana
        #data_out_all2: Matriz que contine todos los datos de output (meses) de la red neuronal. Traslación del día en función de si se pregunta por: hoy, ayer, mañana

    #Inicialització
    #input
    data_input = np.empty((93,343,), "float32")
    data_input[:] = np.NAN
    #output
    data_output = np.empty((93,31,), "float32")
    data_output[:] = np.NAN
    #output
    data_output2 = np.empty((93,12,), "float32")
    data_output2[:] = np.NAN   
       
    for j in range(12):       
        for i in range(dias_mes[j]):
            if i == 0:
                data_all_month = dataIn(L,i+1,j+1)
            else:
                data_all_month = np.concatenate((data_all_month, dataIn(L,i+1,j+1)),axis=0)
                
            if i == 0:
                out = dataOut(i, j, dias_mes)
                data_output=out[0]
                data_output2=out[1]                
            else:
                out = dataOut(i, j, dias_mes)
                data_output = np.concatenate((data_output, out[0]),axis=0)
                data_output2 = np.concatenate((data_output2, out[1]),axis=0)
 
        if j ==0:
            data_all = np.array(data_all_month)
            data_out_all = np.array(data_output)
            data_out_all2 = np.array(data_output2)
        else:        
            data_all = np.concatenate((data_all, data_all_month),axis=0)
            data_out_all = np.concatenate((data_out_all, data_output),axis=0)
            data_out_all2 = np.concatenate((data_out_all2, data_output2),axis=0)  
   
    return data_all,data_out_all, data_out_all2
    
    """
    one epoch = one forward pass and one backward pass of all the training examples
    batch size = the number of training examples in one forward/backward pass.
    """



########################
###Test Neural Network
########################

def predict(word, dia_in, mes_in):
    #Función que nos permite testear la fiabilidad de la red
    #Inputs: 
        #word: expresión temporal (hoy, ayer, mañana)
        #dia_in: día actual o sobre el que se quiere hacer referencia (del 1 al 31)   
        #mes_in: mes actual o sobre el que se quiere hacer referencia (del 1 al 12)
    #Outpet: 
        #dia_out: día resultante del estudio (del 1 al 31)
        #mes_out: mes resultante del estudio (del 1 al 12)
        
    embedding = np.array(word2vec.word_vec(word))
    dia = np.zeros(31, "float32")
    mes = np.zeros(12, "float32")
    dia[dia_in -1]=1 #Restamos 1 para que el usuario pueda introducir una fecha normal (no posición del vector)
    mes[mes_in -1] = 1 #Restamos 1 para que el usuario pueda introducir una fecha normal (no posición del vector)
    set7 = np.concatenate((embedding, dia, mes), axis=0)
    tmp = model.predict(np.array([set7]))
    dia = tmp[0]
    mes = tmp[1]
    dia_out=np.argmax(dia.round()) + 1 #El código retorna posición de un vector. Queremos el dia.
    mes_out=np.argmax(mes.round()) + 1 #El código retorna posición de un vector. Queremos el dia.
    return dia_out, mes_out



#########################################
###Neural Network
#########################################

#Creación de la red neuronal y especificaciones de la misma
inputs = Input(shape=(343,))
x = Dense(500, activation='relu')(inputs) 
x = Dense(700, activation='relu')(x)
x = Dense(900, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
x = Dense(700, activation='relu')(x)
x = Dense(500, activation='relu')(x)
x = Dense(300, activation='relu')(x)
x = Dense(150, activation='relu')(x)

output1 = Dense(31, activation='softmax')(x) 
output2 = Dense(12, activation='softmax')(x)

model = Model(input=inputs, output=[output1, output2])

model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy')

data_all,data_out_all, data_out_all2 = dataset(table)
#Entrenamiento de la red neuronal.
model.fit(data_all, [data_out_all, data_out_all2], nb_epoch=10000, batch_size=1116)

###A continuación se adjuntan los pesos del entrenamiento para no tener que volver a repetirlo.
#model.load_weights('weights-embedding+di+mes') 

predict("ayer", 1, 1)

