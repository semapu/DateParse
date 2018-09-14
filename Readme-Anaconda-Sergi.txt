Creación de un entorno con Anaconda (Anaconda Prompt) que permita trabajar con tensorflow y keras:

TestSergi2 --> entorno preparado para tensorflow i keras:
	https://www.dataweekends.com/blog/2017/03/09/set-up-your-mac-for-deep-learning-with-python-keras-and-tensorflow
	
	1) conda create -n SergiTest2 python=3.5 pandas scikit-learn jupyter matplotlib 
	1.1) activate SergiTest2
	2) pip install tensorflow 
	3) pip install keras 
	4) ipython 
	5) Parecido a:
		Python 2.7.13 |Continuum Analytics, Inc.| (default, Dec 20 2016, 23:05:08) Type "copyright", "credits" or "license" for more information. IPython 5.3.0 -- An enhanced Interactive Python. ? -> Introduction and overview of IPython's features. %quickref -> Quick reference. help -> Python's own help system. object? -> Details about 'object', use 'object??' for extra details. details. In [1]: 

	6) import keras, tensorflow


activate SergiTest2
conda info --envs
conda list

-------------------------------------------

spyder:
model.save --> kguarder el pesos
	model.save('filepath')
	model.save_weights('my_model_weights.h5')

model.load --> carregar pesos xarxa ja entrenada
	models.load_model(filepath)
	model.load_weights('my_model_weights.h5')

----------------------------------------------

https://en.wikipedia.org/wiki/Activation_function

------------------------------------------------

model.add(Dense(31, activation='sigmoid'))
#características del modelo que estamos utilizando
model.compile(loss='mean_squared_error',
              optimizer='adam')


------------------------------------------------








































