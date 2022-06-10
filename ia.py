import os
import time
import requests
import discord
import keras
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#for analyses
from tensorflow.keras.callbacks import TensorBoard
#win
#tensorboard --logdir=logs/
#unix
#tensorboard --logdir logs/

EPOCHS=25
debug = False

categories =  ["20 km/h", "30 km/h", "50 km/h", "60 km/h", "70 km/h", "80 km/h", "Fim de zona de ultrapassagem, limite antigo nao vale mais", 
"100 km/h", "120 km/h", "proibido ultrapassar","proibido caminhao ultrapassar", "Prioridade na interseccao", "Estrada de prioridade", "Preferencia",
"Pare", "Red circle", "Proibido caminhoes", "Entrada proibida", "Atencao", "Atencao a esquerda", "Atencao a direita",
"zig-zag" ,"Estrada ruim a frente", "Estrada escorregadia", "uniao de pistas", "Obras", "Farol a frente", "Faixa de pedestres a frente",
"Atencao, criancas", "Atencao, bicicletas", "Perigo neve", "Animais silvestres", "Fim de zona de ultrapassagem",
"Virada obrigatoria a direita", "Virada obrigatoria a esquerda", "Obrigatorio ir reto", "Reto ou direita", "Reto ou esquerda",
"Passagem obrigatoria a direita", "Passagem obrigatoria a esquerda", "Rotatoria", "Fim de zona proibida ultrapassagem para carros",
"Fim de zona proibida para ultrapassagem de caminhoes"]

def prepare_image(file_path):
	data = []
	image = Image.open(file_path)
	image = image.resize((30, 30))
	data.append(np.array(image))
	return np.array(data)

def load(model_name):
	model = load_model(model_name)
	return model

def predict(model, img_path):
	prediction = model.predict([prepare_image(img_path)])
	label = np.argmax(prediction)
	if debug:
		print(f"LABEL = {label}")
		print(f"CATEGORIE = {categories[label]}")
	return categories[label]

def test_accuracy(c_model):
	model = c_model
	data = []
	y = pd.read_csv('database/Train.csv')
	labels = y["ClassId"].values
	images = y["Path"].values
	for img in images:
	  image = Image.open(f"database/{img}")
	  image = image.resize((30, 30))
	  data.append(np.array(image))
	X = np.array(data)
	p_x = model.predict(X)

	data = []
	y = pd.read_csv('database/Test.csv')
	labels = y["ClassId"].values
	images = y["Path"].values

	for img in images:
		image = Image.open(f"database/{img}")
		image = image.resize((30, 30))
		data.append(np.array(image))
		X = np.array(data)
	p_x = model.predict(X)
	model.save('models/classifier.keras')


def create_training_data():
	data = []
	labels = []
	classes = 43
	error = False
	print("\nCreating training data..\n")
	for i in range(classes):
		path = os.path.join(os.getcwd(), 'database/Train/', str(i))
		images = os.listdir(path)
		for pic in images:
			try:
				img = Image.open(f"{path}/{pic}")
				#img = img.convert('L')
				img = img.resize((30,30))
				img = np.array(img)
				data.append(img)
				labels.append(i)
			except:
				error = True
				print(f"Image {pic} failed to load..")

	if not error:
		print("All files loaded OK.\n")

	data = np.array(data)
	labels = np.array(labels)

	return data, labels
			

def train(model_name):
	log_name = f"traffic-{int(time.time())}"
	tensorboard = TensorBoard(log_dir=f"logs/{log_name}")
	data, labels = create_training_data()

	X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

	y_train = to_categorical(y_train, 43) 
	y_test = to_categorical(y_test, 43) 

	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
	model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
	model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())

	#helps avoiding overfitting
	model.add(Dropout(rate=0.20))

	model.add(Dense(256, activation='relu'))
	model.add(Dense(43, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

	new_model = model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=[tensorboard])

	model.save(model_name)	
	model.summary()
