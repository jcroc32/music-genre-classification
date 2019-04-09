from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras import layers, Input
from pandas import get_dummies
import numpy as np
from model.load import load_data
from model.nn import genre_model
from model.transform_data import transform_song
from scipy import stats

sr = 22050
songs, genres = load_data(sr)
x_train, x_test, y_train, y_test = train_test_split(songs, genres, stratify=genres, test_size=0.25, random_state=50)
songs = 0
genres = np.loadtxt('model/data/genres.txt', dtype=str, delimiter=" ")
'''
model_ = 10*[0]
input_ = 10*[0]
output_ = 10*[0]
for i in range(10):
	model_[i] = genre_model(10, (66150,1))
	output_[i] = model_[i].output
	input_[i] = model_[i].input

print(x_train.shape)
print(x_train[0].shape)

x_test_ = 10*[0]
for j in range(10):
	temp = []
	for i in x_test:
		temp.append(i[j*66150:(j+1)*66150])
	x_test_[j] = np.asarray(temp)
	x_test_[j] = np.expand_dims(x_test_[j], axis=3)

print(x_test_[0].shape)
#input_layer = layers.Lambda(lambda x: x)(input)
#input_layer = layers.concatenate(input_,axis=-1)(input_layer)
	
output = layers.concatenate(output_,axis=-1)
output = layers.Dense(10,activation='softmax')(output)
model = Model(input_,output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

loss, accuracy = model.evaluate(x_test_, y_test)
print('test accuracy:', accuracy)
print('test loss:', loss)
'''

model=load_model('model/model.h5')

sum = 0
i = 0
for x in x_test:
	x = transform_song(x)
	pred = model.predict(x)
	n = np.argmax(pred,axis =1)
	n = stats.mode(n)[0]
	if genres[n[0]] == y_test[i]:
		sum = sum+1
	i = i+1
print(sum)
print(sum/y_test.shape[0])
