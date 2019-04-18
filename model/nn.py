import numpy as np
from scipy import stats
from keras import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
# my wavenet model
def genre_model(num_genres, input_shape):
	input = Input(shape=input_shape)
	model = BatchNormalization()(input)
	model = Conv1D(filters=128, kernel_size=9, activation='relu', dilation_rate=1)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=128, kernel_size=9, activation='relu', dilation_rate=2)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=64, kernel_size=9, activation='relu', dilation_rate=4)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=64, kernel_size=9, activation='relu', dilation_rate=8)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=32, kernel_size=9, activation='relu', dilation_rate=16)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=32, kernel_size=7, activation='relu', dilation_rate=32)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Flatten()(model)
	model = Dropout(0.25)(model)
	output = Dense(num_genres, activation='softmax')(model)
	model = Model(input,output)
	return(model)

def predict_song_index(song,model):
	pred = model.predict(song)
	pred = np.argmax(pred,axis =1)
	return stats.mode(pred)[0]
	
def test_model(songs,genres,genres_list,transform_song,sr,time_length,model):
	correct = 0
	i = 0
	for song in songs:
		song = transform_song(song,sr,time_length)
		index = predict_song_index(song,model)
		if genres_list[index[0]] == genres[i]:
			correct = correct+1
		i = i+1
	print('test accuracy: ', correct/genres.shape[0])

# old model using Fourier transform and transfer learning
from keras.models import Sequential
from keras.applications.vgg16 import VGG16

def old_genre_model(num_genres, input_shape, freezed_layers = 5):
	input_tensor = Input(shape=input_shape)
	vgg16 = VGG16(include_top=False, weights='imagenet',input_tensor=input_tensor)
	top = Sequential()
	top.add(Flatten(input_shape=vgg16.output_shape[1:]))
	top.add(Dense(256, activation='relu'))
	top.add(Dropout(0.5))
	top.add(Dense(num_genres, activation='softmax'))
	model = Model(inputs=vgg16.input, outputs=top(vgg16.output))
	for layer in model.layers[:freezed_layers]:
		layer.trainable = False
	return model