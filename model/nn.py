from __future__ import print_function
import sys
import numpy as np
from scipy import stats
from keras import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
#import tensorflow as tf
#from tensorboard.plugins.hparams import api as hp
# my wavenet model
def genre_model(num_genres, input_shape):
	input = Input(shape=input_shape)
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
	model.summary()
	return(model)
# get index integer value that nn predicts
def predict_song_index(song,model):
	pred = model.predict(song)
	#print(pred)
	pred = np.argmax(pred,axis =1)
	#print(pred)
	#print(sort(pred))
	return stats.mode(pred)[0]
# predict genre for a single song (genre_list is possible genres to pick from)
def predict_song(song,genres_list,transform_song,sr,time_length,model):
	song = transform_song(song,sr,time_length)
	index = predict_song_index(song,model)
	return genres_list[index][0]
# determine the accuracy of model given test data songs, genres
def test_model(songs,genres,genres_list,transform_song,sr,time_length,model):
	print(' Testing model on full length songs', end='')
	correct = 0
	i = 0
	for song in songs:
		genre = predict_song(song,genres_list,transform_song,sr,time_length,model)
		if genre == genres[i]:
			correct = correct+1
		i = i+1
		print('.', end='')
		sys.stdout.flush()
	print('\n Test accuracy for full length songs: ', correct/genres.shape[0])